import argparse
import json
import sys
import warnings
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from polygon import ReferenceClient, StocksClient

warnings.filterwarnings("ignore")

# Scaling constants for calculating points
# see docs/scoring.md for more info
ROI_SCALE = 6              # typical range 10-25
DIVERSITY_SCALE = 12        # typical range 6-12
CLI_SAT_SCALE = 20          # typical range 6-15
RAR_SCALE = 12              # typical range 8-15
DRAWDOWN_SCALE = 3          # typical range 3-8
TAIL_RISK_SCALE = 7        # enable 4-8 when used
REGIME_ROBUSTNESS_SCALE = 0 # enable 6-10 when used
RANDOM_SCALE = 2            # typical range 0-3
SKEWNESS_SCALE = 0          # typical range 0-4
ENTROPY_SCALE = 0           # if entropy method: 6-12
ROI_TRANSFORM: str | None = None  # Options: None | "log" | "sqrt" | "sigmoid"
DIVERSITY_METHOD: str = "mse"
ROI_FLOOR: float | None = None  # Minimum ROI after transform (None disables)
ROI_CEILING: float | None = None  # Maximum ROI after transform (None disables)

# Employment status configuration
UNEMPLOYED_RISK_FACTOR = 1.5

# RANDOM VARIABLE
RANDOM_MIN = -2.0  # Lower bound for random factor (before scaling)
RANDOM_MAX = 2.0   # Upper bound for random factor (before scaling)
RANDOM_SEED: int | None = None  # Optional fixed seed for reproducibility of random term

# Target Volatility configuration
CLIENT_SAT_TARGET_VOL_DEFAULT = 0.08

# Age tolerance configuration
AGE_YOUNG = 30
AGE_MID = 50
AGE_YOUNG_DIVISOR = 12  # (age - AGE_MIN) / AGE_YOUNG_DIVISOR for ramp up
AGE_OLD_DIVISOR = 20    # (AGE_MAX - age) / AGE_OLD_DIVISOR for decline

# Optional portfolio safety limits
MAX_STOCKS_LIMIT: int | None = None  # e.g. 25 means only first 25 stocks count
MAX_POINTS_LIMIT: float | None = None  # e.g. 10000 caps points to +/- 10000
MIN_UNIQUE_STOCKS: int | None = None  # e.g. 8 requires at least 8 distinct tickers for full points

# Early random scoring (simple toggle). If enabled, final points are replaced
# with a random float each run in the configured range.
EARLY_RANDOM_SCORE_ENABLED: bool = False
EARLY_RANDOM_SCORE_MIN = -20.0
EARLY_RANDOM_SCORE_MAX = 5.0

# DONT CHANGE
DEBUG = True
AGE_MIN = 18
AGE_MAX = 80
AGE_TOL_DEFAULT = 0.6
EARLY_RANDOM_SCORE_DECIMALS = 1


@dataclass
class Context:
    """Context struct"""

    timestamp: str
    start: str
    end: str
    age: int
    employed: bool
    salary: float
    budget: float
    dislikes: set


def mse_from_ideal(weights_map: list[int]):
    if len(weights_map) == 1:
        # One stock is NOT a portfolio, punish
        weights_map.append(10000000)

    weights = np.array(weights_map, dtype=float)
    n = weights.size

    if n == 0:
        return 0.0

    ideal = weights.sum() / n
    mse = np.mean((weights - ideal) ** 2)

    return mse


def get_tickers_agg_bars(
    client: StocksClient, data: List[Tuple[str, int]], start: str, end: str
) -> pd.DataFrame | None:
    df = pd.DataFrame(None, columns=["ticker", "v", "vw", "o", "c", "h", "l", "t", "n"])
    problematic_tickers = []
    for ticker, qty in data:
        bars = client.get_aggregate_bars(
            ticker, from_date=start, to_date=end, timespan="day"
        )
        if "results" not in bars:
            problematic_tickers.append(ticker)
            continue
        tmpdf = pd.DataFrame(bars["results"])
        tmpdf["ticker"] = ticker
        tmpdf["qty"] = qty

        df = pd.concat([df, tmpdf])
    df.set_index(["t", "ticker"], inplace=True)
    df["value"] = df["c"] * df["qty"]
    return df, problematic_tickers


def init_price_breaches_threshold(df: pd.DataFrame, threshold: float) -> tuple[bool, float]:
    value = df.groupby(level=1).first()["value"].sum()
    return (value > threshold, value)


def is_industry_in_dislikes(
    stock: str,
    ticker_details: Dict[str, Dict],
    context: Context,
    sic_industry: Dict[str, List[str]],
) -> bool:
    stock_details = ticker_details[stock]
    sic_code = stock_details["results"]["sic_code"]
    for industry in sic_industry[sic_code]:
        if industry in context.dislikes:
            return True
    return False


def sharpe(
    df: pd.DataFrame,
    rates: pd.DataFrame,
    days: int = 252,
):

    # Group the values; this returns a Series
    values = df.groupby(level=0)["value"].apply(lambda x: x.sum())
    values.index = pd.to_datetime(values.index, unit="ms")

    # Convert the Series to a DataFrame
    values = values.to_frame(name="value")

    # Ensure both indexes are sorted
    values = values.sort_index()
    rates = rates.sort_index()

    nearest_idx = rates.index.get_indexer(values.index, method="nearest")

    values["returns"] = (values["value"] / values["value"].shift(1)) - 1
    daily_risk_free = ((1 + rates.iloc[nearest_idx]["value"] / 100) ** (1 / 252)) - 1
    values["risk_free_rate"] = daily_risk_free.values
    values["diff"] = values["returns"] - values["risk_free_rate"]

    value = values["diff"].mean() / values["diff"].std()
    value *= np.sqrt(days)
    return value


def sortino(
    df: pd.DataFrame,
    context: Context,
    rates: pd.DataFrame,
    days: int = 252,
):
    """
    Calculate the Sortino ratio which only considers downside volatility.
    """
    values = df.groupby(level=0)["value"].apply(lambda x: x.sum())
    values.index = pd.to_datetime(values.index, unit="ms")
    values = values.to_frame(name="value")
    values = values.sort_index()
    rates = rates.sort_index()
    nearest_idx = rates.index.get_indexer(values.index, method="nearest")

    values["returns"] = values["value"].pct_change()
    daily_risk_free = ((1 + rates.iloc[nearest_idx]["value"] / 100) ** (1 / 252)) - 1
    values["risk_free_rate"] = daily_risk_free.values
    values["excess_returns"] = values["returns"] - values["risk_free_rate"]

    # Only consider negative excess returns for downside risk
    downside_returns = values["excess_returns"].where(values["excess_returns"] < 0, 0)
    downside_deviation = downside_returns.std()

    mean_excess_return = values["excess_returns"].mean()
    # Avoid division by zero; if no downside volatility, default to Sharpe
    if downside_deviation == 0:
        return np.nan
    sortino_ratio = mean_excess_return / downside_deviation * np.sqrt(days)
    return sortino_ratio


def risk_profile(context: Context) -> float:
    """Risk profile returns a float from [0, 1.2] where a higher score indicates
    more a more risk_averse client"""

    def age_profile(age):
        x0 = 45  # inflection age
        k = 0.1  # steepness factor
        risk_tolerance_logistic = 1 / (1 + np.exp(k * (age - x0)))
        return risk_tolerance_logistic

    salary_budget_ratio = min(context.budget, context.salary) / max(1, context.salary)

    risk_factor = (age_profile(context.age) + salary_budget_ratio) / 2

    # FIXME: Think about employment_status harder.
    if not context.employed:
        risk_factor *= 1.2

    return risk_factor


def get_points(
    df: pd.DataFrame,
    profit: float,
    stocks: List[Tuple[str, int]],
    legal_stocks: list[tuple[str, int]],
    context: Context,
    unique_industries: Set[str],
    basedir: str,
    sic_industry: dict[str, list[str]],
    ticker_details: dict,
) -> int:
    n_allowed_industries = len(unique_industries) - len(context.dislikes)

    roi = return_on_investment(profit, context)
    diversity_mse = diversity_score(
        df,
        stocks,
        unique_industries,
        n_allowed_industries,
        sic_industry,
        ticker_details,
    )
    diversity_entropy = entropy_diversity_score(
        df,
        stocks,
        unique_industries,
        n_allowed_industries,
        sic_industry,
        ticker_details,
    ) if ENTROPY_SCALE != 0 or DIVERSITY_METHOD == "entropy" else 0.0
    client_sat = (
        client_satisfaction(df, risk_profile(context), context.age)
        if CLI_SAT_SCALE != 0
        else 0.0
    )
    rar = risk_adjusted_returns(df, context, basedir) if RAR_SCALE != 0 else 0.0
    drawdown = max_drawdown_score(df) if DRAWDOWN_SCALE != 0 else 0.0
    tail_risk = tail_risk_score(df) if TAIL_RISK_SCALE != 0 else 0.0
    regime_robustness = (
        regime_robustness_score(df) if REGIME_ROBUSTNESS_SCALE != 0 else 0.0
    )
    skewness = skewness_score(df) if SKEWNESS_SCALE != 0 else 0.0

    # Random additive factor (distinct from early override). Generates a value
    # in [RANDOM_MIN, RANDOM_MAX] each evaluation and scales it. Use seed for reproducibility.
    if RANDOM_SCALE != 0:
        rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random
        random_term = rng.uniform(RANDOM_MIN, RANDOM_MAX)
    else:
        random_term = 0.0

    if DEBUG:
        print(
            "[DEBUG] metrics:",
            f"roi={roi:.4f}",
            f"diversity_mse={diversity_mse:.4f}",
            f"diversity_entropy={diversity_entropy:.4f}",
            f"diversity_method={DIVERSITY_METHOD}",
            f"client_sat={client_sat:.4f}",
            f"rar={rar:.4f}",
            f"drawdown={drawdown:.4f}",
            f"tail_risk={tail_risk:.4f}",
            f"regime_robustness={regime_robustness:.4f}",
            f"skewness={skewness:.4f}",
            f"random_term={random_term:.4f}",
        )

    points = 0.0
    if ROI_SCALE != 0:
        points += ROI_SCALE * roi
    if DIVERSITY_METHOD == "mse" and DIVERSITY_SCALE != 0:
        points += DIVERSITY_SCALE * diversity_mse
    elif DIVERSITY_METHOD == "entropy" and ENTROPY_SCALE != 0:
        points += ENTROPY_SCALE * diversity_entropy
    if CLI_SAT_SCALE != 0:
        points += CLI_SAT_SCALE * client_sat
    if RAR_SCALE != 0:
        points += RAR_SCALE * rar
    if DRAWDOWN_SCALE != 0:
        points += DRAWDOWN_SCALE * drawdown
    if TAIL_RISK_SCALE != 0:
        points += TAIL_RISK_SCALE * tail_risk
    if REGIME_ROBUSTNESS_SCALE != 0:
        points += REGIME_ROBUSTNESS_SCALE * regime_robustness
    if RANDOM_SCALE != 0:
        points += RANDOM_SCALE * random_term
    if SKEWNESS_SCALE != 0:
        points += SKEWNESS_SCALE * skewness

    # Enforce minimum unique stocks requirement (multiplicative penalty if below)
    if MIN_UNIQUE_STOCKS is not None and MIN_UNIQUE_STOCKS > 0:
        unique_count = len({s for s, _ in stocks})
        if unique_count < MIN_UNIQUE_STOCKS:
            # Scale points down proportionally to coverage; avoid negative flip here
            scarcity_factor = unique_count / MIN_UNIQUE_STOCKS
            points *= max(0.0, scarcity_factor)

    value = df.groupby(level=1).first()["value"].sum()
    points *= 1 - (value / context.budget)

    points = points * 10  # make a nicer scale for points

    # Dock fixed amount of points, if illegal
    if len(legal_stocks) != len(stocks):
        diff = abs(len(legal_stocks) - len(stocks))
        profit = -(0.1 * diff) * profit if profit > 0 else profit * (1 + (0.1 * diff))
        points = -(0.1 * diff) * points if points > 0 else points * (1 + (0.1 * diff))
    # Final sign adjustment based on profit
    points = points if profit > 0 else -abs(points)

    # Apply max points safety cap if configured
    if MAX_POINTS_LIMIT is not None:
        points = max(-MAX_POINTS_LIMIT, min(MAX_POINTS_LIMIT, points))

    return points


def max_drawdown_score(df: pd.DataFrame) -> float:
    """Return a drawdown score in [0,1].

    Score is (1 - max_drawdown_fraction). A portfolio with 0% drawdown scores 1.0.
    A 50% max drawdown scores 0.5. A 100% drawdown scores 0.0.
    If insufficient data (<2 points) returns 0.5 as neutral.
    """
    try:
        equity = df.groupby(level=0)["value"].sum().astype(float)
    except Exception:
        return 0.5
    if equity.size < 2:
        return 0.5
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (equity - peak) / peak
    mdd = abs(np.nanmin(drawdowns)) if np.isfinite(np.nanmin(drawdowns)) else 0.0
    score = 1 - mdd
    return float(max(0.0, min(1.0, score)))


def tail_risk_score(df: pd.DataFrame) -> float:
    """Tail risk score in [0,1] using CVaR (Expected Shortfall) at 95%.

    Steps:
      1. Compute daily portfolio returns.
      2. Find 5th percentile (VaR_95).
      3. CVaR = mean of returns <= VaR_95 (expected tail loss).
    Mapping: If CVaR >= 0 (no negative tail), score = 1. Otherwise
    score = 1 + CVaR / 0.25 (assuming -25% is worst typical daily tail). Clamp to [0,1].
    Fallback neutral score 0.5 if insufficient data.
    """
    try:
        values = df.groupby(level=0)["value"].sum().astype(float)
        returns = values.pct_change().dropna()
    except Exception:
        return 0.5
    if returns.size < 5:
        return 0.5
    try:
        var_95 = np.quantile(returns, 0.05)
        tail = returns[returns <= var_95]
        if tail.size == 0:
            return 1.0
        cvar = tail.mean()
    except Exception:
        return 0.5
    if cvar >= 0:
        return 1.0
    score = 1 + (cvar / 0.25)  # cvar negative => subtract proportionally
    return float(max(0.0, min(1.0, score)))


def regime_robustness_score(df: pd.DataFrame, segments: int = 3) -> float:
    """Regime robustness score in [0,1].

    Split the evaluation window into `segments` equal chronological parts.
    Compute ROI per segment. Measure consistency via coefficient of variation.
    Score = 1 / (1 + CV). If overall ROI negative, halve score. Neutral 0.5 fallback.
    """
    try:
        # Use cumulative portfolio value series
        values = df.groupby(level=0)["value"].sum().astype(float)
    except Exception:
        return 0.5
    if values.size < segments + 1:
        return 0.5
    # Build segment boundaries via indices split
    idx_arrays = np.array_split(values.index, segments)
    rois: list[float] = []
    for idx in idx_arrays:
        if len(idx) < 2:
            continue
        start_val = float(values.loc[idx[0]])
        end_val = float(values.loc[idx[-1]])
        if start_val <= 0:
            continue
        roi_seg = (end_val - start_val) / start_val
        rois.append(roi_seg)
    if len(rois) < 2:
        return 0.5
    mean_roi = float(np.mean(rois))
    std_roi = float(np.std(rois))
    if mean_roi == 0:
        return 0.5
    cv = std_roi / (abs(mean_roi) + 1e-9)
    score = 1 / (1 + cv)
    overall_roi = (values.iloc[-1] - values.iloc[0]) / max(values.iloc[0], 1e-9)
    if overall_roi < 0:
        score *= 0.5
    return float(max(0.0, min(1.0, score)))


def skewness_score(df: pd.DataFrame) -> float:
    """Skewness score in [0,1].

    Uses daily portfolio returns skewness. Positive skew (occasional large upside
    moves) is rewarded; heavy negative skew (large downside tail) penalized.

    Calculation (unbiased Fisher/Pearson):
        skew = n/((n-1)*(n-2)) * sum(((x_i - mean)/std)^3)

    Mapping to score: 0.5 + 0.5 * tanh(skew).
        - Zero skew -> 0.5 neutral
        - Large positive skew -> tends to 1.0
        - Large negative skew -> tends to 0.0

    Fallback neutral 0.5 if insufficient data (<5 return observations) or undefined.
    """
    try:
        values = df.groupby(level=0)["value"].sum().astype(float)
        returns = values.pct_change().dropna()
    except Exception:
        return 0.5
    n = returns.size
    if n < 5:
        return 0.5
    mean = float(returns.mean())
    std = float(returns.std())
    if std == 0 or not np.isfinite(std):
        return 0.5
    normed = ((returns - mean) / std) ** 3
    m3 = float(normed.sum())
    skew = (n / ((n - 1) * (n - 2))) * m3
    score = 0.5 + 0.5 * np.tanh(skew)
    return float(max(0.0, min(1.0, score)))


def return_on_investment(profit: float, context: Context) -> float:
    """Return on investment is defined as the ratio of profit to initial budget"""
    # return np.log(profit) if profit > 0 else -np.log(abs(profit)) if profit < 0 else 0
    raw = profit / context.budget
    if ROI_TRANSFORM is None:
        roi_val = raw
    try:
        if ROI_TRANSFORM == "log":
            # Symmetric log compression
            roi_val = np.sign(raw) * np.log1p(abs(raw))
        elif ROI_TRANSFORM == "sqrt":
            roi_val = np.sign(raw) * np.sqrt(abs(raw))
        elif ROI_TRANSFORM == "sigmoid":
            # Use tanh for smooth saturation
            roi_val = np.tanh(raw)
        else:
            roi_val = raw
    except Exception:
        roi_val = raw

    # Apply optional floor/ceiling clamps if configured
    try:
        if ROI_FLOOR is not None and ROI_CEILING is not None and ROI_CEILING < ROI_FLOOR:
            # Safety swap if misconfigured
            floor, ceiling = ROI_CEILING, ROI_FLOOR
        else:
            floor, ceiling = ROI_FLOOR, ROI_CEILING
        if floor is not None:
            roi_val = max(floor, roi_val)
        if ceiling is not None:
            roi_val = min(ceiling, roi_val)
    except Exception:
        # On any error, leave roi_val unchanged
        pass
    return roi_val


def diversity_score(
    df: pd.DataFrame,
    stocks: List[Tuple[str, int]],
    unique_industries: set[str],
    n_allowed_industries: int,
    sic_industry: dict[str, list[str]],
    ticker_details: dict,
) -> float:
    """MSE-based diversification. Returns 0.0 on any failure."""
    try:
        stock_quantity_prods = df.groupby(level=1).first()["value"]
        unique_stocks = set([s for s, _ in stocks])
        if not unique_stocks:
            return 0.0
        stock_counts = [stock_quantity_prods[s] for s in unique_stocks]
        stocks_per_industry = stock_count_per_industry(ticker_details, stocks, sic_industry)
        stock_mse = mse_from_ideal(stock_counts)
        sic_mse = mse_from_ideal(list(stocks_per_industry.values()))
        stock_score = np.log(1 + len(stocks)) / (1 + len(unique_stocks) * stock_mse)
        sic_score = (len(unique_industries) / max(1, n_allowed_industries)) / (1 + sic_mse)
        value = stock_score + sic_score
        if not np.isfinite(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def entropy_diversity_score(
    df: pd.DataFrame,
    stocks: List[Tuple[str, int]],
    unique_industries: set[str],
    n_allowed_industries: int,
    sic_industry: dict[str, list[str]],
    ticker_details: dict,
) -> float:
    """Entropy-based diversification in [0,2].

    Computes normalized Shannon entropy for stock allocation and industry allocation.
    Stock entropy:
        H_stock = -Σ p_i log(p_i) / log(n)
    Industry entropy similar using quantity per industry.
    Sum of both entropies yields range [0,2].
    """
    try:
        stock_quantity_prods = df.groupby(level=1).first()["value"]
        unique_stocks = set([s for s, _ in stocks])
        if not unique_stocks:
            return 0.0
        stock_weights = np.array([stock_quantity_prods[s] for s in unique_stocks], dtype=float)
        sw_sum = stock_weights.sum()
        if sw_sum <= 0:
            return 0.0
        p = stock_weights / sw_sum
        n_stocks = p.size
        stock_entropy = 0.0
        for pi in p:
            if pi > 0:
                stock_entropy -= pi * np.log(pi)
        if n_stocks > 1:
            stock_entropy /= np.log(n_stocks)
        stocks_per_industry = stock_count_per_industry(ticker_details, stocks, sic_industry)
        industry_vals = np.array(list(stocks_per_industry.values()), dtype=float)
        iv_sum = industry_vals.sum()
        if iv_sum <= 0:
            industry_entropy = 0.0
        else:
            q = industry_vals / iv_sum
            n_inds = q.size
            industry_entropy = 0.0
            for qi in q:
                if qi > 0:
                    industry_entropy -= qi * np.log(qi)
            if n_inds > 1:
                industry_entropy /= np.log(n_inds)
        coverage_factor = len(unique_industries) / max(1, n_allowed_industries)
        industry_entropy *= coverage_factor
        value = stock_entropy + industry_entropy
        if not np.isfinite(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def stock_count_per_industry(
    ticker_details: dict,
    stocks: list[tuple[str, int]],
    sic_industry: dict[str, list[str]],
) -> dict[str, int]:
    """Returns how many stocks were invested per industry"""
    industry_count_map: dict[str, int] = {}

    # XXX: sic_industries map a code to a list of industries in a (theoretically)
    # many-many fashion. Visual inspection suggests that
    # len(sic_industries[TCKR]) is always one. This makes calculating
    # diversification harder. To workaround this, always assign stocks to the
    # **first industry** in the list.

    for tkr, quantity in stocks:
        sic_code = ticker_details[tkr]["results"]["sic_code"]
        industries = sic_industry[sic_code]

        if not industries:
            industry = "unmapped"
        else:
            industry = industries[0]

        if industry not in industry_count_map:
            industry_count_map[industry] = 0

        industry_count_map[industry] += quantity

    return industry_count_map


def client_satisfaction(df: pd.DataFrame, risk_profile: float, age: int | None = None, target_vol: float = CLIENT_SAT_TARGET_VOL_DEFAULT) -> float:
    """Client satisfaction score in [0,1] using returns-based volatility."""
    r = df["value"].pct_change().dropna()
    if r.size == 0:
        return 0.0
    portfolio_vol = r.std()
    if not np.isfinite(portfolio_vol):
        return 0.0

    rp = max(0.0, min(1.2, risk_profile))
    base_from_risk = 0.9 - 0.6 * (rp / 1.2)

    def age_tolerance(a: int | None) -> float:
        if a is None:
            return AGE_TOL_DEFAULT
        if a < AGE_MIN or a >= AGE_MAX:
            return 0.0
        if a <= AGE_YOUNG:
            return (a - AGE_MIN) / AGE_YOUNG_DIVISOR
        if a <= AGE_MID:
            return 1.0
        return (AGE_MAX - a) / AGE_OLD_DIVISOR

    combined_factor = max(0.05, base_from_risk * age_tolerance(age))
    tolerance = target_vol * combined_factor

    if portfolio_vol <= tolerance:
        return 1.0

    score = 1 - (portfolio_vol - tolerance) / max(tolerance, 1e-9)
    return float(max(0.0, min(1.0, score)))


def risk_adjusted_returns(
    df: pd.DataFrame,
    context: Context,
    basedir: str,
) -> float:
    """Calculate the risk-adjusted returns using Sharpe and Sortino ratios"""
    rates = pd.read_csv(f"{basedir}/bond-rate.csv")
    rates.set_index("date", inplace=True)
    rates.index = pd.to_datetime(rates.index)

    rar = sharpe(df, rates)
    s_ratio = sortino(df, context, rates)
    if np.isnan(s_ratio):
        risk_adjusted = rar
    else:
        risk_adjusted = (rar + s_ratio) / 2

    return risk_adjusted


def evaluate(
    df: pd.DataFrame,
    stocks: List[Tuple[str, int]],
    context: Context,
    sic_industry: Dict[str, List[str]],
    unique_industries: Set[str],
    ref_client: ReferenceClient,
    basedir: str,
) -> Tuple[bool, str, float, int]:
    # Check did not send multiple stocks
    if len(stocks) != len(set([s for s, _ in stocks])):
        return False, f"Error: duplicate tickers: {stocks}", 0.0, -1

    if min([i for _, i in stocks]) <= 0:
        return False, f"Error: invalid stock weight: {stocks}", 0.0, -1

    failure, value = init_price_breaches_threshold(df, context.budget)
    if failure:
        # Breached the max price that someone has
        return (
            False,
            f"Error: budget breached (your portfolio value: {value}, budget: {context.budget})",
            0.0,
            -1,
        )

    # Verify all tickers exist and grab details
    ticker_details = {}
    for stock, _ in stocks:
        details = ref_client.get_ticker_details(stock)
        if details["status"] != "OK":
            return False, f"Error: invalid ticker: {stock}", 0.0, -1
        elif details["results"]["type"] != "CS":
            return (
                False,
                f"Error: invalid ticker type: {stock} of type {details['results']['type']}",
                0.0,
                -1,
            )
        ticker_details[stock] = details

    # Remove stocks if they are not legal, so we only calculate using the legal stocks.
    legal_stocks = [
        (stock, qty)
        for stock, qty in stocks
        if not is_industry_in_dislikes(stock, ticker_details, context, sic_industry)
    ]

    profit = (
        df.groupby(level=1).last()["value"].sum()
        - df.groupby(level=1).first()["value"].sum()
    )

    # Early random scoring short-circuit: generate points immediately and skip metrics
    if EARLY_RANDOM_SCORE_ENABLED:
        rand_points = round(
            random.uniform(EARLY_RANDOM_SCORE_MIN, EARLY_RANDOM_SCORE_MAX),
            EARLY_RANDOM_SCORE_DECIMALS,
        )
        return True, "", profit, rand_points

    points = get_points(
        df,
        profit,
        stocks,
        legal_stocks,
        context,
        unique_industries,
        basedir,
        sic_industry,
        ticker_details,
    )

    return True, "", profit, points


def main(api_key: str, data: Dict[str, Union[List[Dict[str, int]], Any]]):
    stocks_client = StocksClient(api_key)
    ref_client = ReferenceClient(api_key)
    if "context" not in data:
        print("context not passed through")
        return

    with open(f"{args.basedir}/sic_industry.json", "r") as f:
        sic_industry = json.loads(f.read())
    with open(f"{args.basedir}/unique_industries.json", "r") as f:
        unique_industries = json.loads(f.read())

    context = Context(**data["context"])
    # Ensure dislikes is a set for faster membership tests & dedup
    if not isinstance(context.dislikes, set):
        try:
            context.dislikes = set(context.dislikes)
        except Exception:
            context.dislikes = set()

    stocks = []
    for stock in data["stocks"]:
        stocks.append([stock["ticker"], stock["quantity"]])

    # Enforce max stocks limit (ignore extras) before fetching data
    if MAX_STOCKS_LIMIT is not None and len(stocks) > MAX_STOCKS_LIMIT:
        stocks = stocks[:MAX_STOCKS_LIMIT]

    df, problematic_tickers = get_tickers_agg_bars(
        stocks_client,
        stocks,
        # done-todo: replace string slicing with proper parsing
        #   https://stackoverflow.com/questions/1941927/convert-an-rfc-3339-time-to-a-standard-python-timestamp
        start=context.start,
        end=context.end,
    )
    if df is None or len(problematic_tickers):
        print(
            json.dumps(
                {
                    "passed": False,
                    "profit": 0.0,
                    "points": -1.0,
                    "error": f"invalid ticker(s) passed in {stocks}. The error here either means you have passed in invalid ticker(s) OR the tickers are not valid for the time range provided. Please ensure that the ticker(s) is (are) trading publicly during the ENTIRE time frame provided. Additionally, make sure you are using the CANNONICAL TICKER. Problematic Stocks: {problematic_tickers}",
                }
            )
        )
        return

    passed, error_message, profit, points = evaluate(
        df, stocks, context, sic_industry, unique_industries, ref_client, args.basedir
    )

    print(
        json.dumps(
            {
                "passed": passed,
                "profit": profit,
                "points": 0 if np.isnan(points) else points,
                "error": error_message,
            }
        )
    )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--apikey", help="The polygon api key", required=True)
        parser.add_argument(
            "--basedir", help="Directory base to look for files.", default="./"
        )
        args = parser.parse_args()

        data = json.loads(sys.stdin.read())
        main(args.apikey, data)
    except Exception as e:
        print(e)
        sys.exit(1)