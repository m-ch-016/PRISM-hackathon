import argparse
import json
import sys
import warnings
import random
import logging
import os
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from polygon import ReferenceClient, StocksClient


warnings.filterwarnings("ignore")

# Scaling constants for calculating points
# see docs/scoring.md for more info
ROI_SCALE = 0              #
DIVERSITY_SCALE = 0        # typical range 6-12
CLI_SAT_SCALE = 15          # typical range 6-15
RAR_SCALE = 8              # typical range 8-15
DRAWDOWN_SCALE = 6          # typical range 3-8
TAIL_RISK_SCALE = 7      # enable 4-8 when used
REGIME_ROBUSTNESS_SCALE = 0 # enable 6-10 when used
RANDOM_SCALE = 2            # typical range 0-3
SKEWNESS_SCALE = 1          # typical range 0-4
ENTROPY_SCALE = 6           # if entropy method: 6-12
ROI_TRANSFORM: str = "sqrt"  # Options: None | "log" | "sqrt" | "sigmoid"
DIVERSITY_METHOD: str = "entropy"
ROI_FLOOR: float | None = None  # Minimum ROI after transform (None disables)
ROI_CEILING: float | None = None  # Maximum ROI after transform (None disables)

# High performing tickers (approx last 20y) that tend to dominate naive strategies.
# We apply a gentle multiplicative penalty to discourage blindly stuffing these.
# NOTE: This is a tuning surface; keep list short and update if regime changes.
TOP_PERFORMERS_LAST_20Y: list[str] = [
    "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "META", "GOOGL", "PLTR"
]
TOP_PERFORMER_PENALTY_ENABLED: bool = True
TOP_PERFORMER_PENALTY_MULTIPLIER: float = 0.8  # per top-performer stock (compounds)
TOP_PERFORMER_MAX_PENALTY_MULTIPLIER: float = 0.4  # floor so large counts don't erase score
TOP_PERFORMER_PENALIZE_NEGATIVE: bool = False  # keep False so losses aren't reduced (no benefit)

# Penalize portfolios that produce only a negligible positive profit ("farming" safety metrics).
NEAR_ZERO_ROI_THRESHOLD = 0.02          # 1% ROI threshold below which profit is considered negligible
NEAR_ZERO_ROI_PENALTY_FACTOR = 0.2      # Multiplicative penalty applied to points if ROI in (0, threshold)

# Ultra-low volatility penalty (prevents parking in ultra-safe assets to farm metrics)
LOW_VOL_PENALTY_ENABLED = True           # Toggle to enable ultra-low volatility penalty curve
LOW_VOL_MIN_FACTOR = 0.20                # Fraction of tolerance defining the lower band start (only penalize insanely safe)
LOW_VOL_FLOOR_SCORE = 0.6                # Satisfaction score at (near) zero volatility (less harsh)
LOW_VOL_PENALTY_RISK_AVERSION_THRESHOLD = 0.75  # If risk_profile >= this, suppress or soften low-vol penalty

# Employment status configuration
UNEMPLOYED_RISK_FACTOR = 2

# RANDOM VARIABLE
RANDOM_MIN = -2.0  # Lower bound for random factor (before scaling)
RANDOM_MAX = 2.0   # Upper bound for random factor (before scaling)
RANDOM_SEED: int | None = None  # Optional fixed seed for reproducibility of random term

# Target Volatility configuration (ANNUALIZED intuitive value)
CLIENT_SAT_TARGET_VOL_ANNUAL_DEFAULT = 0.035
TRADING_DAYS_PER_YEAR = 252
# Backward compatible alias (deprecated): if other code references the old name.
CLIENT_SAT_TARGET_VOL_DEFAULT = CLIENT_SAT_TARGET_VOL_ANNUAL_DEFAULT  # DEPRECATED alias

# Age tolerance configuration
AGE_YOUNG = 30
AGE_MID = 50
AGE_YOUNG_DIVISOR = 12  # (age - AGE_MIN) / AGE_YOUNG_DIVISOR for ramp up
AGE_OLD_DIVISOR = 20    # (AGE_MAX - age) / AGE_OLD_DIVISOR for decline

# Optional portfolio safety limits
MAX_STOCKS_LIMIT: int = 15  # static fallback upper bound on counted stocks
MAX_POINTS_LIMIT: float = 500  # e.g. 10000 caps points to +/- 10000
MIN_UNIQUE_STOCKS: int = 8  # static fallback minimum distinct tickers for full points
UNIQUE_PENALTY_EXPONENT: float = 1.8  # exponent >1 increases severity for concentrated portfolios

# Dynamic stock limit configuration (randomized per run to discourage hard-coding strategies).
# When enabled, runtime chosen values override the static constants above.
DYNAMIC_STOCK_LIMITS_ENABLED: bool = True
MAX_STOCKS_LIMIT_RANGE: tuple[int, int] = (12, 18)  # inclusive range for maximum stocks counted
MIN_UNIQUE_STOCKS_RANGE: tuple[int, int] = (6, 10)  # inclusive range for minimum unique required
DYNAMIC_LIMITS_SEED: int | None = None  # set an int for deterministic selection (e.g. during testing)

# Runtime-selected limits (populated in main()).
RUNTIME_MAX_STOCKS_LIMIT: int | None = None
RUNTIME_MIN_UNIQUE_STOCKS: int | None = None

# Early random scoring (simple toggle). If enabled, final points are replaced
# with a random float each run in the configured range.
EARLY_RANDOM_SCORE_ENABLED: bool = False
EARLY_RANDOM_SCORE_MIN = -20.0
EARLY_RANDOM_SCORE_MAX = 5.0

# Debug / logging configuration
DEBUG = True
LOG_FILE_DEFAULT = "prism_eval_debug.log"
LOG_MAX_BYTES = 2 * 1024 * 1024  # 2MB per file
LOG_BACKUP_COUNT = 3

AGE_MIN = 18
AGE_MAX = 80
AGE_TOL_DEFAULT = 0.6
EARLY_RANDOM_SCORE_DECIMALS = 1

def _init_logging():
    """Initialize rotating file logging if DEBUG is enabled.

    Safe to call multiple times (idempotent). Any exceptions are swallowed to
    avoid breaking evaluation. Stdout remains clean for JSON output; only the
    log file receives debug/info lines.
    """
    if not DEBUG:
        return
    try:
        log_path = LOG_FILE_DEFAULT
        # Create parent directory if user provided a path component
        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handler = RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root = logging.getLogger()
        # Prevent duplicate handlers
        if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
            root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        logging.debug("logging initialized (DEBUG=%s, file=%s)", DEBUG, log_path)
    except Exception:
        # Intentionally ignore logging setup failures
        pass


# Initialize logging immediately if DEBUG is True (no CLI flag required)
_init_logging()

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
    try:
        n_allowed_industries = len(unique_industries) - len(getattr(context, "dislikes", set()))
    except Exception:
        n_allowed_industries = len(unique_industries)

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
        client_satisfaction(
            df,
            risk_profile(context),
            context.age,
            CLIENT_SAT_TARGET_VOL_ANNUAL_DEFAULT,  # explicit annual volatility parameter
        )
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
        logging.debug(
            "metrics roi=%s diversity_mse=%s diversity_entropy=%s diversity_method=%s client_sat=%s rar=%s drawdown=%s tail_risk=%s regime_robustness=%s skewness=%s random_term=%s",
            f"{roi:.4f}",
            f"{diversity_mse:.4f}",
            f"{diversity_entropy:.4f}",
            DIVERSITY_METHOD,
            f"{client_sat:.4f}",
            f"{rar:.4f}",
            f"{drawdown:.4f}",
            f"{tail_risk:.4f}",
            f"{regime_robustness:.4f}",
            f"{skewness:.4f}",
            f"{random_term:.4f}",
        )
        logging.debug(
            "portfolio %s",
            ", ".join(f"{t}:{q}" for t, q in stocks) or "<empty>",
        )

    points = 0.0
    if ROI_SCALE != 0:
        # ROI now primary driver; scale raised. ROI = profit / budget (possibly transformed).
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

    # Previous logic zeroed points when utilization ~ 100% (points *= 1 - invested/budget).
    # Replace with mild idle cash penalty: up to 10% reduction if large uninvested cash.
    try:
        value = df.groupby(level=1).first()["value"].sum()
        utilization = value / max(1e-9, context.budget)
        idle_fraction = max(0.0, 1 - utilization)  # portion of budget not deployed
        # Apply at most 10% penalty linearly with idle fraction.
        points *= (1 - 0.10 * idle_fraction)
    except Exception:
        # If anything fails, skip penalty to avoid wiping ROI impact.
        pass

    points = points * 10  # make a nicer scale for points

    # Near-zero positive ROI farming penalty: if profit tiny but > 0, scale down points.
    try:
        roi_raw = profit / context.budget if context.budget else 0.0
        if 0 < roi_raw < NEAR_ZERO_ROI_THRESHOLD and points > 0:
            points *= NEAR_ZERO_ROI_PENALTY_FACTOR
    except Exception:
        pass

    # Dock fixed amount of points, if illegal
    if len(legal_stocks) != len(stocks):
        diff = abs(len(legal_stocks) - len(stocks))
        profit = -(0.1 * diff) * profit if profit > 0 else profit * (1 + (0.1 * diff))
        points = -(0.1 * diff) * points if points > 0 else points * (1 + (0.1 * diff))
    # Final sign adjustment based on profit
    points = points if profit > 0 else -abs(points)

    # Late concentration penalty (applies after sign + all additive components but before capping).
    # Strongly penalizes portfolios with fewer than MIN_UNIQUE_STOCKS by an exponential factor.
    # Resolve effective dynamic minimum unique stocks.
    effective_min_unique = RUNTIME_MIN_UNIQUE_STOCKS if RUNTIME_MIN_UNIQUE_STOCKS is not None else MIN_UNIQUE_STOCKS
    if effective_min_unique is not None and effective_min_unique > 0:
        unique_count = len({s for s, _ in stocks})
        if unique_count < effective_min_unique and unique_count > 0:
            scarcity_ratio = unique_count / effective_min_unique
            penalty_factor = scarcity_ratio ** UNIQUE_PENALTY_EXPONENT
            points *= penalty_factor
            if DEBUG:
                logging.debug(
                    "concentration penalty applied unique_count=%s min_required=%s exponent=%s factor=%.4f",
                    unique_count,
                    effective_min_unique,
                    UNIQUE_PENALTY_EXPONENT,
                    penalty_factor,
                )

    # Top performer abuse mitigation: apply multiplicative penalty per included ticker.
    try:
        if TOP_PERFORMER_PENALTY_ENABLED:
            top_set = set(TOP_PERFORMERS_LAST_20Y)
            portfolio_tickers = {t for t, _ in stocks}
            overlap_count = len(portfolio_tickers & top_set)
            if overlap_count > 0 and (points > 0 or TOP_PERFORMER_PENALIZE_NEGATIVE):
                raw_factor = TOP_PERFORMER_PENALTY_MULTIPLIER ** overlap_count
                # Prevent factor from dropping below configured floor.
                penalty_factor = max(TOP_PERFORMER_MAX_PENALTY_MULTIPLIER, raw_factor)
                original_points = points
                points *= penalty_factor
    except Exception:
        # Fail-safe: ignore any unexpected errors
        pass

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


def client_satisfaction(
    df: pd.DataFrame,
    risk_profile: float,
    age: int | None = None,
    target_vol_annual: float = CLIENT_SAT_TARGET_VOL_ANNUAL_DEFAULT,
) -> float:
    """Return client satisfaction in [0,1] based on realized daily volatility.

    This version expects an ANNUAL target volatility. It converts to daily internally:
        daily_target = annual / sqrt(TRADING_DAYS_PER_YEAR)

    Scoring shape:
        1. Compute tolerance = daily_target * combined_factor(age,risk).
        2. If LOW_VOL_PENALTY_ENABLED is False:
              volatility <= tolerance -> 1.0 else linear decay above tolerance.
        3. If LOW_VOL_PENALTY_ENABLED is True:
              Below low_band (tolerance * LOW_VOL_MIN_FACTOR) -> ramp from LOW_VOL_FLOOR_SCORE to ~0.95.
              Between low_band and tolerance -> ramp 0.95 -> 1.0.
              Above tolerance -> linear decay (1 - (excess/tolerance)).
    """
    # Daily returns
    r = df["value"].pct_change().dropna()
    if r.size == 0:
        return 0.0
    portfolio_vol = r.std()
    if not np.isfinite(portfolio_vol):
        return 0.0

    # Risk & age modulation
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
    daily_target_vol = target_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    tolerance = max(1e-9, daily_target_vol * combined_factor)

    # Determine whether to apply low-vol penalty strength.
    # risk_profile: high => more risk-averse. If above threshold, weaken penalty.
    risk_aversion_scale = 0.0 if risk_profile >= LOW_VOL_PENALTY_RISK_AVERSION_THRESHOLD else 1.0

    # If user budget is much smaller than salary (or unemployed), allow low volatility.
    # We approximate salary/budget ratio indirectly through passed risk_profile (already influenced).
    # Additional softening: interpolate penalty scale when near threshold.
    if 0 < risk_profile < LOW_VOL_PENALTY_RISK_AVERSION_THRESHOLD:
        # Linear fade as risk_profile approaches threshold from below.
        risk_aversion_scale = (LOW_VOL_PENALTY_RISK_AVERSION_THRESHOLD - risk_profile) / LOW_VOL_PENALTY_RISK_AVERSION_THRESHOLD
    # Clamp
    risk_aversion_scale = max(0.0, min(1.0, risk_aversion_scale))

    # Simple path (no low-vol penalty feature or penalty disabled by aversion) retains original semantics
    if not LOW_VOL_PENALTY_ENABLED or risk_aversion_scale == 0.0:
        if portfolio_vol <= tolerance:
            return 1.0
        score_simple = 1 - (portfolio_vol - tolerance) / tolerance
        return float(max(0.0, min(1.0, score_simple)))

    # Low-vol penalty path with conditional scaling
    low_band = tolerance * LOW_VOL_MIN_FACTOR
    if portfolio_vol < low_band:
        frac = portfolio_vol / max(low_band, 1e-12)  # [0,1)
        # Adjust dynamic floor and peak based on aversion: high aversion -> floor closer to 1.0
        dyn_floor = LOW_VOL_FLOOR_SCORE * risk_aversion_scale + (1.0) * (1 - risk_aversion_scale)
        dyn_peak = 0.95 * risk_aversion_scale + 1.0 * (1 - risk_aversion_scale)
        score_low = dyn_floor + (dyn_peak - dyn_floor) * frac
        return float(max(0.0, min(1.0, score_low)))
    if portfolio_vol <= tolerance:
        frac = (portfolio_vol - low_band) / max(tolerance - low_band, 1e-12)
        dyn_peak = 0.95 * risk_aversion_scale + 1.0 * (1 - risk_aversion_scale)
        dyn_top = 1.0  # always 1 at tolerance
        score_mid = dyn_peak + (dyn_top - dyn_peak) * frac
        return float(max(0.0, min(1.0, score_mid)))
    # Above tolerance: penalty unaffected (investor already exceeding comfort zone)
    excess = portfolio_vol - tolerance
    score_high = 1 - excess / tolerance
    return float(max(0.0, min(1.0, score_high)))


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
    if DEBUG:
        logging.debug("evaluate called with %d stocks", len(stocks))
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
        if DEBUG:
            logging.debug("ticker_details loaded for %s", stock)

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
    if DEBUG:
        logging.debug("raw profit computed: %s", profit)

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

    if DEBUG:
        logging.debug("evaluation complete profit=%s points=%s", profit, points)
    return True, "", profit, points


def init_logging(debug: bool, logfile: str | None = None):
    """Initialize rotating file logging without polluting stdout.

    Stdout is reserved for JSON protocol responses. All diagnostic output goes to the
    specified log file. Use --debug for DEBUG level; otherwise INFO is used. Errors
    still propagate via exceptions / stdout JSON payload.
    """
    level = logging.DEBUG if debug else logging.INFO
    log_path = logfile or LOG_FILE_DEFAULT
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    except Exception:
        pass
    handler = RotatingFileHandler(log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger()
    # Avoid duplicate handlers if re-init (e.g. in tests)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)
    logging.info("Logging initialized debug=%s file=%s", debug, log_path)


def main(api_key: str, data: Dict[str, Union[List[Dict[str, int]], Any]]):
    stocks_client = StocksClient(api_key)
    ref_client = ReferenceClient(api_key)
    if "context" not in data:
        print("context not passed through")
        return
    # Debug trace of which API key is being used. We keep this out of stdout to avoid leaking
    # credentials in protocol responses. If needed, adjust masking granularity.
    try:
        logging.debug("polygon api key: %s", api_key)
    except Exception:
        logging.debug("polygon api key (masked)")

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

    # Initialize dynamic limits once per run.
    global RUNTIME_MAX_STOCKS_LIMIT, RUNTIME_MIN_UNIQUE_STOCKS
    if DYNAMIC_STOCK_LIMITS_ENABLED:
        try:
            rng_dyn = random.Random(DYNAMIC_LIMITS_SEED) if DYNAMIC_LIMITS_SEED is not None else random
            max_low, max_high = MAX_STOCKS_LIMIT_RANGE
            min_low, min_high = MIN_UNIQUE_STOCKS_RANGE
            chosen_max = rng_dyn.randint(max_low, max_high)
            chosen_min = rng_dyn.randint(min_low, min_high)
            # Ensure consistency: min unique cannot exceed chosen max.
            if chosen_min > chosen_max:
                chosen_min = chosen_max - 1 if chosen_max > 1 else 1
            # Never drop below 2 (portfolio needs diversity logic to engage meaningfully)
            chosen_min = max(2, chosen_min)
            RUNTIME_MAX_STOCKS_LIMIT = chosen_max
            RUNTIME_MIN_UNIQUE_STOCKS = chosen_min
            logging.debug(
                "dynamic limits selected max_stocks=%s min_unique=%s (ranges max=%s min=%s seed=%s)",
                RUNTIME_MAX_STOCKS_LIMIT,
                RUNTIME_MIN_UNIQUE_STOCKS,
                MAX_STOCKS_LIMIT_RANGE,
                MIN_UNIQUE_STOCKS_RANGE,
                DYNAMIC_LIMITS_SEED,
            )
        except Exception:
            # Fallback to static if something unexpected occurs.
            RUNTIME_MAX_STOCKS_LIMIT = MAX_STOCKS_LIMIT
            RUNTIME_MIN_UNIQUE_STOCKS = MIN_UNIQUE_STOCKS
            logging.debug("dynamic limit selection failed; falling back to static limits")
    else:
        RUNTIME_MAX_STOCKS_LIMIT = MAX_STOCKS_LIMIT
        RUNTIME_MIN_UNIQUE_STOCKS = MIN_UNIQUE_STOCKS

    effective_max_stocks = RUNTIME_MAX_STOCKS_LIMIT if RUNTIME_MAX_STOCKS_LIMIT is not None else MAX_STOCKS_LIMIT
    # Enforce max stocks limit (ignore extras) before fetching data
    if effective_max_stocks is not None and len(stocks) > effective_max_stocks:
        stocks = stocks[:effective_max_stocks]
        if DEBUG:
            logging.debug("portfolio truncated to effective_max_stocks=%s", effective_max_stocks)

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
    if DEBUG:
        logging.debug("JSON output emitted")


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