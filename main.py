# wws_probe.py
# Minimal yfinance probes to fetch inputs for WWS/VC2-style factors
# Prints the raw ingredients + a few ratios so you can verify them.

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from clients import YFinanceClient
from constants import (
    DEFAULT_PRICE_HISTORY_DAYS,
    DIVIDEND_PAYLOAD_LIMIT,
    PRICE_HISTORY_PAYLOAD_LIMIT,
    SHARES_LOOKBACK_DAYS,
    SHARE_COUNT_PAYLOAD_LIMIT,
    TTM_WINDOW_DAYS,
)
from domain import DividendInfo, PriceSnapshot, ShareCounts
from persistence import StockRepository, append_delisted, read_delisted
from persistence.delisted import DEFAULT_DELISTED_FILE
from services import FinancialCalculator, absolute_vc2_score, compute_relative_vc2
from utils import (
    format_metric_value,
    frame_payload,
    sanitize_float,
    series_payload,
    statement_payload,
)


CALCULATOR = FinancialCalculator()
DELISTED_FILE = DEFAULT_DELISTED_FILE


class FinancialMetrics(BaseModel):
    ticker: str
    price: float | None = None
    shares_now: float | None = None
    shares_prev: float | None = None
    shares_diluted: float | None = None
    revenue_ttm: float | None = None
    net_income_ttm: float | None = None
    ebitda_ttm: float | None = None
    cfo_ttm: float | None = None
    cash: float | None = None
    debt_total: float | None = None
    equity: float | None = None
    bvps: float | None = None
    mkt_cap: float | None = None
    enterprise_value: float | None = None
    dividend_ttm_per_sh: float | None = None
    dividend_yield_ttm: float | None = None
    buyback_yield_ttm: float | None = None
    shareholder_yield_ttm: float | None = None
    pb: float | None = None
    pe: float | None = None
    ps: float | None = None
    ev_ebitda: float | None = None
    pcf: float | None = None
    mom_6m: float | None = None
    vc2_score: float | None = None
    vc2_abs_score: float | None = None
    price_timestamp: str | None = None

    @field_validator(
        "price",
        "shares_now",
        "shares_prev",
        "shares_diluted",
        "revenue_ttm",
        "net_income_ttm",
        "ebitda_ttm",
        "cfo_ttm",
        "cash",
        "debt_total",
        "equity",
        "bvps",
        "mkt_cap",
        "enterprise_value",
        "dividend_ttm_per_sh",
        "dividend_yield_ttm",
        "buyback_yield_ttm",
        "shareholder_yield_ttm",
        "pb",
        "pe",
        "ps",
        "ev_ebitda",
        "pcf",
        "mom_6m",
        "vc2_score",
        "vc2_abs_score",
        mode="before",
    )
    @classmethod
    def _sanitize_numeric(cls, value: Any) -> Optional[float]:
        return sanitize_float(value)


class FetchError(BaseModel):
    stage: str
    message: str


class FetchResult(BaseModel):
    ticker: str
    metrics: FinancialMetrics | None = None
    errors: List[FetchError] = Field(default_factory=list)
    payloads: Dict[str, Any] = Field(default_factory=dict)

    def add_error(self, stage: str, message: str) -> None:
        self.errors.append(FetchError(stage=stage, message=message))

    def add_payload(self, stage: str, data: Any) -> None:
        if data in ({}, []):
            return
        self.payloads[stage] = data


def resolve_stage(
    stage: str,
    supplier: Callable[[], Any],
    result: FetchResult,
    payload_builder: Callable[[Any], Any] | None = None,
):
    try:
        data = supplier()
    except Exception as exc:
        result.add_error(stage, str(exc))
        if "No data found" in str(exc) or "symbol may be delisted" in str(exc):
            result.add_error("delisted", str(exc))
        return None

    if callable(data):
        try:
            data = data()
        except Exception as exc:
            result.add_error(f"{stage}_call", str(exc))
            return None

    if data is None:
        return None

    payload: Any | None = None
    if payload_builder is not None:
        try:
            payload = payload_builder(data)
        except Exception as exc:
            result.add_error(f"{stage}_payload", str(exc))
            payload = None
    elif isinstance(data, pd.DataFrame):
        payload = statement_payload(data)
    elif isinstance(data, pd.Series):
        payload = series_payload(data)

    if payload is not None:
        result.add_payload(stage, payload)

    return data


def grab(df: pd.DataFrame | None, *keys: str):
    """
    Try multiple row labels; return the first non-null scalar.
    yfinance financial statements are DataFrames with row labels and date columns.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.nan
    for k in keys:
        if k in df.index:
            vals = df.loc[k]
            try:
                # If multiple columns (periods), take the most recent (first col)
                return vals.iloc[0] if hasattr(vals, "iloc") else vals
            except Exception:
                pass
    return np.nan


def ttm_or_last(df_ttm: pd.DataFrame | None, df_annual: pd.DataFrame | None, *keys: str):
    """Prefer TTM statement row; fall back to last annual."""
    v = grab(df_ttm, *keys)
    if pd.isna(v):
        v = grab(df_annual, *keys)
    return v


def fetch_one(ticker: str, price_history_days: int = DEFAULT_PRICE_HISTORY_DAYS) -> FetchResult:
    result = FetchResult(ticker=ticker)

    try:
        client = YFinanceClient(ticker)
    except Exception as exc:
        result.add_error("initialization", str(exc))
        return result

    # ---------- PRICE ----------
    hist = resolve_stage(
        "price_history",
        lambda: client.price_history(price_history_days, auto_adjust=True, actions=False),
        result,
        payload_builder=lambda df: frame_payload(df, limit=PRICE_HISTORY_PAYLOAD_LIMIT),
    )
    price = np.nan
    price_timestamp_iso: str | None = None
    close_series: pd.Series | None = None
    if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
        close_series = hist["Close"].dropna()
        if not close_series.empty:
            price = float(close_series.iloc[-1])
            price_timestamp_iso = close_series.index[-1].isoformat()

    price_snapshot = PriceSnapshot(
        history=hist if isinstance(hist, pd.DataFrame) else None,
        close=sanitize_float(price),
        timestamp=price_timestamp_iso,
    )

    # ---------- DIVIDENDS (TTM) ----------
    divs = resolve_stage(
        "dividends",
        client.dividends,
        result,
        payload_builder=lambda s: series_payload(s, limit=DIVIDEND_PAYLOAD_LIMIT),
    )
    if divs is not None and len(divs) > 0:
        cutoff = divs.index.max() - pd.Timedelta(days=TTM_WINDOW_DAYS)
        div_ttm = float(divs[divs.index >= cutoff].sum())
    else:
        div_ttm = 0.0
    dividend_yield = CALCULATOR.dividend_yield(div_ttm, price_snapshot.close)
    dividend_info = DividendInfo(
        series=divs if isinstance(divs, pd.Series) else None,
        ttm_total=sanitize_float(div_ttm),
        dividend_yield=dividend_yield,
    )

    # ---------- SHARES OUTSTANDING (for Mkt Cap & Buyback Yield) ----------
    s_now = np.nan
    s_prev = np.nan
    start_date = (datetime.now(timezone.utc) - timedelta(days=SHARES_LOOKBACK_DAYS)).date()

    shares = resolve_stage(
        "share_counts",
        lambda: client.shares_full(start=start_date),
        result,
        payload_builder=lambda data: (
            frame_payload(data, limit=SHARE_COUNT_PAYLOAD_LIMIT)
            if isinstance(data, pd.DataFrame)
            else series_payload(data, limit=SHARE_COUNT_PAYLOAD_LIMIT)
        ),
    )
    if isinstance(shares, (pd.Series, pd.DataFrame)):
        shares_series = shares.squeeze() if isinstance(shares, pd.DataFrame) else shares
        if isinstance(shares_series, pd.Series):
            shares_series = shares_series.dropna()
            if len(shares_series):
                s_now = float(shares_series.iloc[-1])
                cutoff = shares_series.index.max() - pd.Timedelta(days=TTM_WINDOW_DAYS)
                if (shares_series.index <= cutoff).any():
                    s_prev = float(shares_series[shares_series.index <= cutoff].iloc[-1])
                else:
                    s_prev = float(shares_series.iloc[0])

    share_counts = ShareCounts(
        current=sanitize_float(s_now),
        previous=sanitize_float(s_prev),
    )

    buyback_yield_shares = CALCULATOR.buyback_yield_from_shares(
        share_counts.current,
        share_counts.previous,
    )
    buyback_yield = buyback_yield_shares

    # ---------- FINANCIALS (TTM preferred; fall back to annual) ----------
    inc_ttm = resolve_stage("ttm_income_stmt", client.ttm_income_statement, result)
    cfs_ttm = resolve_stage("ttm_cashflow", client.ttm_cashflow, result)

    def get_income_stmt():
        return client.income_statement(freq="yearly")

    def get_income_stmt_quarterly():
        return client.income_statement(freq="quarterly")

    def get_cashflow():
        return client.cashflow(freq="yearly")

    def get_balance_sheet():
        return client.balance_sheet(freq="yearly")

    def get_balance_sheet_quarterly():
        return client.balance_sheet(freq="quarterly")

    inc_ann = resolve_stage("income_stmt_annual", get_income_stmt, result)
    inc_qtr = resolve_stage("income_stmt_quarterly", get_income_stmt_quarterly, result)
    cfs_ann = resolve_stage("cashflow_annual", get_cashflow, result)
    bse = resolve_stage("balance_sheet_annual", get_balance_sheet, result)
    bse_qtr = resolve_stage("balance_sheet_quarterly", get_balance_sheet_quarterly, result)

    # Income Statement items
    revenue_ttm = ttm_or_last(inc_ttm, inc_ann, "Total Revenue", "TotalRevenue", "Revenue")
    net_inc_ttm = ttm_or_last(inc_ttm, inc_ann, "Net Income Common Stockholders", "NetIncome", "Net Income")
    ebitda_ttm = ttm_or_last(inc_ttm, inc_ann, "EBITDA")

    # Cash Flow items
    cfo_ttm = ttm_or_last(cfs_ttm, cfs_ann, "Operating Cash Flow", "Total Cash From Operating Activities")
    net_stock_issuance = ttm_or_last(
        cfs_ttm,
        cfs_ann,
        "NetCommonStockIssuance",
        "Net Common Stock Issuance",
        "CommonStockIssuance",
    )
    if pd.isna(net_stock_issuance):
        repurchase = ttm_or_last(
            cfs_ttm,
            cfs_ann,
            "RepurchaseOfCapitalStock",
            "CommonStockPayments",
            "RepurchaseOfCommonStock",
        )
        issuance = ttm_or_last(
            cfs_ttm,
            cfs_ann,
            "CommonStockIssuance",
            "IssuanceOfCapitalStock",
        )
        if not pd.isna(repurchase) or not pd.isna(issuance):
            repurchase = 0 if pd.isna(repurchase) else repurchase
            issuance = 0 if pd.isna(issuance) else issuance
            net_stock_issuance = issuance + repurchase

    # Balance Sheet items (last annual used for EV pieces & equity)
    cash = grab(bse, "Cash And Cash Equivalents", "CashCashEquivalentsAndShortTermInvestments", "Cash")
    short_long_term_debt = grab(bse, "Short Long Term Debt", "ShortTermDebt")
    long_term_debt = grab(bse, "Long Term Debt", "LongTermDebt")
    total_debt = grab(bse, "Total Debt")
    if pd.isna(total_debt):
        total_debt = (0 if pd.isna(short_long_term_debt) else short_long_term_debt) + (
            0 if pd.isna(long_term_debt) else long_term_debt
        )

    equity_quarterly = grab(
        bse_qtr,
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "StockholdersEquity",
        "Stockholders Equity",
    )
    equity_annual = grab(
        bse,
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "StockholdersEquity",
        "Stockholders Equity",
    )

    diluted_shares = grab(
        inc_qtr,
        "DilutedAverageShares",
        "Diluted Average Shares",
        "DilutedWeightedAverageShares",
        "Diluted Weighted Average Shares",
    )
    equity_quarterly_val = sanitize_float(equity_quarterly)
    equity_annual_val = sanitize_float(equity_annual)
    diluted_shares_val = sanitize_float(diluted_shares)
    shares_now_val = sanitize_float(s_now)

    equity_value = equity_quarterly_val if equity_quarterly_val is not None else equity_annual_val
    shares_basis = diluted_shares_val if diluted_shares_val is not None else shares_now_val

    if equity_value is None or shares_basis is None or shares_basis <= 0:
        bvps = np.nan
    else:
        bvps = equity_value / shares_basis

    price_val = price_snapshot.close
    shares_now_val = share_counts.current
    shares_prev_val = share_counts.previous
    shares_diluted_val = sanitize_float(diluted_shares)
    revenue_val = sanitize_float(revenue_ttm)
    net_income_val = sanitize_float(net_inc_ttm)
    ebitda_val = sanitize_float(ebitda_ttm)
    cfo_val = sanitize_float(cfo_ttm)
    cash_val = sanitize_float(cash)
    debt_total_val = sanitize_float(total_debt)
    equity_val = sanitize_float(equity_value)
    bvps_val = sanitize_float(bvps)
    div_ttm_val = dividend_info.ttm_total
    div_yield_val = sanitize_float(dividend_info.dividend_yield)
    net_stock_issuance_val = sanitize_float(net_stock_issuance)

    # ---------- DERIVED METRICS ----------
    mkt_cap = CALCULATOR.market_cap(price_val, shares_now_val)
    ev = CALCULATOR.enterprise_value(mkt_cap, debt_total_val, cash_val)
    pe = CALCULATOR.price_to_earnings(price_val, net_income_val, shares_now_val)
    ps = CALCULATOR.price_to_sales(mkt_cap, revenue_val)
    ev_ebitda = CALCULATOR.ev_to_ebitda(ev, ebitda_val)
    pcf = CALCULATOR.price_to_cashflow(price_val, cfo_val, shares_now_val)
    pb = CALCULATOR.price_to_book(price_val, bvps_val)

    buyback_yield_cf = CALCULATOR.buyback_yield_from_cashflow(net_stock_issuance_val, mkt_cap)
    buyback_yield = CALCULATOR.select_buyback_yield(buyback_yield_shares, buyback_yield_cf)
    shareholder_yield = CALCULATOR.shareholder_yield(dividend_info.dividend_yield, buyback_yield)
    mom_6m = CALCULATOR.momentum(close_series)

    buyback_yield_val = sanitize_float(buyback_yield)
    shareholder_yield_val = sanitize_float(shareholder_yield)
    mkt_cap_val = sanitize_float(mkt_cap)
    ev_val = sanitize_float(ev)
    pb_val = sanitize_float(pb)
    pe_val = sanitize_float(pe)
    ps_val = sanitize_float(ps)
    ev_ebitda_val = sanitize_float(ev_ebitda)
    pcf_val = sanitize_float(pcf)
    mom_6m_val = sanitize_float(mom_6m)

    vc2_abs_score = absolute_vc2_score(
        {
            "pb": pb_val,
            "pe": pe_val,
            "ps": ps_val,
            "ev_ebitda": ev_ebitda_val,
            "pcf": pcf_val,
            "shareholder_yield_ttm": shareholder_yield_val,
        }
    )

    metrics = FinancialMetrics(
        ticker=ticker,
        price=price_val,
        shares_now=shares_now_val,
        shares_prev=shares_prev_val,
        shares_diluted=shares_diluted_val,
        revenue_ttm=revenue_val,
        net_income_ttm=net_income_val,
        ebitda_ttm=ebitda_val,
        cfo_ttm=cfo_val,
        cash=cash_val,
        debt_total=debt_total_val,
        equity=equity_val,
        bvps=bvps_val,
        mkt_cap=mkt_cap_val,
        enterprise_value=ev_val,
        dividend_ttm_per_sh=div_ttm_val,
        dividend_yield_ttm=div_yield_val,
        buyback_yield_ttm=buyback_yield_val,
        shareholder_yield_ttm=shareholder_yield_val,
        pb=pb_val,
        pe=pe_val,
        ps=ps_val,
        ev_ebitda=ev_ebitda_val,
        pcf=pcf_val,
        mom_6m=mom_6m_val,
        vc2_abs_score=vc2_abs_score,
        price_timestamp=price_snapshot.timestamp,
    )
    result.metrics = metrics

    return result


def main():
    p = argparse.ArgumentParser(description="Probe yfinance fields for WWS-style factor inputs")
    p.add_argument("tickers", nargs="*", help="Tickers to fetch (e.g., AAPL MSFT NVDA)")
    p.add_argument(
        "--ticker-file",
        type=str,
        help="Path to file containing one ticker per line (appended to CLI tickers).",
    )
    p.add_argument(
        "--price-days",
        type=int,
        default=DEFAULT_PRICE_HISTORY_DAYS,
        help=(
            "Number of days to request for the price history window "
            f"(default: {DEFAULT_PRICE_HISTORY_DAYS})."
        ),
    )
    p.add_argument(
        "--show-endpoints",
        action="store_true",
        help="Display the raw payload (truncated) returned by each yfinance endpoint.",
    )
    p.add_argument(
        "--json-output",
        action="store_true",
        help="Dump the collected metrics as JSON after the summary table.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing results to SQLite (defaults to stocks.db).",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default="stocks.db",
        help="SQLite database file to store the computed metrics (default: stocks.db).",
    )
    args = p.parse_args()

    metrics_rows: List[Dict[str, Any]] = []

    essentials_order = [
        "price",
        "shares_now",
        "shares_diluted",
        "revenue_ttm",
        "net_income_ttm",
        "ebitda_ttm",
        "cfo_ttm",
        "cash",
        "debt_total",
        "equity",
        "bvps",
        "mkt_cap",
        "enterprise_value",
        "dividend_yield_ttm",
        "buyback_yield_ttm",
        "shareholder_yield_ttm",
        "pb",
        "pe",
        "ps",
        "ev_ebitda",
        "pcf",
        "mom_6m",
        "vc2_abs_score",
    ]

    cli_tickers = list(args.tickers)
    file_tickers: List[str] = []
    if args.ticker_file:
        with open(args.ticker_file) as fh:
            file_tickers = [
                line.strip()
                for line in fh
                if line.strip() and not line.lstrip().startswith("#")
            ]

    delisted = read_delisted()
    combined = cli_tickers + file_tickers
    all_tickers = [ticker for ticker in combined if ticker.upper() not in delisted]
    skipped = sorted({t.upper() for t in combined if t.upper() in delisted})
    if not all_tickers:
        if skipped:
            raise SystemExit("All specified tickers are marked delisted in delisted_list.txt")
        raise SystemExit("No tickers provided via CLI or --ticker-file")

    if skipped:
        print(f"Skipping {len(skipped)} delisted tickers: {', '.join(skipped)}")

    for ticker in all_tickers:
        print(f"\n=== Fetching {ticker} ===")
        result = fetch_one(ticker, price_history_days=args.price_days)

        if result.errors:
            for err in result.errors:
                print(f"{ticker}: {err.stage} -> {err.message}")
            messages = [err.message.lower() for err in result.errors]
            if any("no data found" in msg or "delisted" in msg for msg in messages):
                append_delisted(ticker)
                print(f"{ticker}: added to {DELISTED_FILE}")
                continue

        if args.show_endpoints and result.payloads:
            print("-- Endpoint payloads --")
            for stage, payload in result.payloads.items():
                try:
                    payload_str = json.dumps(payload, indent=2, default=str)
                except TypeError:
                    payload_str = str(payload)
                print(f"[{stage}]\n{payload_str}")

        metrics = result.metrics
        if metrics is None:
            if not result.errors:
                print(f"{ticker}: no metrics available (endpoint returned no data).")
            continue

        metrics_data = metrics.model_dump()
        metrics_rows.append(metrics_data)

        for key in essentials_order:
            value = metrics_data.get(key)
            print(f"{ticker:>6} | {key:>22}: {format_metric_value(value)}")

    if metrics_rows:
        df = pd.DataFrame(metrics_rows).set_index("ticker")

        vc_components: Dict[str, bool] = {
            "pb": True,
            "pe": True,
            "ps": True,
            "ev_ebitda": True,
            "pcf": True,
            "shareholder_yield_ttm": False,
        }
        vc2_series = compute_relative_vc2(df, vc_components)
        if not vc2_series.empty:
            df["vc2_score"] = vc2_series
            for row in metrics_rows:
                ticker = row.get("ticker")
                if ticker in vc2_series:
                    row["vc2_score"] = float(vc2_series.loc[ticker])

        numeric_cols = [
            "price",
            "mkt_cap",
            "pb",
            "pe",
            "ps",
            "ev_ebitda",
            "pcf",
            "dividend_yield_ttm",
            "buyback_yield_ttm",
            "shareholder_yield_ttm",
            "mom_6m",
            "vc2_score",
            "vc2_abs_score",
        ]
        string_cols = ["price_timestamp"]

        summary_df = pd.DataFrame(index=df.index)
        for col in numeric_cols:
            if col in df.columns:
                summary_df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
        for col in string_cols:
            if col in df.columns:
                summary_df[col] = df[col]

        if not summary_df.empty:
            print("\n=== Summary table ===")
            print(summary_df.to_string())

        if not args.no_save:
            repo = StockRepository(args.db_path)
            repo.save_metrics(df)

    if args.json_output and metrics_rows:
        print("\n=== JSON metrics ===")
        print(json.dumps(metrics_rows, indent=2))


if __name__ == "__main__":
    main()
