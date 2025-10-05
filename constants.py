"""Shared configuration values for FinanceBro metrics pipeline."""

DEFAULT_PRICE_HISTORY_DAYS = 400
SHARES_LOOKBACK_DAYS = 500
TTM_WINDOW_DAYS = 365
MOMENTUM_MONTHS = 6

PRICE_HISTORY_PAYLOAD_LIMIT = 10
DIVIDEND_PAYLOAD_LIMIT = 12
SHARE_COUNT_PAYLOAD_LIMIT = 12
STATEMENT_PAYLOAD_LIMIT = 10

ABS_VC2_THRESHOLDS = {
    "pb": ("<=", 3.0),
    "pe": ("<=", 20.0),
    "ps": ("<=", 1.5),
    "ev_ebitda": ("<=", 10.0),
    "pcf": ("<=", 15.0),
    "shareholder_yield_ttm": (">=", 0.0),
}
