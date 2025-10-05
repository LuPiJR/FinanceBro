import tempfile
import unittest
from pathlib import Path

import pandas as pd

from persistence import StockRepository, append_delisted, read_delisted


class TestStockRepository(unittest.TestCase):
    def test_save_and_load_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            repo = StockRepository(tmp.name)
            df = pd.DataFrame(
                {
                    "price": [100.0],
                    "mkt_cap": [1e12],
                    "pb": [3.0],
                    "pe": [15.0],
                    "ps": [2.0],
                    "ev_ebitda": [10.0],
                    "pcf": [12.0],
                    "dividend_yield_ttm": [0.02],
                    "buyback_yield_ttm": [0.03],
                    "shareholder_yield_ttm": [0.05],
                    "mom_6m": [0.1],
                    "vc2_score": [80.0],
                    "vc2_abs_score": [83.3],
                    "price_timestamp": ["2025-10-03T00:00:00-04:00"],
                },
                index=["AAPL"],
            )
            repo.save_metrics(df)
            refreshed = repo.get_all_metrics()
            self.assertIn("AAPL", refreshed.index)
            self.assertAlmostEqual(refreshed.loc["AAPL", "price"], 100.0)
            self.assertIn("last_updated_at", refreshed.columns)

            df2 = pd.DataFrame(
                {
                    "price": [200.0],
                    "mkt_cap": [2e12],
                },
                index=["MSFT"],
            )
            repo.save_metrics(df2)
            refreshed = repo.get_all_metrics()
            self.assertIn("AAPL", refreshed.index)
            self.assertIn("MSFT", refreshed.index)
            self.assertIn("last_updated_at", refreshed.columns)

    def test_delisted_helpers(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        try:
            append_delisted("TEST", Path(path))
            append_delisted("test", Path(path))  # duplicate ignored
            self.assertEqual(read_delisted(Path(path)), {"TEST"})
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
