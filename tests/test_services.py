import unittest

import pandas as pd

from services import FinancialCalculator, absolute_vc2_score, compute_relative_vc2


class TestFinancialCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = FinancialCalculator()

    def test_market_cap(self):
        self.assertEqual(self.calculator.market_cap(100.0, 10.0), 1000.0)

    def test_price_to_earnings(self):
        self.assertEqual(self.calculator.price_to_earnings(120.0, 12.0, 3.0), 30.0)

    def test_momentum(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        prices = pd.Series(range(200), index=dates, dtype=float)
        momentum = self.calculator.momentum(prices, months=6)
        self.assertIsNotNone(momentum)
        self.assertGreater(momentum, 0)


class TestVC2Scorer(unittest.TestCase):
    def test_absolute_score(self):
        metrics = {
            "pb": 2.5,
            "pe": 15.0,
            "ps": 1.2,
            "ev_ebitda": 8.0,
            "pcf": 12.0,
            "shareholder_yield_ttm": 0.05,
        }
        score = absolute_vc2_score(metrics)
        self.assertEqual(score, 100.0)

    def test_relative_score(self):
        df = pd.DataFrame(
            {
                "pb": [2.0, 5.0],
                "pe": [10.0, 50.0],
                "shareholder_yield_ttm": [0.03, 0.00],
            },
            index=["VALUE", "GROWTH"],
        )
        directions = {"pb": True, "pe": True, "shareholder_yield_ttm": False}
        scores = compute_relative_vc2(df, directions)
        self.assertGreater(scores.loc["VALUE"], scores.loc["GROWTH"])


if __name__ == "__main__":
    unittest.main()
