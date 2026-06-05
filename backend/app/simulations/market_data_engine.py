"""MarketDataEngine loads and indexes market time-series data."""

from typing import Any

from app.infrastructure.data_utils import load_market_data


class MarketDataEngine:
    def __init__(self, market_file: str) -> None:
        self.market_file = market_file
        self.market_data = load_market_data(market_file)

    def __len__(self) -> int:
        return len(self.market_data)

    def get_market_at(self, current_time: int) -> tuple[Any, int]:
        if len(self.market_data) == 0:
            raise RuntimeError("market_data is empty; cannot index")
        idx = min(current_time, len(self.market_data) - 1)
        return self.market_data[idx], idx