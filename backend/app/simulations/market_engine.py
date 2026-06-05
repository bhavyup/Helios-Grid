"""MarketEngine coordinates the MarketModel and market snapshots."""

from typing import Dict

import numpy as np

from app.core.project_config import config
from app.domain.models.market_model import MarketModel
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class Order:
    household_id: int
    side: str  # "buy" | "sell"
    quantity: float
    limit_price: float


@dataclass(frozen=True)
class Trade:
    buyer_household_id: int
    seller_household_id: int
    quantity: float
    price: float


class MarketEngine:
    """Initialize and step the market model."""

    def __init__(self) -> None:
        market_cfg = config.get("market", {})
        default_price = float(
            market_cfg.get("default_price", 0.3) if hasattr(market_cfg, "get") else 0.3
        )
        price_min = float(
            market_cfg.get("price_min", 0.1) if hasattr(market_cfg, "get") else 0.1
        )
        price_max = float(
            market_cfg.get("price_max", 1.0) if hasattr(market_cfg, "get") else 1.0
        )

        self.market_model = MarketModel(
            default_price=default_price,
            price_min=price_min,
            price_max=price_max,
        )
        self.last_snapshot = self.market_model.reset()
        self.last_orders: list[dict[str, Any]] = []
        self.last_trades: list[dict[str, Any]] = []

    def reset(self) -> Dict[str, float]:
        self.last_snapshot = self.market_model.reset()
        return dict(self.last_snapshot)

    def _build_orders_from_house_states(
        self,
        house_states: list[tuple[int, list[float]]],
        clearing_price: float,
        max_qty: float = 1.0,
    ) -> list[Order]:
        orders: list[Order] = []
        for household_id, state in house_states:
            if not state or len(state) < 10:
                continue

            price = float(state[4])
            p2p_buy = float(state[6])
            p2p_sell = float(state[7])

            buy_qty = max(0.0, min(max_qty, p2p_buy))
            sell_qty = max(0.0, min(max_qty, p2p_sell))

            # use household-local price as limit; fallback to clearing_price
            bid_price = max(0.01, price if price > 0 else clearing_price)
            ask_price = max(0.01, price if price > 0 else clearing_price)

            if buy_qty > 1e-6:
                orders.append(
                    Order(
                        household_id=household_id,
                        side="buy",
                        quantity=buy_qty,
                        limit_price=bid_price,
                    )
                )
            if sell_qty > 1e-6:
                orders.append(
                    Order(
                        household_id=household_id,
                        side="sell",
                        quantity=sell_qty,
                        limit_price=ask_price,
                    )
                )

        return orders

    def _match_cda(self, orders: list[Order]) -> tuple[list[Trade], float]:
        bids = sorted(
            [o for o in orders if o.side == "buy"],
            key=lambda o: o.limit_price,
            reverse=True,
        )
        asks = sorted(
            [o for o in orders if o.side == "sell"], key=lambda o: o.limit_price
        )

        trades: list[Trade] = []
        prices: list[float] = []

        i = 0
        j = 0
        while i < len(bids) and j < len(asks):
            bid = bids[i]
            ask = asks[j]

            if bid.limit_price < ask.limit_price:
                break

            qty = min(bid.quantity, ask.quantity)
            trade_price = 0.5 * (bid.limit_price + ask.limit_price)

            trades.append(
                Trade(
                    buyer_household_id=bid.household_id,
                    seller_household_id=ask.household_id,
                    quantity=float(qty),
                    price=float(trade_price),
                )
            )
            prices.append(float(trade_price))

            # reduce quantities (copy into mutable)
            bids[i] = Order(
                bid.household_id, "buy", bid.quantity - qty, bid.limit_price
            )
            asks[j] = Order(
                ask.household_id, "sell", ask.quantity - qty, ask.limit_price
            )

            if bids[i].quantity <= 1e-6:
                i += 1
            if asks[j].quantity <= 1e-6:
                j += 1

        avg_price = float(sum(prices) / len(prices)) if prices else 0.0
        return trades, avg_price

    def step(
        self,
        supply: float,
        demand: float,
        market_action: int,
        solar: float = 0.0,
        wind: float = 0.0,
        weather: Dict[str, float] | None = None,
        house_states: list[tuple[int, list[float]]] | None = None,
    ) -> Dict[str, Any]:
        # Allow passing a weather dict to augment market decision inputs.
        if weather is not None:
            # prefer explicit pv_power if present, otherwise fall back to irradiance
            solar_val = float(
                weather.get("pv_power", weather.get("solar_irradiance", solar))
            )
            wind_val = float(weather.get("wind_speed", wind))
        else:
            solar_val = float(solar)
            wind_val = float(wind)

        orders: list[Order] = []
        trades: list[Trade] = []
        cda_price = 0.0

        if house_states is not None:
            orders = self._build_orders_from_house_states(
                house_states=house_states,
                clearing_price=float(self.last_snapshot.get("clearing_price", 0.3)),
                max_qty=1.0,
            )
            trades, cda_price = self._match_cda(orders)

        self.last_orders = [
            {
                "household_id": o.household_id,
                "side": o.side,
                "quantity": o.quantity,
                "limit_price": o.limit_price,
            }
            for o in orders
        ]
        self.last_trades = [
            {
                "buyer_household_id": t.buyer_household_id,
                "seller_household_id": t.seller_household_id,
                "quantity": t.quantity,
                "price": t.price,
            }
            for t in trades
        ]

        base_snapshot = self.market_model.step(
            supply=supply,
            demand=demand,
            market_action=int(market_action),
            solar=solar_val,
            wind=wind_val,
        )

        snapshot = dict(base_snapshot)
        snapshot["p2p_orders"] = list(self.last_orders)
        snapshot["p2p_trades"] = list(self.last_trades)
        snapshot["p2p_traded_volume"] = float(
            sum(t["quantity"] for t in self.last_trades)
        )
        snapshot["p2p_cda_price"] = float(cda_price)

        self.last_snapshot = snapshot
        return dict(self.last_snapshot)

    def get_market_state(self, num_households: int) -> np.ndarray:
        price = float(self.last_snapshot.get("clearing_price", 0.0))
        imbalance = float(self.last_snapshot.get("imbalance", 0.0))
        market_vector = np.array([price, imbalance], dtype=np.float32)
        return np.tile(market_vector, (num_households, 1))
