from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _day_shape(hour: float) -> float:
    # two peaks: morning and evening, plus a small midday bump
    morning = math.exp(-0.5 * ((hour - 7.5) / 1.8) ** 2)
    evening = math.exp(-0.5 * ((hour - 19.0) / 2.3) ** 2)
    midday = 0.35 * math.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2)
    base = 0.25
    return base + 0.9 * morning + 1.2 * evening + midday


def generate_household_csv(
    out_path: Path,
    n_households: int = 64,
    steps: int = 96,  # 15-min day
    step_minutes: int = 15,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    start = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
    ts = [start + timedelta(minutes=step_minutes * i) for i in range(steps)]

    # per-house scaling and noise
    house_scale = np.clip(
        np_rng.normal(loc=1.0, scale=0.25, size=n_households), 0.55, 1.8
    )
    np.clip(np_rng.normal(loc=1.0, scale=0.15, size=n_households), 0.6, 1.5)

    rows = []
    for _i, t in enumerate(ts):
        hour = t.hour + t.minute / 60.0
        shape = _day_shape(hour)

        # aggregate-ish baseline (kW)
        agg = 55.0 + 18.0 * shape + rng.uniform(-2.0, 2.0)

        # allocate across houses with diversity
        per = agg * (house_scale / house_scale.sum())
        # add per-house jitter but keep non-negative
        per = np.clip(
            per * (1.0 + np_rng.normal(0.0, 0.08, size=n_households)), 0.0, None
        )

        row = {
            "utc_timestamp": t.isoformat(),
            "consumption": float(per.sum()),
        }
        for h in range(n_households):
            row[f"consumption_{h + 1}"] = float(per[h])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def generate_market_csv(
    out_path: Path,
    household_csv: Path,
    price_only: bool = False,
    seed: int = 42,
) -> Path:
    random.Random(seed)
    df_h = pd.read_csv(household_csv)
    demand = pd.to_numeric(df_h["consumption"], errors="coerce").fillna(0.0).to_numpy()

    # synthetic supply tracks demand but with noise (think renewables)
    supply = np.clip(
        0.78 * demand + np.random.default_rng(seed).normal(0, 3.5, size=len(demand)),
        0.0,
        None,
    )

    # price increases with demand and shortage
    shortage = np.clip(demand - supply, 0.0, None)
    price = (
        0.18
        + 0.0042 * (demand / max(demand.max(), 1.0))
        + 0.0065 * (shortage / max(shortage.max(), 1.0))
    )
    price = np.clip(
        price + np.random.default_rng(seed + 1).normal(0, 0.01, size=len(price)),
        0.05,
        2.0,
    )

    out = pd.DataFrame(
        {"utc_timestamp": df_h["utc_timestamp"], "price": price.astype(np.float32)}
    )
    if not price_only:
        out.insert(1, "supply", supply.astype(np.float32))
        out.insert(2, "demand", demand.astype(np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    backend_root = Path(__file__).resolve().parents[1]
    household_out = (
        backend_root
        / "data"
        / "historical_energy_data"
        / "demo_household_64_96steps.csv"
    )
    market_price_only_out = (
        backend_root / "data" / "market_prices" / "demo_market_price_only_96steps.csv"
    )
    market_full_out = (
        backend_root / "data" / "market_prices" / "demo_market_full_96steps.csv"
    )

    hh = generate_household_csv(household_out, n_households=64, steps=96, seed=42)
    generate_market_csv(market_price_only_out, hh, price_only=True, seed=42)
    generate_market_csv(market_full_out, hh, price_only=False, seed=42)

    print("Generated:")
    print(" -", household_out)
    print(" -", market_price_only_out)
    print(" -", market_full_out)
