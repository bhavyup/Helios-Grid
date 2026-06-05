"""CLI helper: derive a weather CSV from a wide source then reset the simulation.

Run with project's venv:
  .venv\Scripts\python scripts\cli_derive_and_reset.py
"""

from __future__ import annotations

import json
import sys

import requests

API_BASE = "http://localhost:8000"
CREDS = {"email": "dev@helios.local", "password": "DevPass123!"}
SOURCE_PATH = (
    "/app/backend/data/uploads/weather/20260529T151518Z_sample_nasa_weather.csv"
)


def login() -> str:
    r = requests.post(f"{API_BASE}/auth/login", json=CREDS, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def derive(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "file_path": SOURCE_PATH,
        "solar_column": "ALLSKY_SFC_SW_DWN",
        "wind_column": "WS10M",
        "timestamp_column": "timestamp",
        "temperature_column": "T2M",
        "humidity_column": "RH2M",
        "normalize_signals": True,
    }
    r = requests.post(
        f"{API_BASE}/simulation/data/derive-weather",
        json=body,
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def profile(token: str, path: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {"file_path": path, "role": "weather", "preview_rows": 5}
    r = requests.post(
        f"{API_BASE}/simulation/data/profile", json=body, headers=headers, timeout=30
    )
    r.raise_for_status()
    return r.json()


def reset(token: str, path: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "weather_data_path": path,
        "seed": 42,
        "num_households": 8,
        "max_episode_steps": 24,
    }
    r = requests.post(
        f"{API_BASE}/simulation/reset", json=body, headers=headers, timeout=30
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    try:
        token = login()
        print("LOGIN_OK")
        derived = derive(token)
        print("DERIVE_OK")
        print(json.dumps(derived, indent=2))

        derived_path = derived.get("resolved_path") or derived.get("file_path")
        if not derived_path:
            print("No derived path returned", file=sys.stderr)
            return 2

        prof = profile(token, derived_path)
        print("PROFILE_DERIVED_OK")
        print(json.dumps(prof, indent=2))

        if prof.get("can_use_now"):
            rst = reset(token, derived_path)
            print("RESET_OK")
            print(json.dumps(rst, indent=2))
        else:
            print("Derived file still not ready for runtime. Inspect profile output.")

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
