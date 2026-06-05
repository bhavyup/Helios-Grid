"""CLI test: login, profile uploaded CSV, reset if compatible.

Run with project's venv:
  .venv\Scripts\python scripts\cli_test_profile_reset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"
CREDS = {"email": "dev@helios.local", "password": "DevPass123!"}
RESOLVED_PATH = (
    "/app/backend/data/uploads/weather/20260529T151518Z_sample_nasa_weather.csv"
)


def login() -> str:
    r = requests.post(f"{API_BASE}/auth/login", json=CREDS, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def profile(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {"file_path": RESOLVED_PATH, "role": "auto", "preview_rows": 5}
    r = requests.post(
        f"{API_BASE}/simulation/data/profile", json=body, headers=headers, timeout=30
    )
    r.raise_for_status()
    return r.json()


def reset(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "weather_data_path": RESOLVED_PATH,
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
        prof = profile(token)
        print("PROFILE_OK")
        print(json.dumps(prof, indent=2))

        if prof.get("can_use_now"):
            print("Attempting RESET...")
            rst = reset(token)
            print("RESET_OK")
            print(json.dumps(rst, indent=2))
        else:
            print("PROFILE indicates file not ready for reset; derive first if needed.")

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
