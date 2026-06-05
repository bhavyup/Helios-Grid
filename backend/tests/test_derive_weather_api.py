from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd


def test_derive_weather_endpoint_creates_output(client, auth_headers) -> None:
    backend_root = Path(__file__).resolve().parents[1]
    artifacts_dir = backend_root / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex
    source_path = artifacts_dir / f"derive_weather_source_{run_id}.csv"
    output_path = artifacts_dir / f"derive_weather_output_{run_id}.csv"

    df = pd.DataFrame(
        {
            "utc_timestamp": [
                "2026-05-31T00:00:00Z",
                "2026-05-31T01:00:00Z",
                "2026-05-31T02:00:00Z",
            ],
            "solar_irradiance": [0.0, 400.0, 800.0],
            "wind_speed": [1.0, 3.0, 2.0],
            "temperature": [20.0, 22.0, 24.0],
            "humidity": [50.0, 55.0, 60.0],
        }
    )
    df.to_csv(source_path, index=False)

    payload = {
        "file_path": str(source_path),
        "solar_column": "solar_irradiance",  # required by route schema
        "wind_column": "wind_speed",  # required by route schema
        "timestamp_column": "utc_timestamp",
        "temperature_column": "temperature",
        "humidity_column": "humidity",
        "irradiance_column": "solar_irradiance",
        "output_path": str(output_path),
        "normalize_signals": True,
    }

    response = client.post(
        "/simulation/data/derive-weather",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert "output_file_path" in body
    derived_path = Path(body["output_file_path"])

    assert derived_path.exists(), f"Expected derived CSV to exist at {derived_path}"
    assert derived_path.resolve() == output_path.resolve()

    derived_df = pd.read_csv(derived_path)
    for col in [
        "temperature",
        "solar_irradiance",
        "wind_speed",
        "humidity",
        "pv_power",
    ]:
        assert col in derived_df.columns
