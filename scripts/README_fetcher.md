# Weather fetcher (NASA POWER)

This small utility fetches daily weather variables from NASA POWER for a given lat/lon and date range, writes a CSV and can upload it to the backend `POST /simulation/data/upload-weather` endpoint.

Quick run (replace token if uploading):

```powershell
python scripts/fetch_weather.py --start 2026-05-01 --end 2026-05-07 --lat 51.509865 --lon -0.118092 \
  --out sample_nasa_weather.csv --upload-url http://localhost:8000/simulation/data/upload-weather --token $env:HELIOS_TOKEN
```

Notes:
- The script uses NASA POWER daily point API and requests the `ALLSKY_SFC_SW_DWN` (surface shortwave down), `T2M` (2m temperature), `WS10M` (10m wind speed), and `RH2M` (2m relative humidity) by default.
- The output CSV format is `timestamp,VAR1,VAR2,...` with ISO date strings in the `timestamp` column.
- If you want continuous ingestion, run this script on a schedule (cron, Windows Task Scheduler, or a container) and either upload the CSV via the endpoint or write into a folder mounted into the backend container and use the `Existing CSV Path` UI option.

Security:
- If `--upload-url` is provided and you want to upload to the backend, provide a bearer token via `--token` or set `HELIOS_TOKEN` in the environment and reference it when calling the script.

If you want, I can run a sample fetch now and upload it to the backend (needs outgoing network access). Reply `run` to let me fetch and upload a 7-day sample using the default vars and coordinates, or reply `no` to skip.
