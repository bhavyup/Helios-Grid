"""WeatherEngine loads and indexes weather time-series data."""

from typing import Any

from app.infrastructure.data_utils import load_weather_data


class WeatherEngine:
    """Load weather data and provide safe time indexing."""

    def __init__(self, weather_file: str) -> None:
        self.weather_file = weather_file
        self.weather_data = load_weather_data(weather_file)

    def __len__(self) -> int:
        return len(self.weather_data)

    def get_weather_at(self, current_time: int) -> tuple[Any, int]:
        if len(self.weather_data) == 0:
            raise RuntimeError("weather_data is empty; cannot index")
        idx = min(current_time, len(self.weather_data) - 1)
        return self.weather_data[idx], idx
