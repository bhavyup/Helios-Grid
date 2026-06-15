"""HouseholdDataEngine loads and indexes household time-series data."""

from typing import Any

from app.infrastructure.data_utils import load_household_data


class HouseholdDataEngine:
    def __init__(self, household_file: str) -> None:
        self.household_file = household_file
        self.household_data = load_household_data(household_file)

    def __len__(self) -> int:
        return len(self.household_data)

    def get_household_at(self, current_time: int) -> tuple[Any, int]:
        if len(self.household_data) == 0:
            raise RuntimeError("household_data is empty; cannot index")
        idx = min(current_time, len(self.household_data) - 1)
        return self.household_data[idx], idx
