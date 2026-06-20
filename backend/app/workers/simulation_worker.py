from __future__ import annotations

from typing import Any

import ray

from app.services.simulation_service import SimulationService


@ray.remote
def run_simulation(
    steps: int,
    seed: int | None,
    num_households: int | None,
    max_episode_steps: int | None,
    weather_data_path: str | None,
    use_autopilot: bool,
    market_action: int | None,
) -> dict[str, Any]:
    service = SimulationService()
    service.reset(
        seed=seed,
        num_households=num_households,
        max_episode_steps=max_episode_steps,
        weather_data_path=weather_data_path,
    )
    return service.run(
        steps=steps,
        use_autopilot=use_autopilot,
        market_action=market_action,
    )
