# Topology Defaults and Reset Controls

This note documents the default 64-household topology and the new reset controls that let operators override it from the dashboard.

## What changed

- The backend default household count now starts at 64.
- Reset requests can override the household count and max episode steps from the UI.
- The reset response always includes the topology payload, so the 3D view and map refresh immediately after reset.

## Backend defaults

The default household count is now 64 via configuration. The authoritative values live in the config files and are read during simulation reset:

- `config.yml`: `env.num_households` set to 64.
- `agent_config.yml`: `env.num_households` and `num_households` alias set to 64.

This keeps the default topology large enough for the 8x8 neighborhood layout without needing a manual parameter on every reset.

## Frontend reset controls

The Operate panel now exposes two additional inputs:

- **Households**: Number of households to generate in the topology (1 to 256).
- **Episode steps**: Maximum steps for a single episode before the environment returns `done`.

These values are passed to `/simulation/reset` alongside seed and weather source. If you leave them blank, the backend defaults are used.

## Operator workflow

1. Set **Households** to 64 (default) or any supported value.
2. Optionally set **Episode steps**.
3. Click **Reset**.
4. The topology in the 3D scene and the 2D map refresh immediately with the new household count.

## API equivalent

To reset directly over HTTP:

```bash
curl -X POST http://localhost:8000/simulation/reset \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"seed":123,"num_households":64,"max_episode_steps":8}'
```

The response includes `topology.nodes` with per-household `x/y` coordinates and `bounds` for rendering.
