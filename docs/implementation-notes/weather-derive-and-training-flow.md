# Weather Derive and Training Flow

This note documents the current weather-CSV workflow, the PV estimation model, the unit-safety gate, and how weather features are carried into training.

## Why this exists

The derive flow had a correctness risk: generation-like columns could be mistaken for irradiance-like inputs. That can produce a believable but wrong weather file, which then contaminates the simulation and training pipeline.

The current implementation protects against that by:

- preferring explicit irradiance columns when available
- warning when a column looks like PV generation instead of irradiance
- requiring an explicit user confirmation before derive continues when warnings are present
- keeping PV model parameters optional so backend config defaults still work

## User Flow

1. Profile the CSV.
2. Inspect the profile warnings.
3. Select the weather-role columns.
4. Optionally enter PV parameters.
5. Confirm the warning checkbox if a unit mismatch is detected.
6. Derive the weather CSV.
7. Reset the simulation with the derived file.

The review step now lives in the Inspect tab, not the Operate tab. That keeps the profile, warnings, and derive decision in the same workflow surface.

## Frontend behavior

The derive panel now exposes optional inputs for:

- panel tilt
- panel azimuth
- panel area
- panel efficiency
- temperature coefficient

If the user leaves these blank, the backend defaults from config are used.

The UI also blocks the derive action when `profile_csv` returns `unit_warnings` unless the user explicitly confirms the mismatch.

### Detection intent

The UI heuristics now prefer explicit irradiance labels such as:

- irradiance
- solar_irradiance
- ghi
- dni
- dhi

Generation-like labels are treated as PV-power candidates instead of being assumed to be irradiance.

## Backend derive behavior

`/simulation/data/derive-weather` now accepts the optional PV parameters and uses them when computing `pv_power`.

When `pv_power` is not provided explicitly:

- the backend starts from irradiance in W/m^2
- applies panel area and efficiency
- applies a temperature derating factor
- applies a geometry factor based on tilt and azimuth

If a timestamp column is available, the geometry factor is timestamp-aware.
If not, the derive step falls back to a conservative orientation factor based on panel tilt and azimuth.

## Geometry model

The geometry model is intentionally simple and deterministic.

Inputs used when present:

- timestamp
- panel tilt
- panel azimuth
- site latitude from config, with a safe fallback

What it does:

- estimates solar position from time-of-day and day-of-year
- computes a sun vector
- computes the panel normal vector
- uses the incidence angle to build a plane-of-array multiplier

This is not meant to replace a full solar library. It is a practical middle ground that improves realism while keeping the flow lightweight and reproducible.

## Unit validation

`profile_csv` now emits warnings when a column looks suspicious for the selected role.

Examples:

- a solar-like column with very small values may actually be PV power in kW
- a generation-like column with large values may actually be irradiance in W/m^2

These warnings are surfaced in the UI and can block the derive step.

Each warning is structured with:

- the flagged column name
- the warning kind
- the human-readable reason
- a suggested fix

That lets the UI show the warning next to the selected field instead of hiding it in a generic summary.

For compatibility, the UI also normalizes older warning payloads that may still arrive as plain strings. Those are shown as legacy warnings and should trigger a re-profile so the backend can emit field-level diagnostics.

The Inspect panel now groups the warning list, caps the visible items, and provides an `Auto-fill safe mappings` action that populates the obvious weather fields before deriving.

The mapping state is now profile-scoped. When a newly profiled CSV arrives, role-specific mapping inputs are reset first, then re-populated with the latest safe suggestions. This prevents old-file mappings from carrying into a different CSV.

Manual clear behavior is also preserved: users can clear a mapped field and keep it empty without immediate auto-refill loops.

Field-level warning visuals are intentionally compact (badge + highlighted control container) to avoid pushing neighboring labels and inputs out of alignment.

If warnings are present, derive stays blocked until the user explicitly confirms the review checkbox.

## Training wiring

Weather now affects the training path in two ways:

- `GridEnv` forwards weather rows to the market engine and households
- PPO training augments each household state with weather features before the policy consumes it

The weather feature bundle used for PPO includes:

- solar irradiance
- wind speed
- temperature
- humidity

That means the training dataset now sees weather explicitly instead of only inferring it indirectly.

## Validation approach

Recommended checks after changes to this workflow:

1. Profile a CSV that contains irradiance-like columns and verify that warnings are sensible.
2. Try a generation-only CSV and confirm the derive button blocks until you confirm the warning.
3. Derive with empty PV fields and confirm backend defaults are used.
4. Derive with explicit PV fields and confirm the returned output changes.
5. Reset the simulation with the derived output and verify it loads cleanly.
6. Run PPO training and confirm the augmented state size remains consistent.
7. Verify that the warning card appears in Inspect and the matching field rows show inline review badges.
8. Profile one CSV, switch to a different schema, and verify stale mappings are cleared before new suggestions apply.
9. Clear one suggested field manually and verify it remains cleared unless the user runs the explicit auto-fill action again.

## Implementation notes

- Keep new PV parameters optional.
- Keep the safety gate visible and explicit.
- Do not silently treat generation as irradiance.
- Prefer deterministic math and clear fallbacks over hidden heuristics.
