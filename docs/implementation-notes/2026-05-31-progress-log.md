# 2026-05-31 Progress Log

## Current changes

- Added optional PV model inputs to the derive form.
- Added a confirmation gate when `profile_csv` reports unit warnings.
- Updated backend derive logic so PV estimation uses panel tilt, azimuth, area, efficiency, and temperature coefficient when provided.
- Added a timestamp-aware geometry factor for PV estimation.
- Added profile-time unit warnings to reduce irradiance/generation confusion.
- Moved warnings into the Inspect tab and attached them to the specific selected fields.
- Loosened derive gating so the confirmation checkbox unlocks derive once the source columns are set.
- Wired weather features into PPO training inputs by augmenting household states with weather vectors.
- Grouped the warning review, capped the visible list, and added auto-fill remediation in Inspect.
- Tightened weather and PV column heuristics so generation-like columns are less likely to be treated as irradiance.
- Reset mapping inputs per profiled CSV key to prevent stale mappings from leaking across files.
- Updated autofill and warning rendering so users can clear fields manually and keep Inspect field layout stable.
- Re-ran frontend typecheck (`tsc --noEmit`) after the mapping/warning fixes.

## What to verify next

1. Profile a weather-like CSV and confirm warnings are shown correctly.
2. Try to derive from a generation-only file and confirm the confirmation checkbox is required.
3. Derive with and without PV parameters and compare the output file contents.
4. Run a short PPO training pass and confirm the augmented state path does not break model shapes.
5. Confirm the derived weather file still resets the simulation cleanly.

## Notes on implementation style

- The new inputs stay optional.
- Config defaults remain the fallback.
- The derive path is fail-closed when warnings are present.
- The geometry model is deterministic and lightweight rather than physics-complete.
