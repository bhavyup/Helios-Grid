Optimized tool selectionCompacted conversationI'll create a short todo plan for this work, then give concise guidance and an implementation offer.Optimized tool selection

Updated todo list

**Quick plan:** recommend CSV structure and mapping fields, propose clear UI behavior (auto-detect + override), and offer to implement the role-aware mapping form in simulation-controls.tsx.

- **Recommendation: separate files:** Prefer one CSV per role (weather / household / market). Reasons: simpler validation, clearer expectations, smaller upload surface, easier pipelines and provenance, and cleaner UX when profiling and deriving mappings.

- **Core required fields**
  - **Universal:** timestamp (ISO/UTC preferred; e.g., `utc_timestamp`). Always required.
  - **Weather (typical):** timestamp, temperature, solar_irradiance or solar_generation, wind_speed or wind_generation, humidity (optional). At minimum: timestamp + one physical signal (solar or temperature or wind).
  - **Household (typical):** timestamp, consumption (kWh or kW). `consumption` is the essential field — household CSVs must include timestamp + consumption. Optional: `household_id`, `pv_generation`, `ev_charge`.
  - **Market (typical):** timestamp, price. Optional: `supply`, `demand`, `market_id`.

- **Derived-mapping form behavior (UX)**
  - Show mapping fields that match the selected role. E.g., when role = Weather, show Solar, Wind, Temperature, Humidity, Timestamp fields; when role = Household, show Consumption and Timestamp (and optional household id); when role = Market, show Price + Timestamp.
  - Populate each mapping control with a dropdown of detected CSV columns (from the analyzer). Pre-select the best match where available.
  - Display a compatibility badge and list of matched/missing fields; for household require `consumption` — if missing, show explicit guidance: "Add a consumption column to make this household-compatible."
  - Provide a visible "Role (auto-detected)" indicator and a manual override control (allow user to force role).
  - Show a small sample preview (first N rows) to help users confirm mapping.

- **Auto-detect logic (recommended)**
  - Build expected-sets per role (e.g., Weather: {timestamp, temp, solar, wind, humidity}; Household: {timestamp, consumption}; Market: {timestamp, price, supply, demand}).
  - For each role compute matched_count = number of expected fields present (case-insensitive, allow common aliases).
  - Choose role with highest matched_count. If (matched_count_difference >= 1) or winner meets required-fields rule, auto-select that role.
  - Required-fields rule example:
    - Household: require timestamp + consumption.
    - Market: require timestamp + price.
    - Weather: require timestamp + at least one of {solar,temp,wind}.
  - For ties or low coverage (< required), set role to Unknown and show guidance to the user.
  - Let the UI show a numeric compatibility score (e.g., "3/5 fields matched") and highlight which fields were auto-matched.

- **UX edge cases & heuristics**
  - Accept common column name variants (e.g., `ts`, `timestamp`, `utc_timestamp`; `pv_gen`, `solar_generation_actual`), using a small alias map.
  - Allow manual override even when auto-detect picked a role.
  - Persist last successful mapping per role as a preset for convenience.

- **Implementation notes (what I'll change)**
  - Update simulation-controls.tsx to:
    - Accept analyzer result `columns: string[]`.
    - Run a `detectRole(columns)` function implementing the scoring/alias rules.
    - Render role-specific mapping fields and dropdowns populated from `columns`.
    - Add role-override toggle, compatibility badge, and missing-field validation.
  - Validation: block the "Derive" action until required fields present for the selected role, with clear inline copy.

- **Small detection snippet (TypeScript)**
  - Pseudocode you can reuse:
    - detectRole(columns:string[]) -> { role: 'weather'|'household'|'market'|'unknown', matches: Record<string,string|null>, scores: Record }
    - Use alias map and required-fields checks inside.

- **Next step / offer**
  - I can implement this now in simulation-controls.tsx (update UI, add detectRole, prefill logic, validation + tests). Shall I proceed and make the changes, run `npm run typecheck` and `npm run build`? If yes, confirm any preferred column name aliases you want me to include (I’ll include sensible defaults: timestamp/utc_timestamp/ts, consumption/load/power_kw, solar_generation/solar_irradiance, wind_onshore_generation/wind_speed, temperature/temp, price/cost).