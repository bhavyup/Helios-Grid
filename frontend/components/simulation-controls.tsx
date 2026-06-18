"use client";

import {
  ArrowBigDownDash,
  ArrowRight,
  Database,
  FileUp,
  Loader2,
  Play,
  PlayCircle,
  RefreshCcw,
  RotateCcw,
  Sparkles,
  StepForward,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  CsvProfilePayload,
  CsvRole,
  CsvPathsPayload,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  PolicyMode,
  UploadedWeatherPayload,
  DerivedHouseholdPayload,
  DerivedMarketPayload,
  UploadedHouseholdPayload,
  UploadedMarketPayload,
} from "@/lib/types";
import { Portal } from "@/utils/create-portal";

type WeatherSourceMode = "default" | "path" | "upload";
type DataSourceMode = "default" | "path" | "upload";

type ControlTab =
  | "operate"
  | "weather"
  | "household"
  | "market"
  | "inspect"
  | "automation";

interface SimulationControlsProps {
  mode: PolicyMode;
  setMode: (mode: PolicyMode) => void;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;

  isBusy: boolean;

  // Weather refinery
  isProfilingCsv: boolean;
  isDerivingWeather: boolean;
  isUploadingWeather: boolean;

  // Household refinery
  isDerivingHousehold: boolean;
  isUploadingHousehold: boolean;

  // Market refinery
  isDerivingMarket: boolean;
  isUploadingMarket: boolean;

  isRunningDemo: boolean;

  csvSchemas: CsvSchemasPayload | null;
  csvPaths: CsvPathsPayload | null;
  csvProfile: CsvProfilePayload | null;

  derivedWeather: DerivedWeatherPayload | null;
  uploadedWeather: UploadedWeatherPayload | null;

  derivedHousehold: DerivedHouseholdPayload | null;
  uploadedHousehold: UploadedHouseholdPayload | null;

  derivedMarket: DerivedMarketPayload | null;
  uploadedMarket: UploadedMarketPayload | null;

  csvError: string | null;

  onReset: (input?: {
    seed?: number;
    weatherDataPath?: string;
    householdDataPath?: string;
    marketDataPath?: string;
    numHouseholds?: number;
    maxEpisodeSteps?: number;
  }) => Promise<void>;

  onStep: () => Promise<void>;
  onRun: (steps: number) => Promise<void>;
  onRefresh: () => Promise<void>;

  onAnalyzeCsv: (filePath: string, role: CsvRole) => Promise<void>;

  onDeriveWeatherFromCsv: (input: {
    file_path: string;
    solar_column: string;
    wind_column: string;
    timestamp_column?: string;
    temperature_column?: string;
    humidity_column?: string;
    irradiance_column?: string;
    ghi_column?: string;
    dni_column?: string;
    dhi_column?: string;
    pv_power_column?: string;
    panel_tilt?: number;
    panel_azimuth?: number;
    panel_area?: number;
    panel_efficiency?: number;
    temp_coefficient?: number;
    normalize_signals?: boolean;
  }) => Promise<void>;

  onDeriveHouseholdFromCsv: (input: {
    file_path: string;
    consumption_column: string;
    timestamp_column?: string;
    household_id_column?: string;
    pv_generation_column?: string;
    net_load_column?: string;
    normalize_signals?: boolean;
  }) => Promise<void>;

  onDeriveMarketFromCsv: (input: {
    file_path: string;
    price_column: string;
    supply_column?: string;
    demand_column?: string;
    timestamp_column?: string;
    bid_column?: string;
    ask_column?: string;
    clearing_price_column?: string;
    normalize_signals?: boolean;
  }) => Promise<void>;

  onUploadWeatherCsv: (input: { file: File }) => Promise<void>;
  onUploadHouseholdCsv: (input: { file: File }) => Promise<void>;
  onUploadMarketCsv: (input: { file: File }) => Promise<void>;

  onRunDemoSequence: () => Promise<void>;
  demoPhase?: string;
  demoProgress?: number;
}

function statusTone(mode: PolicyMode): string {
  return mode === "rule" ? "text-[#f6e7be]" : "text-[#d8f2eb]";
}

function sourceModeLabel(
  mode: DataSourceMode,
  domain: "weather" | "household" | "market",
): string {
  if (mode === "default") {
    return domain === "weather" ? "Bundled default" : "Synthetic baseline";
  }
  if (mode === "path") return "Existing path";
  return "Uploaded CSV";
}

function tabTone(active: boolean): string {
  return active
    ? "border-[rgba(212,175,55,0.36)] bg-[rgba(212,175,55,0.12)] text-[#f6e7be]"
    : "border-white/10 bg-[rgba(255,255,255,0.03)] text-slate-300 hover:border-white/20 hover:bg-[rgba(255,255,255,0.05)]";
}

interface SearchableColumnSelectProps {
  label: string;
  value: string;
  options: string[];
  placeholder?: string;
  suffix?: string;
  onChange: (value: string) => void;
}

function SearchableColumnSelect({
  label,
  value,
  options,
  placeholder = "-- optional --",
  suffix = "column",
  onChange,
}: SearchableColumnSelectProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (
        wrapperRef.current &&
        event.target instanceof Node &&
        !wrapperRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    if (isOpen) {
      setQuery(value);
    }
  }, [isOpen, value]);

  const filteredOptions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return options;
    }

    return options.filter((option) =>
      option.toLowerCase().includes(normalizedQuery),
    );
  }, [options, query]);

  return (
    <div
      ref={wrapperRef}
      className="relative grid min-w-0 gap-2 text-sm text-slate-200"
    >
      <span className="section-eyebrow text-[10px]">
        {label} {suffix}
      </span>
      <button
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        className="flex h-[56px] w-full min-w-0 items-center justify-between gap-3 overflow-hidden rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-left text-sm text-slate-100 outline-none transition hover:border-white/20 focus:border-[rgba(212,175,55,0.45)]"
      >
        <span
          className={`min-w-0 flex-1 truncate ${value ? "text-slate-100" : "text-slate-400"}`}
        >
          {value || placeholder}
        </span>
        <span className="flex-shrink-0 text-[10px] uppercase tracking-[0.18em] text-slate-400">
          {isOpen ? "Close" : "Search"}
        </span>
      </button>

      {isOpen ? (
        <div className="absolute left-0 top-[calc(100%+0.5rem)] z-30 w-full min-w-[18rem] rounded-2xl border border-white/10 bg-[#0f172a] p-3 shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
          <input
            autoFocus
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type to filter columns"
            className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-100 outline-none placeholder:text-slate-500 focus:border-[rgba(212,175,55,0.45)]"
          />
          <div className="modern-scrollbar mt-3 max-h-56 overflow-auto pr-1">
            <button
              type="button"
              onClick={() => {
                onChange("");
                setIsOpen(false);
              }}
              className="mb-2 flex w-full min-w-0 items-center justify-between rounded-xl border border-white/10 px-3 py-2 text-left text-slate-300 transition hover:border-white/20 hover:bg-white/[0.04]"
            >
              <span>{placeholder}</span>
              <span className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                Clear
              </span>
            </button>
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => {
                    onChange(option);
                    setIsOpen(false);
                  }}
                  className={`flex w-full min-w-0 items-center justify-between rounded-xl border px-3 py-2 text-left transition ${
                    option === value
                      ? "border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] text-[#f6e7be]"
                      : "border-white/10 text-slate-200 hover:border-white/20 hover:bg-white/[0.04]"
                  }`}
                >
                  <span className="min-w-0 flex-1 truncate break-words">
                    {option}
                  </span>
                  {option === value ? (
                    <span className="text-[10px] uppercase tracking-[0.16em]">
                      Selected
                    </span>
                  ) : null}
                </button>
              ))
            ) : (
              <div className="rounded-xl border border-white/10 px-3 py-3 text-xs text-slate-400">
                No columns match that search.
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

interface SearchablePathInputProps {
  label: string;
  value: string;
  options: string[];
  placeholder: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

function SearchablePathInput({
  label,
  value,
  options,
  placeholder,
  onChange,
  disabled = false,
}: SearchablePathInputProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState(value);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (
        wrapperRef.current &&
        event.target instanceof Node &&
        !wrapperRef.current.contains(event.target)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  const exactMatch = useMemo(
    () => options.find((option) => option === query.trim()) ?? null,
    [options, query],
  );

  const filteredOptions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return options;
    }

    return options.filter((option) =>
      option.toLowerCase().includes(normalizedQuery),
    );
  }, [options, query]);

  return (
    <div
      ref={wrapperRef}
      className="relative grid min-w-0 gap-2 text-sm text-slate-200"
    >
      <span className="section-eyebrow text-[10px]">{label}</span>
      <input
        type="text"
        value={query}
        disabled={disabled}
        onFocus={() => setIsOpen(true)}
        onClick={() => setIsOpen(true)}
        onChange={(event) => {
          const nextValue = event.target.value;
          setQuery(nextValue);
          onChange(nextValue);
          setIsOpen(true);
        }}
        onBlur={() => {
          const trimmedQuery = query.trim();
          const matchingOptions = options.filter((option) =>
            option.toLowerCase().includes(trimmedQuery.toLowerCase()),
          );
          if (!exactMatch && matchingOptions.length === 1) {
            setQuery(matchingOptions[0]);
            onChange(matchingOptions[0]);
          }
          setTimeout(() => setIsOpen(false), 120);
        }}
        placeholder={placeholder}
        title={value || placeholder}
        className="flex h-[56px] w-full min-w-0 items-center rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-[rgba(212,175,55,0.45)] disabled:cursor-not-allowed disabled:opacity-60 ont-mono overflow-hidden text-ellipsis whitespace-nowrap"
      />

      {isOpen ? (
        <div className="absolute left-0 top-[calc(100%+0.5rem)] !z-50 w-full min-w-[18rem] rounded-2xl border border-white/10 bg-[#0f172a] p-3 shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
          <div className="modern-scrollbar max-h-60 overflow-auto pr-1">
            <button
              type="button"
              title="Clear the current path"
              onClick={() => {
                setQuery("");
                onChange("");
                setIsOpen(false);
              }}
              className="mb-2 flex w-full min-w-0 items-center justify-between rounded-xl border border-white/10 px-3 py-2 text-left text-slate-300 transition hover:border-white/20 hover:bg-white/[0.04]"
            >
              <span>Clear</span>
              <span className="text-[10px] uppercase tracking-[0.16em] text-slate-500">
                Reset
              </span>
            </button>
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <button
                  key={option}
                  type="button"
                  title={option}
                  onClick={() => {
                    setQuery(option);
                    onChange(option);
                    setIsOpen(false);
                  }}
                  className={`flex w-full min-w-0 items-center justify-between rounded-xl border px-3 py-2 text-left transition ${
                    option === value
                      ? "border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] text-[#f6e7be]"
                      : "border-white/10 text-slate-200 hover:border-white/20 hover:bg-white/[0.04]"
                  }`}
                >
                  <span className="min-w-0 flex-1 truncate break-words">
                    {option}
                  </span>
                  {option === value ? (
                    <span className="text-[10px] uppercase tracking-[0.16em]">
                      Selected
                    </span>
                  ) : null}
                </button>
              ))
            ) : (
              <div className="rounded-xl border border-white/10 px-3 py-3 text-xs text-slate-400">
                No paths match that search.
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function SimulationControls({
  mode,
  setMode,
  autoRefresh,
  setAutoRefresh,
  isBusy,
  isProfilingCsv,
  isDerivingWeather,
  isUploadingWeather,

  isDerivingHousehold,
  isUploadingHousehold,

  isDerivingMarket,
  isUploadingMarket,

  isRunningDemo,
  csvSchemas,
  csvPaths,
  csvProfile,
  derivedWeather,
  uploadedWeather,

  derivedHousehold,
  uploadedHousehold,

  derivedMarket,
  uploadedMarket,

  csvError,
  onReset,
  onStep,
  onRun,
  onRefresh,
  onAnalyzeCsv,
  onDeriveWeatherFromCsv,
  onDeriveHouseholdFromCsv,
  onDeriveMarketFromCsv,
  onUploadWeatherCsv,
  onUploadHouseholdCsv,
  onUploadMarketCsv,
  onRunDemoSequence,
  demoPhase,
  demoProgress,
}: SimulationControlsProps): JSX.Element {
  const [activeTab, setActiveTab] = useState<ControlTab>("operate");
  const [runSteps, setRunSteps] = useState<number>(12);
  const [seedInput, setSeedInput] = useState<string>("");
  const [householdCountInput, setHouseholdCountInput] = useState<string>("64");
  const [maxEpisodeStepsInput, setMaxEpisodeStepsInput] = useState<string>("");
  const [weatherSourceMode, setWeatherSourceMode] =
    useState<WeatherSourceMode>("default");
  const [householdSourceMode, setHouseholdSourceMode] =
    useState<DataSourceMode>("default");
  const [marketSourceMode, setMarketSourceMode] =
    useState<DataSourceMode>("default");
  const [csvPath, setCsvPath] = useState<string>("");
  const [csvRole, setCsvRole] = useState<CsvRole>("auto");
  const [solarColumnInput, setSolarColumnInput] = useState<string>("");
  const [windColumnInput, setWindColumnInput] = useState<string>("");
  const [temperatureColumnInput, setTemperatureColumnInput] =
    useState<string>("");
  const [humidityColumnInput, setHumidityColumnInput] = useState<string>("");
  const [timestampColumnInput, setTimestampColumnInput] = useState<string>("");
  const [ghiColumnInput, setGhiColumnInput] = useState<string>("");
  const [dniColumnInput, setDniColumnInput] = useState<string>("");
  const [dhiColumnInput, setDhiColumnInput] = useState<string>("");
  const [irradianceColumnInput, setIrradianceColumnInput] =
    useState<string>("");
  const [pvPowerColumnInput, setPvPowerColumnInput] = useState<string>("");
  const [panelTiltInput, setPanelTiltInput] = useState<string>("");
  const [panelAzimuthInput, setPanelAzimuthInput] = useState<string>("");
  const [panelAreaInput, setPanelAreaInput] = useState<string>("");
  const [panelEfficiencyInput, setPanelEfficiencyInput] = useState<string>("");
  const [tempCoefficientInput, setTempCoefficientInput] = useState<string>("");
  const [confirmUnitMismatch, setConfirmUnitMismatch] =
    useState<boolean>(false);

  const [consumptionColumnInput, setConsumptionColumnInput] =
    useState<string>("");
  const [householdIdColumnInput, setHouseholdIdColumnInput] =
    useState<string>("");
  const [pvGenerationColumnInput, setPvGenerationColumnInput] =
    useState<string>("");
  const [netLoadColumnInput, setNetLoadColumnInput] = useState<string>("");
  const [meterColumnInput, setMeterColumnInput] = useState<string>("");
  const [demandColumnInput, setDemandColumnInput] = useState<string>("");

  const [priceColumnInput, setPriceColumnInput] = useState<string>("");
  const [askColumnInput, setAskColumnInput] = useState<string>("");
  const [bidColumnInput, setBidColumnInput] = useState<string>("");
  const [supplyColumnInput, setSupplyColumnInput] = useState("");
  const [marketDemandColumnInput, setMarketDemandColumnInput] = useState("");
  const [quantityColumnInput, setQuantityColumnInput] = useState<string>("");
  const [clearingPriceColumnInput, setClearingPriceColumnInput] =
    useState<string>("");

  const [householdPath, setHouseholdPath] = useState("");
  const [marketPath, setMarketPath] = useState("");
  const [WeatherPath, setWeatherPath] = useState("");

  const [selectedUploadFile, setSelectedUploadFile] = useState<File | null>(
    null,
  );
  const [selectedHouseholdUploadFile, setSelectedHouseholdUploadFile] =
    useState<File | null>(null);
  const [selectedMarketUploadFile, setSelectedMarketUploadFile] =
    useState<File | null>(null);

  const lastProfileAutoFillRef = useRef<string>("");

  const [openPvParameterModal, setOpenPvParameterModal] =
    useState<boolean>(false);

  function onClickPvParameters() {
    setOpenPvParameterModal(!openPvParameterModal);
  }

  const warningDetails = useMemo(
    () =>
      (csvProfile?.unit_warnings ?? []).map((warning) => {
        if (typeof warning === "string") {
          const warningText = String(warning);
          const columnMatch = warningText.match(/Column ['\"]([^'\"]+)['\"]/i);
          const extractedColumn = columnMatch?.[1] ?? "unknown";
          const stringKind = /irradiance.*not.*generation|not PV power/i.test(
            warningText,
          )
            ? "likely_irradiance_not_generation"
            : /generation.*not.*irradiance|not irradiance/i.test(warningText)
              ? "likely_generation_not_irradiance"
              : "legacy_warning";
          return {
            column: extractedColumn,
            kind: stringKind,
            message: warningText,
            suggestion:
              extractedColumn === "unknown"
                ? "Re-profile the CSV so the backend can emit field-level diagnostics."
                : "Review the mapped field and compare the column units before deriving.",
          };
        }

        return {
          column: warning.column ?? "unknown",
          kind: warning.kind ?? "legacy_warning",
          message:
            warning.message ??
            "This field was flagged during profile analysis.",
          suggestion:
            warning.suggestion ??
            "Re-check the column mapping and compare the units before deriving.",
        };
      }),
    [csvProfile?.unit_warnings],
  );

  const visibleWarningDetails = useMemo(() => warningDetails, [warningDetails]);

  const warningColumnCount = useMemo(
    () => new Set(warningDetails.map((warning) => warning.column)).size,
    [warningDetails],
  );
  const warningSignature = useMemo(
    () => JSON.stringify(warningDetails),
    [warningDetails],
  );

  const suggestedColumns = useMemo(() => {
    const columns = csvProfile?.columns ?? [];
    const solarPatterns = [
      /irradiance_wm2/i,
      /irradiance/i,
      /solar_irradiance/i,
      /ghi/i,
      /dni/i,
      /dhi/i,
      /pv_power/i,
    ];

    const windPatterns = [
      /wind(_|.*)generation_actual/i,
      /wind_profile/i,
      /wind/i,
      /wind_speed/i,
      /wind_power/i,
    ];

    const temperaturePatterns = [/temperature/i, /temp/i, /t2m/i, /air_temp/i];
    const humidityPatterns = [
      /humidity/i,
      /rh2m/i,
      /relative_humidity/i,
      /rh/i,
    ];
    const timestampPatterns = [
      /utc_timestamp/i,
      /timestamp/i,
      /date/i,
      /time/i,
    ];

    const findByPatterns = (patterns: RegExp[]) =>
      columns.find((columnName) => patterns.some((p) => p.test(columnName)));

    const solar = findByPatterns(solarPatterns);
    const wind = findByPatterns(windPatterns);
    const temperature = findByPatterns(temperaturePatterns);
    const humidity = findByPatterns(humidityPatterns);
    const timestamp = findByPatterns(timestampPatterns);

    const ghi = columns.find((c) => /ghi|dni|dhi/i.test(c));
    const dni = columns.find((c) => /dni/i.test(c));
    const dhi = columns.find((c) => /dhi/i.test(c));
    const irradiance = columns.find((c) =>
      /irradiance|irradiance_wm2/i.test(c),
    );
    const pvPower = columns.find((c) =>
      /pv_power|pv_power_kw|pv_power_w|pv_generation|pv_output|generation_actual|generation/i.test(
        c,
      ),
    );

    const consumption = columns.find((c) =>
      /consump|consumption|load/i.test(c),
    );
    const householdId = columns.find((c) =>
      /household_id|household|meter_id|customer_id/i.test(c),
    );
    const pvGeneration = columns.find((c) =>
      /pv_generation|pv|pv_output/i.test(c),
    );
    const netLoad = columns.find((c) => /net_load|netload/i.test(c));
    const meter = columns.find((c) => /meter|meter_id/i.test(c));

    const price = columns.find((c) =>
      /price|market_price|price_usd|price_eur/i.test(c),
    );
    const supply = columns.find((c) => /supply/i.test(c));
    const demand = columns.find((c) => /demand/i.test(c));
    const ask = columns.find((c) => /ask/i.test(c));
    const bid = columns.find((c) => /bid/i.test(c));
    const quantity = columns.find((c) => /quantity|qty|volume/i.test(c));
    const clearingPrice = columns.find((c) =>
      /clearing_price|clear_price/i.test(c),
    );
    return {
      solar,
      wind,
      temperature,
      humidity,
      timestamp,
      ghi,
      dni,
      dhi,
      irradiance,
      pvPower,
      consumption,
      householdId,
      pvGeneration,
      netLoad,
      meter,
      price,
      ask,
      bid,
      quantity,
      clearingPrice,
      supply,
      demand,
    };
  }, [csvProfile]);

  // helper: detect role by scoring how many expected fields match the CSV columns
  const detectRoleFromColumns = (columns: string[] | undefined): CsvRole => {
    if (!columns || columns.length === 0) return "auto";
    const colsLower = columns.map((c) => c.toLowerCase());

    const roleFields: Record<CsvRole, string[]> = {
      auto: [],
      weather: [
        "solar_irradiance",
        "irradiance",
        "wind",
        "wind_speed",
        "temperature",
        "temp",
        "humidity",
        "timestamp",
        "utc_timestamp",
        "solar_irradiance",
      ],
      household: [
        "consump",
        "consumption",
        "load",
        "net_load",
        "household",
        "household_id",
        "pv_generation",
        "pv",
        "meter",
        "demand",
        "timestamp",
        "utc_timestamp",
      ],
      market: [
        "price",
        "ask",
        "bid",
        "market_price",
        "clearing_price",
        "price_usd",
        "price_eur",
        "timestamp",
        "utc_timestamp",
        "supply",
        "demand",
      ],
    };

    const scoreFor = (role: CsvRole) => {
      const fields = roleFields[role] ?? [];
      let score = 0;
      for (const f of fields) {
        // count a field if any column includes the substring
        if (colsLower.some((c) => c.includes(f))) score += 1;
      }
      return score;
    };

    const scores: Record<CsvRole, number> = {
      auto: 0,
      weather: scoreFor("weather"),
      household: scoreFor("household"),
      market: scoreFor("market"),
    };

    // pick role with max score; if tie or all zero -> auto
    const entries = (Object.keys(scores) as CsvRole[]).map(
      (r) => [r, scores[r]] as const,
    );
    entries.sort((a, b) => b[1] - a[1]);
    const [bestRole, bestScore] = entries[0];
    const secondScore = entries[1]?.[1] ?? 0;

    if (bestScore === 0 || bestScore === secondScore) return "auto";
    return bestRole;
  };

  useEffect(() => {
    if (!csvProfile) {
      return;
    }

    const profileKey = `${csvProfile.resolved_path}|${csvProfile.selected_role}|${csvProfile.columns.join("|")}`;
    if (lastProfileAutoFillRef.current === profileKey) {
      return;
    }
    lastProfileAutoFillRef.current = profileKey;

    // Reset role-specific mappings when a new CSV profile arrives.
    setSolarColumnInput("");
    setWindColumnInput("");
    setTimestampColumnInput("");
    setTemperatureColumnInput("");
    setHumidityColumnInput("");
    setGhiColumnInput("");
    setDniColumnInput("");
    setDhiColumnInput("");
    setIrradianceColumnInput("");
    setPvPowerColumnInput("");
    setConsumptionColumnInput("");
    setHouseholdIdColumnInput("");
    setPvGenerationColumnInput("");
    setNetLoadColumnInput("");
    setMeterColumnInput("");
    setDemandColumnInput("");
    setPriceColumnInput("");
    setAskColumnInput("");
    setBidColumnInput("");
    setQuantityColumnInput("");
    setClearingPriceColumnInput("");

    const columns = csvProfile.columns ?? [];
    let nextIrradiance = suggestedColumns.irradiance ?? "";
    let nextPvPower = suggestedColumns.pvPower ?? "";

    warningDetails.forEach((warning) => {
      const flagged = warning.column?.trim();
      if (!flagged || flagged === "unknown") {
        return;
      }

      if (warning.kind === "likely_generation_not_irradiance") {
        if (nextIrradiance.toLowerCase() === flagged.toLowerCase()) {
          if (!nextPvPower) {
            nextPvPower = flagged;
          }
          nextIrradiance = findIrradianceAlternative(columns, flagged);
        }
      }

      if (warning.kind === "likely_irradiance_not_generation") {
        if (nextPvPower.toLowerCase() === flagged.toLowerCase()) {
          if (!nextIrradiance) {
            nextIrradiance = flagged;
          }
          nextPvPower = findPvPowerAlternative(columns, flagged);
        }
      }
    });

    setSolarColumnInput(suggestedColumns.solar ?? "");
    setWindColumnInput(suggestedColumns.wind ?? "");
    setTimestampColumnInput(suggestedColumns.timestamp ?? "");
    setTemperatureColumnInput(suggestedColumns.temperature ?? "");
    setHumidityColumnInput(suggestedColumns.humidity ?? "");
    setGhiColumnInput(suggestedColumns.ghi ?? "");
    setDniColumnInput(suggestedColumns.dni ?? "");
    setDhiColumnInput(suggestedColumns.dhi ?? "");
    setIrradianceColumnInput(nextIrradiance);
    setPvPowerColumnInput(nextPvPower);

    if (!panelTiltInput) {
      setPanelTiltInput("30");
    }
    if (!panelAzimuthInput) {
      setPanelAzimuthInput("180");
    }
    if (!panelAreaInput) {
      setPanelAreaInput("1.0");
    }
    if (!panelEfficiencyInput) {
      setPanelEfficiencyInput("0.15");
    }
    if (!tempCoefficientInput) {
      setTempCoefficientInput("-0.004");
    }

    applySmartAutoFillFromProfile();
  }, [
    csvProfile,
    panelTiltInput,
    panelAzimuthInput,
    panelAreaInput,
    panelEfficiencyInput,
    tempCoefficientInput,
    suggestedColumns,
    warningSignature,
    warningDetails,
  ]);

  useEffect(() => {
    setConfirmUnitMismatch(false);
  }, [csvProfile?.resolved_path, warningSignature, csvProfile?.selected_role]);

  // When a new profile is available, auto-select role if csvRole is 'auto'
  useEffect(() => {
    if (!csvProfile) return;
    if (csvRole === "auto") {
      const inferred = detectRoleFromColumns(csvProfile.columns);
      if (inferred !== "auto") {
        // don't override explicit user choice; only set if we inferred
        setCsvRole(inferred);
      }
    }
  }, [csvProfile, csvRole]);

  const canDeriveWeather = Boolean(
    csvProfile?.resolved_path &&
    solarColumnInput.trim() &&
    windColumnInput.trim() &&
    (!(csvProfile?.unit_warnings?.length ?? 0) || confirmUnitMismatch),
  );

  const effectiveRole =
    csvRole === "auto" ? (csvProfile?.inferred_role ?? "auto") : csvRole;

  const backendCsvPathOptions = useMemo(() => {
    const paths = new Set<string>();

    const addPath = (pathValue?: string | null) => {
      if (pathValue && pathValue.trim().length > 0) {
        paths.add(pathValue.trim());
      }
    };

    csvPaths?.paths.forEach((option) => addPath(option.path));
    addPath(uploadedWeather?.resolved_path);
    addPath(csvProfile?.file_path);
    addPath(csvProfile?.resolved_path);
    addPath(derivedWeather?.source_file_path);
    addPath(derivedWeather?.resolved_source_path);
    addPath(derivedWeather?.output_file_path);
    addPath(uploadedHousehold?.resolved_path);
    addPath(uploadedMarket?.resolved_path);
    addPath(derivedHousehold?.output_file_path);
    addPath(derivedMarket?.output_file_path);

    return Array.from(paths);
  }, [
    csvPaths,
    csvProfile,
    derivedWeather,
    uploadedWeather,
    uploadedHousehold,
    uploadedMarket,
    derivedHousehold,
    derivedMarket,
  ]);

  useEffect(() => {
    if (!uploadedWeather?.resolved_path) {
      return;
    }

    setWeatherPath(uploadedWeather.resolved_path);
    setWeatherSourceMode("upload");
    setSelectedUploadFile(null);
  }, [uploadedWeather]);

  useEffect(() => {
    if (!derivedWeather?.output_file_path) {
      return;
    }

    setWeatherPath(derivedWeather.output_file_path);
    setWeatherSourceMode("path");
  }, [derivedWeather?.output_file_path]);

  useEffect(() => {
    if (!uploadedHousehold?.resolved_path) return;
    setHouseholdPath(uploadedHousehold.resolved_path);
    setHouseholdSourceMode("upload");
    setSelectedHouseholdUploadFile(null);
  }, [uploadedHousehold]);

  useEffect(() => {
    if (!uploadedMarket?.resolved_path) return;
    setMarketPath(uploadedMarket.resolved_path);
    setMarketSourceMode("upload");
    setSelectedMarketUploadFile(null);
  }, [uploadedMarket]);

  useEffect(() => {
    if (!derivedHousehold?.output_file_path) return;
    setHouseholdPath(derivedHousehold.output_file_path);
    setHouseholdSourceMode("path");
  }, [derivedHousehold?.output_file_path]);

  useEffect(() => {
    if (!derivedMarket?.output_file_path) return;
    setMarketPath(derivedMarket.output_file_path);
    setMarketSourceMode("path");
  }, [derivedMarket?.output_file_path]);

  const resolvedWeatherPath =
    weatherSourceMode === "upload"
      ? (uploadedWeather?.resolved_path ?? (WeatherPath.trim() || undefined))
      : WeatherPath.trim() || undefined;

  const resolvedHouseholdPath =
    householdSourceMode === "upload"
      ? (uploadedHousehold?.resolved_path ??
        (householdPath.trim() || undefined))
      : householdPath.trim() || undefined;

  const resolvedMarketPath =
    marketSourceMode === "upload"
      ? (uploadedMarket?.resolved_path ?? (marketPath.trim() || undefined))
      : marketPath.trim() || undefined;

  const parseOptionalInt = (
    value: string,
    minValue: number,
    maxValue: number,
  ): number | undefined => {
    if (!value.trim()) {
      return undefined;
    }

    const parsed = Math.floor(Number(value));
    if (!Number.isFinite(parsed)) {
      return undefined;
    }

    return Math.min(Math.max(parsed, minValue), maxValue);
  };

  const resolveResetInputs = () => {
    return {
      parsedHouseholds: parseOptionalInt(householdCountInput, 1, 256),
      parsedMaxEpisodeSteps: parseOptionalInt(maxEpisodeStepsInput, 1, 100000),
    };
  };

  const handleReset = async () => {
    const maybeSeed =
      seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN)
      ? maybeSeed
      : undefined;
    const { parsedHouseholds, parsedMaxEpisodeSteps } = resolveResetInputs();
    await onReset({
      seed: parsedSeed,
      weatherDataPath: resolvedWeatherPath,
      householdDataPath: resolvedHouseholdPath,
      marketDataPath: resolvedMarketPath,
      numHouseholds: parsedHouseholds,
      maxEpisodeSteps: parsedMaxEpisodeSteps,
    });
  };

  const handleAnalyzeCsv = async () => {
    const resolvedPath = csvPath.trim();
    if (!resolvedPath) {
      return;
    }

    await onAnalyzeCsv(resolvedPath, csvRole);
  };

  const handleResetWithProfile = async () => {
    if (!csvProfile?.can_use_now) {
      return;
    }

    const maybeSeed =
      seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN)
      ? maybeSeed
      : undefined;
    const { parsedHouseholds, parsedMaxEpisodeSteps } = resolveResetInputs();
    await onReset({
      seed: parsedSeed,
      weatherDataPath:
        effectiveRole === "weather"
          ? csvProfile.resolved_path
          : resolvedWeatherPath,
      householdDataPath:
        effectiveRole === "household"
          ? csvProfile.resolved_path
          : resolvedHouseholdPath,
      marketDataPath:
        effectiveRole === "market"
          ? csvProfile.resolved_path
          : resolvedMarketPath,
      numHouseholds: parsedHouseholds,
      maxEpisodeSteps: parsedMaxEpisodeSteps,
    });
    if (effectiveRole === "weather") {
      setWeatherSourceMode("path");
      setWeatherPath(csvProfile.resolved_path);
    }
    if (effectiveRole === "household") {
      setHouseholdSourceMode("path");
      setHouseholdPath(csvProfile.resolved_path);
    }
    if (effectiveRole === "market") {
      setMarketSourceMode("path");
      setMarketPath(csvProfile.resolved_path);
    }
    setCsvPath(csvProfile.resolved_path);
  };

  const handleDeriveWeather = async () => {
    const filePath = csvProfile?.resolved_path ?? csvPath.trim();
    const solarColumn = solarColumnInput.trim();
    const windColumn = windColumnInput.trim();
    const temperatureColumn = temperatureColumnInput.trim();
    const humidityColumn = humidityColumnInput.trim();
    const timestampColumn = timestampColumnInput.trim();
    const irradianceColumn = irradianceColumnInput.trim();
    const ghiColumn = ghiColumnInput.trim();
    const dniColumn = dniColumnInput.trim();
    const dhiColumn = dhiColumnInput.trim();
    const pvPowerColumn = pvPowerColumnInput.trim();
    const panelTilt = panelTiltInput.trim();
    const panelAzimuth = panelAzimuthInput.trim();
    const panelArea = panelAreaInput.trim();
    const panelEfficiency = panelEfficiencyInput.trim();
    const tempCoefficient = tempCoefficientInput.trim();

    const parseOptionalNumber = (value: string): number | undefined => {
      if (!value) {
        return undefined;
      }

      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : undefined;
    };

    if (!filePath || !solarColumn || !windColumn) {
      return;
    }

    await onDeriveWeatherFromCsv({
      file_path: filePath,
      solar_column: solarColumn,
      wind_column: windColumn,
      timestamp_column:
        timestampColumn.length > 0 ? timestampColumn : undefined,
      temperature_column:
        temperatureColumn.length > 0 ? temperatureColumn : undefined,
      humidity_column: humidityColumn.length > 0 ? humidityColumn : undefined,
      irradiance_column:
        irradianceColumn.length > 0 ? irradianceColumn : undefined,
      ghi_column: ghiColumn.length > 0 ? ghiColumn : undefined,
      dni_column: dniColumn.length > 0 ? dniColumn : undefined,
      dhi_column: dhiColumn.length > 0 ? dhiColumn : undefined,
      pv_power_column: pvPowerColumn.length > 0 ? pvPowerColumn : undefined,
      panel_tilt: parseOptionalNumber(panelTilt),
      panel_azimuth: parseOptionalNumber(panelAzimuth),
      panel_area: parseOptionalNumber(panelArea),
      panel_efficiency: parseOptionalNumber(panelEfficiency),
      temp_coefficient: parseOptionalNumber(tempCoefficient),
      normalize_signals: true,
    });
  };

  const handleDeriveHousehold = async () => {
    const filePath = csvProfile?.resolved_path ?? csvPath.trim();
    if (!filePath || !consumptionColumnInput.trim()) return;

    await onDeriveHouseholdFromCsv({
      file_path: filePath,
      consumption_column: consumptionColumnInput.trim(),
      timestamp_column: timestampColumnInput.trim() || undefined,
      household_id_column: householdIdColumnInput.trim() || undefined,
      pv_generation_column: pvGenerationColumnInput.trim() || undefined,
      net_load_column: netLoadColumnInput.trim() || undefined,
      normalize_signals: false,
    });
  };

  const handleDeriveMarket = async () => {
    const filePath = csvProfile?.resolved_path ?? csvPath.trim();
    if (!filePath || !priceColumnInput.trim()) return;

    await onDeriveMarketFromCsv({
      file_path: filePath,
      price_column: priceColumnInput.trim(),
      supply_column: supplyColumnInput.trim() || undefined,
      demand_column: demandColumnInput.trim() || undefined,
      timestamp_column: timestampColumnInput.trim() || undefined,
      bid_column: bidColumnInput.trim() || undefined,
      ask_column: askColumnInput.trim() || undefined,
      clearing_price_column: clearingPriceColumnInput.trim() || undefined,
      normalize_signals: false,
    });
  };

  const handleResetWithDerivedWeather = async () => {
    if (!derivedWeather?.output_file_path) {
      return;
    }

    const maybeSeed =
      seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN)
      ? maybeSeed
      : undefined;
    const { parsedHouseholds, parsedMaxEpisodeSteps } = resolveResetInputs();
    await onReset({
      seed: parsedSeed,
      weatherDataPath: derivedWeather.output_file_path,
      householdDataPath: resolvedHouseholdPath,
      marketDataPath: resolvedMarketPath,
      numHouseholds: parsedHouseholds,
      maxEpisodeSteps: parsedMaxEpisodeSteps,
    });
    setWeatherSourceMode("path");
    setCsvPath(derivedWeather.output_file_path);
    setWeatherPath(derivedWeather.output_file_path);
  };

  const handleResetWithDerivedHousehold = async () => {
    if (!derivedHousehold?.output_file_path) {
      return;
    }

    const maybeSeed =
      seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN)
      ? maybeSeed
      : undefined;
    const { parsedHouseholds, parsedMaxEpisodeSteps } = resolveResetInputs();
    await onReset({
      seed: parsedSeed,
      weatherDataPath: resolvedWeatherPath,
      marketDataPath: resolvedMarketPath,
      householdDataPath: derivedHousehold.output_file_path,
      numHouseholds: parsedHouseholds,
      maxEpisodeSteps: parsedMaxEpisodeSteps,
    });
    setHouseholdSourceMode("path");
    setCsvPath(derivedHousehold.output_file_path);
    setHouseholdPath(derivedHousehold.output_file_path);
  };

  const handleResetWithDerivedMarket = async () => {
    if (!derivedMarket?.output_file_path) {
      return;
    }

    const maybeSeed =
      seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN)
      ? maybeSeed
      : undefined;
    const { parsedHouseholds, parsedMaxEpisodeSteps } = resolveResetInputs();
    await onReset({
      seed: parsedSeed,
      weatherDataPath: resolvedWeatherPath,
      householdDataPath: resolvedHouseholdPath,
      marketDataPath: derivedMarket.output_file_path,
      numHouseholds: parsedHouseholds,
      maxEpisodeSteps: parsedMaxEpisodeSteps,
    });
    setMarketSourceMode("path");
    setCsvPath(derivedMarket.output_file_path);
    setMarketPath(derivedMarket.output_file_path);
  };

  // expose derived column mapping to user by showing a compact summary
  const derivedWeatherMappingSummary = useMemo(() => {
    if (!derivedWeather?.column_mapping) return null;
    const entries = Object.entries(derivedWeather.column_mapping)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k} <-- ${v}`);
    return entries.length > 0 ? entries.join(" · ") : null;
  }, [derivedWeather]);

  const derivedHouseholdMappingSummary = useMemo(() => {
    if (!derivedHousehold?.column_mapping) return null;
    const entries = Object.entries(derivedHousehold.column_mapping)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k} <-- ${v}`);
    return entries.length > 0 ? entries.join(" · ") : null;
  }, [derivedHousehold]);

  const derivedMarketMappingSummary = useMemo(() => {
    if (!derivedMarket?.column_mapping) return null;
    const entries = Object.entries(derivedMarket.column_mapping)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k} <-- ${v}`);
    return entries.length > 0 ? entries.join(" · ") : null;
  }, [derivedMarket]);

  const setCSVPathAfterUpload = (type: "weather" | "household" | "market") => {
    if (type === "weather") {
      setCsvPath(uploadedWeather?.resolved_path ?? "");
      setWeatherPath(uploadedWeather?.resolved_path ?? "");
      setWeatherSourceMode("upload");
    } else if (type === "household") {
      setCsvPath(uploadedHousehold?.resolved_path ?? "");
      setHouseholdPath(uploadedHousehold?.resolved_path ?? "");
      setHouseholdSourceMode("upload");
    } else if (type === "market") {
      setCsvPath(uploadedMarket?.resolved_path ?? "");
      setMarketPath(uploadedMarket?.resolved_path ?? "");
      setMarketSourceMode("upload");
    }
  };

  const warningByColumn = useMemo(() => {
    const map = new Map<string, (typeof warningDetails)[number]>();
    warningDetails.forEach((warning) => {
      if (warning.column) {
        map.set(warning.column.toLowerCase(), warning);
      }
    });
    return map;
  }, [warningDetails]);

  const getWarningForValue = (value: string) =>
    warningByColumn.get(value.trim().toLowerCase()) ?? null;

  const findIrradianceAlternative = (
    columns: string[],
    excluded: string,
  ): string => {
    const candidates = columns.filter(
      (column) => column.toLowerCase() !== excluded.toLowerCase(),
    );
    return (
      candidates.find((column) =>
        /irradiance|solar_irradiance|ghi|dni|dhi/i.test(column),
      ) ?? ""
    );
  };

  const findPvPowerAlternative = (
    columns: string[],
    excluded: string,
  ): string => {
    const candidates = columns.filter(
      (column) => column.toLowerCase() !== excluded.toLowerCase(),
    );
    return (
      candidates.find((column) =>
        /pv_power|pv_generation|pv_output|generation/i.test(column),
      ) ?? ""
    );
  };

  const applySmartAutoFillFromProfile = () => {
    if (!csvProfile) {
      return;
    }

    const columns = csvProfile.columns ?? [];
    const nextSolar = suggestedColumns.solar ?? "";
    const nextWind = suggestedColumns.wind ?? "";
    const nextTimestamp = suggestedColumns.timestamp ?? "";
    const nextTemperature = suggestedColumns.temperature ?? "";
    const nextHumidity = suggestedColumns.humidity ?? "";
    const consumption = suggestedColumns.consumption ?? "";
    const householdId = suggestedColumns.householdId ?? "";
    const pvGeneration = suggestedColumns.pvGeneration ?? "";
    const netLoad = suggestedColumns.netLoad ?? "";
    const price = suggestedColumns.price ?? "";
    const ask = suggestedColumns.ask ?? "";
    const bid = suggestedColumns.bid ?? "";
    const supply = suggestedColumns.supply ?? "";
    const demand = suggestedColumns.demand ?? "";
    const quantity = suggestedColumns.quantity ?? "";
    const clearingPrice = suggestedColumns.clearingPrice ?? "";

    let nextGhi = suggestedColumns.ghi ?? "";
    let nextDni = suggestedColumns.dni ?? "";
    let nextDhi = suggestedColumns.dhi ?? "";
    let nextIrradiance = suggestedColumns.irradiance ?? "";
    let nextPvPower = suggestedColumns.pvPower ?? "";

    warningDetails.forEach((warning) => {
      const flagged = warning.column?.trim();
      if (!flagged || flagged === "unknown") {
        return;
      }

      if (warning.kind === "likely_generation_not_irradiance") {
        if (
          nextIrradiance.toLowerCase() === flagged.toLowerCase() ||
          nextSolar.toLowerCase() === flagged.toLowerCase()
        ) {
          if (!nextPvPower) {
            nextPvPower = flagged;
          }
          if (nextIrradiance.toLowerCase() === flagged.toLowerCase()) {
            nextIrradiance = findIrradianceAlternative(columns, flagged);
          }
        }
      }

      if (warning.kind === "likely_irradiance_not_generation") {
        if (nextPvPower.toLowerCase() === flagged.toLowerCase()) {
          if (!nextIrradiance) {
            nextIrradiance = flagged;
          }
          nextPvPower = findPvPowerAlternative(columns, flagged);
        }
      }
    });

    setSolarColumnInput(nextSolar);
    setWindColumnInput(nextWind);
    setTimestampColumnInput(nextTimestamp);
    setTemperatureColumnInput(nextTemperature);
    setHumidityColumnInput(nextHumidity);
    setGhiColumnInput(nextGhi);
    setDniColumnInput(nextDni);
    setDhiColumnInput(nextDhi);
    setIrradianceColumnInput(nextIrradiance);
    setPvPowerColumnInput(nextPvPower);
    setConsumptionColumnInput(consumption);
    setHouseholdIdColumnInput(householdId);
    setPvGenerationColumnInput(pvGeneration);
    setNetLoadColumnInput(netLoad);
    setPriceColumnInput(price);
    setAskColumnInput(ask);
    setBidColumnInput(bid);
    setSupplyColumnInput(supply);
    setMarketDemandColumnInput(demand);
    setQuantityColumnInput(quantity);
    setClearingPriceColumnInput(clearingPrice);
  };

  const applySafeWeatherFixes = (): void => {
    applySmartAutoFillFromProfile();
    setConfirmUnitMismatch(false);
  };

  const renderSelectWithWarning = (
    label: string,
    value: string,
    setter: (value: string) => void,
  ): JSX.Element => {
    const warning = getWarningForValue(value);
    return (
      <label key={label} className="grid gap-2 text-sm text-slate-200">
        <div className="flex items-center justify-between gap-3">
          {/* <span className="section-eyebrow text-[10px]">{label}</span> */}
          {warning ? (
            <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-amber-200">
              Review field
            </span>
          ) : null}
        </div>
        <div
          className={
            warning
              ? "rounded-2xl border border-amber-300/30 bg-amber-300/5 p-2"
              : ""
          }
        >
          <SearchableColumnSelect
            label={label}
            value={value}
            options={csvProfile?.columns ?? []}
            placeholder="-- optional --"
            onChange={setter}
          />
        </div>
      </label>
    );
  };

  function onViewDemoSequence() {
    const sceneElement = document.getElementById("3d-scene");
    if (sceneElement) {
      sceneElement.scrollIntoView({ behavior: "smooth" });
    }
  }

  const handleUploadWeather = async () => {
    if (!selectedUploadFile) {
      return;
    }

    await onUploadWeatherCsv({ file: selectedUploadFile });
  };

  const handleUploadHousehold = async () => {
    if (!selectedHouseholdUploadFile) return;
    await onUploadHouseholdCsv({ file: selectedHouseholdUploadFile });
  };

  const handleUploadMarket = async () => {
    if (!selectedMarketUploadFile) return;
    await onUploadMarketCsv({ file: selectedMarketUploadFile });
  };

  const weatherSourceCards: Array<{
    id: WeatherSourceMode;
    title: string;
    description: string;
  }> = [
    {
      id: "default",
      title: "Bundled default",
      description:
        "Use the embedded sample source for a quick reset and immediate demo flow.",
    },
    {
      id: "path",
      title: "Existing CSV path",
      description:
        "Point the sim at any uploaded or derived backend path that already exists.",
    },
    {
      id: "upload",
      title: "Upload local CSV",
      description:
        "Send a local file to the backend, then promote the resolved path into the reset flow.",
    },
  ];

  const controlTabs: Array<{
    id: ControlTab;
    label: string;
    description: string;
  }> = [
    { id: "operate", label: "Operate", description: "Mode, cadence, reset" },
    { id: "weather", label: "Weather", description: "Source, upload, path" },
    {
      id: "household",
      label: "Household",
      description: "Source, upload, path",
    },
    { id: "market", label: "Market", description: "Source, upload, path" },
    { id: "inspect", label: "Inspect", description: "Profile, derive, verify" },
    { id: "automation", label: "Automation", description: "Demo sequence" },
  ];

  return (
    <section className="panel-surface overflow-visible">
      <div className="absolute inset-0 surface-grid" />
      <div className="relative z-10 border-b border-white/10 px-6 py-6 lg:px-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <p className="section-eyebrow">Simulation workstation</p>
            </div>
            <h2 className="hero-display text-3xl font-semibold tracking-[-0.03em] text-white lg:text-4xl">
              Weather Refinery
            </h2>
          </div>

          <div className="flex flex-col">
            <div className=" flex items-center justify-end gap-3">
              <span
                className={`rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] ${statusTone(mode)}`}
              >
                {mode === "rule" ? "Rule live" : "PPO preview"}
              </span>
              <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                {autoRefresh ? "Auto poll: on" : "Auto poll: off"}
              </span>
              <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                Weather: {sourceModeLabel(weatherSourceMode, "weather")}
              </span>
              <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                Household: {sourceModeLabel(householdSourceMode, "household")}
              </span>
              <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                Market: {sourceModeLabel(marketSourceMode, "market")}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap justify-end gap-2">
              {controlTabs.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`rounded-full border px-4 py-3 text-left transition ${tabTone(activeTab === tab.id)}`}
                >
                  <span className="block text-xs font-semibold uppercase tracking-[0.18em]">
                    {tab.label}
                  </span>
                  <span className="block text-[9px] uppercase tracking-[0.12em] opacity-75">
                    {tab.description}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {activeTab === "operate" ? (
        <div className="relative z-50 grid gap-6 px-6 py-6 lg:px-8 xl:grid-cols-[0.98fr_1.02fr]">
          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="grid gap-3 md:grid-cols-2">
                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Policy mode
                  </span>
                  <select
                    value={mode}
                    onChange={(event) =>
                      setMode(event.target.value as PolicyMode)
                    }
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
                  >
                    <option className="bg-slate-800" value="rule">
                      Rule Mode (Live)
                    </option>
                    <option className="bg-slate-800" value="ppo-preview">
                      PPO Preview (Analytics)
                    </option>
                  </select>
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">Run steps</span>
                  <input
                    type="number"
                    min={1}
                    max={240}
                    value={runSteps}
                    onChange={(event) =>
                      setRunSteps(Math.max(1, Number(event.target.value) || 1))
                    }
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
                  />
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Households
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={256}
                    value={householdCountInput}
                    onChange={(event) =>
                      setHouseholdCountInput(event.target.value)
                    }
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
                  />
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Episode steps
                  </span>
                  <input
                    type="number"
                    min={1}
                    max={100000}
                    placeholder="1000"
                    value={maxEpisodeStepsInput}
                    onChange={(event) =>
                      setMaxEpisodeStepsInput(event.target.value)
                    }
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
                  />
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Reset seed
                  </span>
                  <input
                    type="number"
                    placeholder="42"
                    value={seedInput}
                    onChange={(event) => setSeedInput(event.target.value)}
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
                  />
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow block text-[10px]">
                    Refresh loop
                  </span>
                  <div className="flex items-center gap-2 w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]">
                    <input
                      type="checkbox"
                      checked={autoRefresh}
                      onChange={(event) => setAutoRefresh(event.target.checked)}
                      className="h-4 w-4 rounded border-white/20 bg-transparent text-[#d4af37] mb-1"
                    />
                    <span>Auto poll</span>
                  </div>
                </label>
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Command path</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Operation sequence
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  Ready
                </span>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {[
                  [
                    "Reset",
                    "Hydrate the grid with the selected weather source.",
                  ],
                  ["Step", "Advance the episode one timestep for inspection."],
                  [
                    "Run",
                    "Execute a short live sequence without leaving this card.",
                  ],
                ].map(([title, description]) => (
                  <article
                    key={title}
                    className="panel-frame rounded-[1.1rem] px-4 py-4"
                  >
                    <p className="text-[11px] uppercase tracking-[0.22em] text-[#f6e7be]">
                      {title}
                    </p>
                    <p className="mt-3 text-xs leading-6 text-slate-300">
                      {description}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Execution flow</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void handleReset()}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset
                </button>

                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onStep()}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <StepForward className="h-4 w-4" />
                  Step once
                </button>

                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onRun(runSteps)}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-sm font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <PlayCircle className="h-4 w-4" />
                  Run {runSteps} steps
                </button>

                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onRefresh()}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RefreshCcw className="h-4 w-4" />
                  Refresh snapshot
                </button>
              </div>
              <div className="mt-4 flex flex-wrap justify-between gap-2 text-[10px] uppercase tracking-[0.18em] text-slate-300">
                <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">
                  Policy toggles preserved
                </span>
                <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">
                  Seeded resets supported
                </span>
                <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">
                  Live refresh optional
                </span>
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Live context</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Weather source
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {sourceModeLabel(weatherSourceMode, "weather")}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300 break-all">
                    {resolvedWeatherPath ? (
                      <>
                        {" "}
                        Path:{" "}
                        <span className="text-slate-100 hover:underline hover:cursor-pointer">
                          {resolvedWeatherPath}
                        </span>
                      </>
                    ) : (
                      "Bundled default is active until you choose a file or path."
                    )}
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Household source
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {sourceModeLabel(householdSourceMode, "household")}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300 break-all">
                    {resolvedHouseholdPath ? (
                      <>
                        {" "}
                        Path:{" "}
                        <span className="text-slate-100 hover:underline hover:cursor-pointer">
                          {resolvedHouseholdPath}
                        </span>
                      </>
                    ) : (
                      "Bundled default is active until you choose a file or path."
                    )}
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Market source
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {sourceModeLabel(marketSourceMode, "market")}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300 break-all">
                    {resolvedMarketPath ? (
                      <>
                        {" "}
                        Path:{" "}
                        <span className="text-slate-100 hover:underline hover:cursor-pointer">
                          {resolvedMarketPath}
                        </span>
                      </>
                    ) : (
                      "Bundled default is active until you choose a file or path."
                    )}
                  </p>
                </article>
                {/* <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Weather profile
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {csvProfile?.can_use_now
                      ? "Runtime ready"
                      : "Profile first"}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {csvProfile?.usage_recommendation ??
                      "Profile a CSV to verify the schema before it becomes the reset source."}
                  </p>
                  
                </article> */}
              </div>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "weather" ? (
        <div className="relative z-50 grid gap-6 px-6 py-6 lg:px-8 xl:grid-cols-[1.02fr_0.98fr]">
          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Weather source</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Choose the source with intent.
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  {sourceModeLabel(weatherSourceMode, "weather")}
                </span>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {weatherSourceCards.map((card) => {
                  const active = weatherSourceMode === card.id;
                  return (
                    <button
                      key={card.id}
                      type="button"
                      onClick={() => setWeatherSourceMode(card.id)}
                      className={`rounded-[1.15rem] border px-4 py-4 text-left transition ${
                        active
                          ? "border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.08)]"
                          : "border-white/10 bg-[rgba(255,255,255,0.03)] hover:border-white/20 hover:bg-[rgba(255,255,255,0.05)]"
                      }`}
                    >
                      <p className="text-sm font-semibold text-white">
                        {card.title}
                      </p>
                      {/* <p className="mt-2 text-sm leading-6 text-slate-300">
                        {card.description}
                      </p> */}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Source map</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Bundled default
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Good for quick resets and demo execution without extra file
                    handling.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Existing path
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Use already uploaded or derived backend paths when they
                    exist.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Upload local CSV
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Send a local file to the backend, then reset against the
                    resolved path.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Reset path
                  </p>
                  <p className="mt-2 text-xs break-all leading-6 text-slate-300">
                    Choose a mode and enter a path to make the reset target
                    explicit.
                  </p>
                </article>
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Resolved Reset path</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2"></div>
              <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                <p className="mt-2 text-sm font-semibold text-white">
                  {resolvedWeatherPath
                    ? "Reset target set"
                    : "No reset path resolved"}
                </p>
                <p className="mt-1 text-sm leading-6 text-slate-300 break-all">
                  {resolvedWeatherPath ??
                    "Choose a source and enter a path to set the reset target explicitly."}
                </p>
              </article>
            </div>
          </section>

          <section className="space-y-4">
            <div className="panel-frame overflow-visible rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Path intake</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Resolve the selected weather path.
                  </h3>
                </div>
                <span className="max-w-[300px] overflow-auto modern-scrollbar rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  <span title={resolvedWeatherPath ?? "Not set"}>
                    {resolvedWeatherPath ?? "Not set"}
                  </span>
                </span>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_220px]">
                <SearchablePathInput
                  label="Weather path"
                  value={WeatherPath}
                  options={backendCsvPathOptions}
                  placeholder="Type or choose a backend CSV path"
                  disabled={weatherSourceMode === "default"}
                  onChange={(nextValue) => {
                    setWeatherSourceMode(
                      nextValue.trim().length > 0 ? "path" : weatherSourceMode,
                    );
                    setWeatherPath(nextValue);
                  }}
                />

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Path status
                  </span>
                  <div className="rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100">
                    {sourceModeLabel(weatherSourceMode, "weather")}
                  </div>
                </label>
              </div>

              <div className="mt-4 flex justify-between flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void handleReset()}
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset with selected source
                </button>

                {weatherSourceMode === "upload" ? (
                  <div className="flex items-center justify-between gap-2">
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(event) =>
                        setSelectedUploadFile(event.target.files?.[0] ?? null)
                      }
                      className="block max-w-full text-xs text-slate-300 file:mr-3 file:rounded-full file:border-0 file:bg-white/[0.06] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-white/[0.1]"
                    />
                    <button
                      type="button"
                      disabled={isUploadingWeather || !selectedUploadFile}
                      onClick={() => void handleUploadWeather()}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-xs font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isUploadingWeather ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <FileUp className="h-4 w-4" />
                      )}
                      {isUploadingWeather ? "Uploading..." : "Upload CSV"}
                    </button>
                  </div>
                ) : null}

                {uploadedWeather ? (
                  <p className="text-[10px] flex justify-end text-end text-emerald-100">
                    Uploaded:{" "}
                    <span className="font-mono">
                      {uploadedWeather.resolved_path}
                    </span>
                  </p>
                ) : null}

                {uploadedWeather ? (
                  <div className="flex items-center justify-end gap-2">
                    <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                      Upload ready
                    </span>
                    <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                      Make sure to{" "}
                      <span
                        onClick={() => {
                          setActiveTab("inspect");
                          setCSVPathAfterUpload("weather");
                        }}
                        className=" underline cursor-pointer hover:text-emerald-200"
                      >
                        Analyze/Profile
                      </span>{" "}
                      the CSV first to validate the schema.
                    </p>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Weather state</p>
              <div className="mt-3 flex flex-col gap-3">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Weather profile
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {(csvProfile?.requested_role === "weather" ||
                      csvProfile?.inferred_role === "weather") &&
                    csvProfile?.can_use_now
                      ? "Runtime ready"
                      : "Profile first then derive"}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {csvProfile?.requested_role === "weather" ||
                    csvProfile?.inferred_role === "weather"
                      ? csvProfile?.usage_recommendation
                      : "Profile a CSV to verify the schema before it becomes the reset source."}
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Derived weather
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {derivedWeather?.output_file_path
                      ? "Derived Successfully"
                      : "Not derived yet"}
                  </p>
                  {derivedWeatherMappingSummary ? (
                    <p className="mt-2 text-xs flex text-slate-400 whitespace-pre-wrap">
                      Mapped:{" "}
                      <span className="text-slate-100">
                        {derivedWeatherMappingSummary.split(" · ").join("\n")}
                      </span>
                    </p>
                  ) : null}
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {derivedWeather?.usage_recommendation ??
                      "Use the inspect tab to map source columns and derive a runtime-ready CSV."}
                  </p>
                </article>
              </div>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "household" ? (
        <div className="relative z-50 grid gap-6 px-6 py-6 lg:px-8 xl:grid-cols-[1.02fr_0.98fr]">
          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Household source</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Choose household signals (optional).
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  {sourceModeLabel(householdSourceMode, "household")}
                </span>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {(["default", "path", "upload"] as DataSourceMode[]).map(
                  (id) => {
                    const active = householdSourceMode === id;
                    const title =
                      id === "default"
                        ? "Synthetic baseline"
                        : id === "path"
                          ? "Existing CSV path"
                          : "Upload local CSV";

                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setHouseholdSourceMode(id)}
                        className={`rounded-[1.15rem] border px-4 py-4 text-left transition ${
                          active
                            ? "border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.08)]"
                            : "border-white/10 bg-[rgba(255,255,255,0.03)] hover:border-white/20 hover:bg-[rgba(255,255,255,0.05)]"
                        }`}
                      >
                        <p className="text-sm font-semibold text-white">
                          {title}
                        </p>
                      </button>
                    );
                  },
                )}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Source map</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Bundled default
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Good for quick resets and demo execution without extra file
                    handling.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Existing path
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Use already uploaded or derived backend paths when they
                    exist.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Upload local CSV
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Send a local file to the backend, then reset against the
                    resolved path.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Reset path
                  </p>
                  <p className="mt-2 text-xs break-all leading-6 text-slate-300">
                    {resolvedWeatherPath ??
                      "Choose a mode and enter a path to make the reset target explicit."}
                  </p>
                </article>
              </div>
            </div>
          </section>

          <section className="space-y-4">
            <div className="panel-frame overflow-visible rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Path intake</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Resolve the household CSV path.
                  </h3>
                </div>
                <span className="max-w-[300px] overflow-auto modern-scrollbar rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  <span title={resolvedHouseholdPath ?? "Not set"}>
                    {resolvedHouseholdPath ?? "Not set"}
                  </span>
                </span>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_220px]">
                <SearchablePathInput
                  label="Household path"
                  value={householdPath}
                  options={backendCsvPathOptions}
                  placeholder="Type or choose a backend CSV path"
                  disabled={householdSourceMode === "default"}
                  onChange={(nextValue) => {
                    setHouseholdPath(nextValue);
                    setHouseholdSourceMode(
                      nextValue.trim().length > 0
                        ? "path"
                        : householdSourceMode,
                    );
                  }}
                />

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Path status
                  </span>
                  <div className="rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100">
                    {sourceModeLabel(householdSourceMode, "household")}
                  </div>
                </label>
              </div>

              <div className="mt-4 flex justify-between flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void handleReset()}
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset with selected sources
                </button>

                {householdSourceMode === "upload" ? (
                  <div className="flex items-center justify-between gap-2">
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(event) =>
                        setSelectedHouseholdUploadFile(
                          event.target.files?.[0] ?? null,
                        )
                      }
                      className="block max-w-full text-xs text-slate-300 file:mr-3 file:rounded-full file:border-0 file:bg-white/[0.06] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-white/[0.1]"
                    />
                    <button
                      type="button"
                      disabled={
                        isUploadingHousehold || !selectedHouseholdUploadFile
                      }
                      onClick={() => void handleUploadHousehold()}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-xs font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isUploadingHousehold ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <FileUp className="h-4 w-4" />
                      )}
                      {isUploadingHousehold ? "Uploading..." : "Upload CSV"}
                    </button>
                  </div>
                ) : null}
                {/* </div> */}

                {uploadedHousehold ? (
                  <p className="mt-3 text-[10px] text-emerald-100 break-all">
                    Uploaded:{" "}
                    <span className="font-mono">
                      {uploadedHousehold.resolved_path}
                    </span>
                  </p>
                ) : null}

                {uploadedHousehold ? (
                  <div className="flex items-center justify-end gap-2">
                    <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                      Upload ready
                    </span>
                    <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                      Make sure to{" "}
                      <span
                        onClick={() => {
                          setActiveTab("inspect");
                          setCSVPathAfterUpload("household");
                        }}
                        className=" underline cursor-pointer hover:text-emerald-200"
                      >
                        Analyze/Profile
                      </span>{" "}
                      the CSV first to validate the schema.
                    </p>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Household State</p>
              <div className="mt-3 flex flex-col gap-3">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Household profile
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {(csvProfile?.requested_role === "household" ||
                      csvProfile?.inferred_role === "household") &&
                    csvProfile?.can_use_now
                      ? "Runtime ready"
                      : "Profile first then derive"}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {csvProfile?.requested_role === "household"
                      ? csvProfile?.usage_recommendation
                      : "Profile a CSV to verify the schema before it becomes the reset source."}
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Derived Household
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {derivedHousehold?.output_file_path
                      ? "Derived Successfully"
                      : "Not derived yet"}
                  </p>
                  {derivedHouseholdMappingSummary ? (
                    <p className="mt-2 text-xs flex text-slate-400 whitespace-pre-wrap">
                      Mapped:{" "}
                      <span className="text-slate-100">
                        {derivedHouseholdMappingSummary.split(" · ").join("\n")}
                      </span>
                    </p>
                  ) : null}
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {derivedHousehold?.usage_recommendation ??
                      "Use the inspect tab to map source columns and derive a runtime-ready CSV."}
                  </p>
                </article>
              </div>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "market" ? (
        <div className="relative z-50 grid gap-6 px-6 py-6 lg:px-8 xl:grid-cols-[1.02fr_0.98fr]">
          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Market source</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Choose market signals (optional).
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  {sourceModeLabel(marketSourceMode, "market")}
                </span>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {(["default", "path", "upload"] as DataSourceMode[]).map(
                  (id) => {
                    const active = marketSourceMode === id;
                    const title =
                      id === "default"
                        ? "Synthetic baseline"
                        : id === "path"
                          ? "Existing CSV path"
                          : "Upload local CSV";

                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setMarketSourceMode(id)}
                        className={`rounded-[1.15rem] border px-4 py-4 text-left transition ${
                          active
                            ? "border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.08)]"
                            : "border-white/10 bg-[rgba(255,255,255,0.03)] hover:border-white/20 hover:bg-[rgba(255,255,255,0.05)]"
                        }`}
                      >
                        <p className="text-sm font-semibold text-white">
                          {title}
                        </p>
                      </button>
                    );
                  },
                )}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Source map</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Bundled default
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Good for quick resets and demo execution without extra file
                    handling.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Existing path
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Use already uploaded or derived backend paths when they
                    exist.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Upload local CSV
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Send a local file to the backend, then reset against the
                    resolved path.
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Reset path
                  </p>
                  <p className="mt-2 text-xs break-all leading-6 text-slate-300">
                    Choose a mode and enter a path to make the reset target
                    explicit.
                  </p>
                </article>
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Resolved Reset path</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2"></div>
              <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                <p className="mt-2 text-sm font-semibold text-white">
                  {resolvedMarketPath
                    ? "Reset target set"
                    : "No reset path resolved"}
                </p>
                <p className="mt-1 text-sm leading-6 text-slate-300 break-all">
                  {resolvedMarketPath ??
                    "Choose a source and enter a path to set the reset target explicitly."}
                </p>
              </article>
            </div>
          </section>

          <section className="space-y-4">
            <div className="panel-frame overflow-visible rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Path intake</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Resolve the market CSV path.
                  </h3>
                </div>
                <span className="max-w-[300px] overflow-auto modern-scrollbar rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  <span title={resolvedMarketPath ?? "Not set"}>
                    {resolvedMarketPath ?? "Not set"}
                  </span>
                </span>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_220px]">
                <SearchablePathInput
                  label="Market path"
                  value={marketPath}
                  options={backendCsvPathOptions}
                  placeholder="Type or choose a backend CSV path"
                  disabled={marketSourceMode === "default"}
                  onChange={(nextValue) => {
                    setMarketPath(nextValue);
                    setMarketSourceMode(
                      nextValue.trim().length > 0 ? "path" : marketSourceMode,
                    );
                  }}
                />

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Path status
                  </span>
                  <div className="rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100">
                    {sourceModeLabel(marketSourceMode, "market")}
                  </div>
                </label>
              </div>

              <div className="mt-4 flex justify-between flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void handleReset()}
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <RotateCcw className="h-4 w-4" />
                  Reset with selected sources
                </button>

                {marketSourceMode === "upload" ? (
                  <div className="flex items-center justify-between gap-2">
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(event) =>
                        setSelectedMarketUploadFile(
                          event.target.files?.[0] ?? null,
                        )
                      }
                      className="block max-w-full text-xs text-slate-300 file:mr-3 file:rounded-full file:border-0 file:bg-white/[0.06] file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-white/[0.1]"
                    />
                    <button
                      type="button"
                      disabled={isUploadingMarket || !selectedMarketUploadFile}
                      onClick={() => void handleUploadMarket()}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-xs font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isUploadingMarket ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <FileUp className="h-4 w-4" />
                      )}
                      {isUploadingMarket ? "Uploading..." : "Upload CSV"}
                    </button>
                  </div>
                ) : null}

                {uploadedMarket ? (
                  <p className="mt-3 text-[10px] text-emerald-100 break-all">
                    Uploaded:{" "}
                    <span className="font-mono">
                      {uploadedMarket.resolved_path}
                    </span>
                  </p>
                ) : null}

                {uploadedMarket ? (
                  <div className="flex items-center justify-end gap-2">
                    <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                      Upload ready
                    </span>
                    <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                      Make sure to{" "}
                      <span
                        onClick={() => {
                          setActiveTab("inspect");
                          setCSVPathAfterUpload("market");
                        }}
                        className=" underline cursor-pointer hover:text-emerald-200"
                      >
                        Analyze/Profile
                      </span>{" "}
                      the CSV first to validate the schema.
                    </p>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="panel-frame rounded-[1.35rem] p-5">
              <p className="section-eyebrow">Market State</p>
              <div className="mt-3 flex flex-col gap-3">
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Market profile
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {(csvProfile?.requested_role === "market" ||
                      csvProfile?.inferred_role === "market") &&
                    csvProfile?.can_use_now
                      ? "Runtime ready"
                      : "Profile first then derive"}
                  </p>
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {csvProfile?.requested_role === "market"
                      ? csvProfile?.usage_recommendation
                      : "Profile a CSV to verify the schema before it becomes the reset source."}
                  </p>
                </article>
                <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                    Derived Market
                  </p>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {derivedMarket?.output_file_path
                      ? "Derived Successfully"
                      : "Not derived yet"}
                  </p>
                  {derivedMarketMappingSummary ? (
                    <p className="mt-2 text-xs flex text-slate-400 whitespace-pre-wrap">
                      Mapped:{" "}
                      <span className="text-slate-100">
                        {derivedMarketMappingSummary.split(" · ").join("\n")}
                      </span>
                    </p>
                  ) : null}
                  <p className="mt-1 text-sm leading-6 text-slate-300">
                    {derivedMarket?.usage_recommendation ??
                      "Use the inspect tab to map source columns and derive a runtime-ready CSV."}
                  </p>
                </article>
              </div>
            </div>
          </section>
        </div>
      ) : null}

      {activeTab === "inspect" ? (
        <div className="relative z-50 grid gap-6 px-6 py-6 lg:px-8 xl:grid-cols-[0.95fr_1.05fr]">
          <section className="space-y-4">
            <div className="panel-frame rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">CSV intake analyzer</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    Profile the file before it enters the grid
                  </h3>
                </div>
                <button
                  type="button"
                  disabled={isProfilingCsv || csvPath.trim().length === 0}
                  onClick={() => void handleAnalyzeCsv()}
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-xs  font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isProfilingCsv ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCcw className="h-4 w-4" />
                  )}
                  Analyze CSV
                </button>
              </div>

              <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_220px]">
                <SearchablePathInput
                  label="CSV path"
                  value={csvPath}
                  options={backendCsvPathOptions}
                  placeholder="Type or choose a backend CSV path"
                  onChange={(nextValue) => setCsvPath(nextValue)}
                />

                <label className="grid gap-2 text-sm text-slate-200">
                  <span className="section-eyebrow text-[10px]">
                    Role check
                  </span>
                  <select
                    value={csvRole}
                    onChange={(event) =>
                      setCsvRole(event.target.value as CsvRole)
                    }
                    className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(127,182,168,0.45)]"
                  >
                    <option value="auto">Auto detect</option>
                    <option value="weather">Weather</option>
                    <option value="household">Household</option>
                    <option value="market">Market</option>
                  </select>
                </label>
              </div>

              {csvError ? (
                <p className="mt-4 rounded-[1rem] border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                  {csvError}
                </p>
              ) : null}
            </div>

            {isProfilingCsv || !csvProfile ? (
              <div className="panel-frame rounded-[1.35rem] p-5 text-sm text-slate-300">
                <p className="section-eyebrow text-[10px] mb-2">
                  No profile yet
                </p>
                Profile a CSV to populate the analyzer summary and column
                mapping helpers.
              </div>
            ) : (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <p className="section-eyebrow">Derived mapping</p>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  {/** Role-aware mapping: show selectors populated from profile columns **/}
                  <div className="col-span-2 text-sm text-slate-300">
                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-300">
                        Role
                      </span>
                      <span className="font-semibold text-white">
                        {csvRole === "auto"
                          ? (csvProfile.inferred_role ?? "auto")
                          : csvRole}
                      </span>
                      <span className="ml-2 text-xs text-slate-400">
                        Auto-detected from columns; change to override.
                      </span>
                    </div>
                  </div>

                  {csvProfile.columns && csvProfile.columns.length > 0 ? (
                    <>
                      {effectiveRole === "weather" ? (
                        <>
                          {renderSelectWithWarning(
                            "Solar",
                            solarColumnInput,
                            setSolarColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "Wind",
                            windColumnInput,
                            setWindColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "Timestamp",
                            timestampColumnInput,
                            setTimestampColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "Temperature",
                            temperatureColumnInput,
                            setTemperatureColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "Humidity",
                            humidityColumnInput,
                            setHumidityColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "GHI",
                            ghiColumnInput,
                            setGhiColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "DNI",
                            dniColumnInput,
                            setDniColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "DHI",
                            dhiColumnInput,
                            setDhiColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "Irradiance",
                            irradianceColumnInput,
                            setIrradianceColumnInput,
                          )}
                          {renderSelectWithWarning(
                            "PV Power",
                            pvPowerColumnInput,
                            setPvPowerColumnInput,
                          )}

                          <div className="col-span-2 mt-3 rounded-[1rem] border border-white/10 bg-white/[0.02] p-4.">
                            <div
                              onClick={onClickPvParameters}
                              className="flex items-center justify-between gap-3 cursor-pointer p-4"
                            >
                              <div>
                                <p className="section-eyebrow text-[10px]">
                                  PV model parameters
                                </p>
                                <p className="mt-1 text-xs text-slate-400">
                                  Leave blank/unchanged to use backend config
                                  defaults. Click to{" "}
                                  <span className="underline">
                                    {" "}
                                    {openPvParameterModal ? "hide" : "edit"}
                                  </span>
                                </p>
                              </div>
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-slate-300">
                                Optional
                              </span>
                            </div>

                            {openPvParameterModal && (
                              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3 border-t border-white/10 p-4">
                                {[
                                  [
                                    "Panel tilt (deg)",
                                    panelTiltInput,
                                    setPanelTiltInput,
                                    "30",
                                  ],
                                  [
                                    "Panel azimuth (deg)",
                                    panelAzimuthInput,
                                    setPanelAzimuthInput,
                                    "180",
                                  ],
                                  [
                                    "Panel area (m²)",
                                    panelAreaInput,
                                    setPanelAreaInput,
                                    "1.0",
                                  ],
                                  [
                                    "Panel efficiency",
                                    panelEfficiencyInput,
                                    setPanelEfficiencyInput,
                                    "0.15",
                                  ],
                                  [
                                    "Temp coefficient (/K)",
                                    tempCoefficientInput,
                                    setTempCoefficientInput,
                                    "-0.004",
                                  ],
                                ].map(([label, value, setter, placeholder]) => (
                                  <label
                                    key={String(label)}
                                    className="grid gap-2 text-sm text-slate-200"
                                  >
                                    <span className="section-eyebrow text-[10px]">
                                      {String(label)}
                                    </span>
                                    <input
                                      type="text"
                                      inputMode="decimal"
                                      value={value as string}
                                      onChange={(event) =>
                                        (setter as (value: string) => void)(
                                          event.target.value,
                                        )
                                      }
                                      placeholder={String(placeholder)}
                                      className="flex h-[56px] w-full items-center rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-4 py-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-[rgba(212,175,55,0.45)]"
                                    />
                                  </label>
                                ))}
                              </div>
                            )}
                          </div>
                        </>
                      ) : null}

                      {effectiveRole === "household" ? (
                        <>
                          {[
                            [
                              "Timestamp",
                              timestampColumnInput,
                              setTimestampColumnInput,
                            ],
                            [
                              "Consumption",
                              consumptionColumnInput,
                              setConsumptionColumnInput,
                            ],
                            [
                              "Household ID",
                              householdIdColumnInput,
                              setHouseholdIdColumnInput,
                            ],
                            [
                              "PV Generation",
                              pvGenerationColumnInput,
                              setPvGenerationColumnInput,
                            ],
                            [
                              "Net Load",
                              netLoadColumnInput,
                              setNetLoadColumnInput,
                            ],
                            ["Meter", meterColumnInput, setMeterColumnInput],
                            ["Demand", demandColumnInput, setDemandColumnInput],
                          ].map(([label, value, setter]) => (
                            <label
                              key={String(label)}
                              className="grid gap-2 text-sm text-slate-200"
                            >
                              <SearchableColumnSelect
                                label={String(label)}
                                value={value as string}
                                options={csvProfile.columns}
                                placeholder="-- optional --"
                                onChange={(nextValue) =>
                                  (setter as (value: string) => void)(nextValue)
                                }
                              />
                            </label>
                          ))}
                        </>
                      ) : null}

                      {effectiveRole === "market" ? (
                        <>
                          {[
                            [
                              "Timestamp",
                              timestampColumnInput,
                              setTimestampColumnInput,
                            ],
                            ["Supply", supplyColumnInput, setSupplyColumnInput],
                            [
                              "Demand",
                              marketDemandColumnInput,
                              setMarketDemandColumnInput,
                            ],
                            ["Price", priceColumnInput, setPriceColumnInput],
                            ["Ask", askColumnInput, setAskColumnInput],
                            ["Bid", bidColumnInput, setBidColumnInput],
                            [
                              "Quantity",
                              quantityColumnInput,
                              setQuantityColumnInput,
                            ],
                            [
                              "Clearing Price",
                              clearingPriceColumnInput,
                              setClearingPriceColumnInput,
                            ],
                          ].map(([label, value, setter]) => (
                            <label
                              key={String(label)}
                              className="grid gap-2 text-sm text-slate-200"
                            >
                              <SearchableColumnSelect
                                label={String(label)}
                                value={value as string}
                                options={csvProfile.columns}
                                placeholder="-- optional --"
                                onChange={(nextValue) =>
                                  (setter as (value: string) => void)(nextValue)
                                }
                              />
                            </label>
                          ))}
                        </>
                      ) : null}

                      {effectiveRole === "auto" ? (
                        <div className="col-span-2 panel-frame rounded-[0.75rem] p-3 text-sm text-slate-300">
                          Unable to auto-detect a role confidently. Choose a
                          role above to edit mappings.
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-3">
                  {warningDetails.length ? (
                    <label className="flex items-center gap-3 rounded-full border border-amber-400/20 bg-amber-400/10 px-4 py-2 text-xs text-amber-100">
                      <input
                        type="checkbox"
                        checked={confirmUnitMismatch}
                        onChange={(event) =>
                          setConfirmUnitMismatch(event.target.checked)
                        }
                        className="mb-0.5 h.-4 w.-4 rounded border-amber-300 bg-transparent text-amber-400 focus:ring-amber-300"
                      />
                      <div className="flex flex-col">
                        <span className="leading-3">
                          I reviewed the warnings & want to derive anyway.
                        </span>
                        {!confirmUnitMismatch ? (
                          <span className="text-[8px] text-amber-300">
                            *Derive is blocked until you confirm the review
                            checkbox.
                          </span>
                        ) : (
                          <span className="text-[8px] text-green-300">
                            Derive is available now.
                          </span>
                        )}
                      </div>
                    </label>
                  ) : null}
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-3">
                  {effectiveRole === "weather" ? (
                    <button
                      type="button"
                      disabled={isDerivingWeather || !canDeriveWeather}
                      onClick={() => void handleDeriveWeather()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isDerivingWeather ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4" />
                      )}
                      {isDerivingWeather ? "Deriving..." : "Derive weather CSV"}
                    </button>
                  ) : null}
                  {effectiveRole === "household" ? (
                    <button
                      type="button"
                      disabled={
                        isDerivingHousehold || !consumptionColumnInput.trim()
                      }
                      onClick={() => void handleDeriveHousehold()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isDerivingHousehold ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4" />
                      )}
                      {isDerivingHousehold
                        ? "Deriving..."
                        : "Derive household CSV"}
                    </button>
                  ) : null}

                  {effectiveRole === "market" ? (
                    <button
                      type="button"
                      disabled={isDerivingMarket || !priceColumnInput.trim()}
                      onClick={() => void handleDeriveMarket()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-xs font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {isDerivingMarket ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Database className="h-4 w-4" />
                      )}
                      {isDerivingMarket ? "Deriving..." : "Derive market CSV"}
                    </button>
                  ) : null}

                  {derivedWeather?.output_file_path ? (
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleResetWithDerivedWeather()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-xs font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Sparkles className="h-4 w-4" />
                      Reset with derived CSV
                    </button>
                  ) : null}

                  {derivedHousehold?.output_file_path ? (
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleResetWithDerivedHousehold()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-xs font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Sparkles className="h-4 w-4" />
                      Reset with derived CSV
                    </button>
                  ) : null}

                  {derivedMarket?.output_file_path ? (
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleResetWithDerivedMarket()}
                      className="inline-flex items-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-xs font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Sparkles className="h-4 w-4" />
                      Reset with derived CSV
                    </button>
                  ) : null}

                  {csvProfile?.can_use_now ? (
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleResetWithProfile()}
                      className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-xs font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset with profiled CSV
                    </button>
                  ) : null}
                </div>
              </div>
            )}
          </section>

          <section className="space-y-4">
            {csvProfile ? (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="section-eyebrow">Profile result</p>
                    <h3 className="mt-2 text-lg font-semibold text-white">
                      Analyzer Readout
                    </h3>
                  </div>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                    {csvRole}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <article className="panel-frame rounded-[1rem] px-4 py-3">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Rows
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {csvProfile.rows}
                    </p>
                  </article>
                  <article className="panel-frame rounded-[1rem] px-4 py-3">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Columns
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {csvProfile.column_count}
                    </p>
                  </article>
                  <article className="panel-frame rounded-[1rem] px-4 py-3">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Compatibility
                    </p>
                    <p
                      className={`mt-2 text-2xl font-semibold ${csvProfile.can_use_now ? "text-emerald-400" : "text-rose-400"}`}
                    >
                      {csvProfile.can_use_now ? "Ready" : "Needs work"}
                    </p>
                  </article>
                </div>

                {csvProfile?.time_profile ? (
                  <div className="mt-4 rounded-[1rem] border border-white/10 bg-white/[0.03] px-4 py-4 text-sm text-slate-200">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Time profile
                    </p>
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      <div className="text-amber-100">
                        Column:{" "}
                        <span className="font-mono text-slate-100">
                          {String(
                            csvProfile.time_profile.timestamp_column ?? "—",
                          )}
                        </span>
                      </div>
                      <div className="text-amber-100">
                        Parse_OK:{" "}
                        <span className="font-mono text-slate-100">
                          {Math.round(
                            Number(csvProfile.time_profile.parse_ok_rate ?? 0) *
                              100,
                          )}
                          %
                        </span>
                      </div>
                      <div className="text-amber-100">
                        Step (median):{" "}
                        <span className="font-mono text-slate-100">
                          {csvProfile.time_profile.median_step_seconds ?? "—"} s
                        </span>
                      </div>
                      <div className="text-amber-100">
                        Rows analyzed:{" "}
                        <span className="font-mono text-slate-100">
                          {csvProfile.time_profile.rows_analyzed ?? "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : null}

                <div className="mt-4 flex flex-wrap gap-2">
                  {csvProfile.columns.slice(0, 10).map((columnName) => (
                    <span
                      key={columnName}
                      className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] text-slate-200"
                    >
                      {columnName}
                    </span>
                  ))}
                  <span
                    key={"more-columns"}
                    className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-[11px] text-slate-200"
                  >
                    {csvProfile.columns.length > 10
                      ? `+${csvProfile.columns.length - 10}`
                      : "…"}
                  </span>
                </div>

                <p className="mt-4 text-sm leading-6 text-amber-300">
                  Remarks:{" "}
                  <span className="text-amber-100">
                    {csvProfile.usage_recommendation}
                  </span>
                </p>

                {effectiveRole === "weather" && warningDetails.length > 0 ? (
                  <div className="mt-4 rounded-[1rem] max-h-[50rem] overflow-y-auto border border-amber-300/20 bg-amber-300/10 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="section-eyebrow text-[10px] text-amber-200">
                          Unit warnings
                        </p>
                        <p className="mt-1 text-sm text-amber-50">
                          Review the flagged columns before deriving.
                        </p>
                      </div>
                      <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-amber-100">
                        {warningDetails.length} flagged / {warningColumnCount}{" "}
                        columns
                      </span>
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => applySafeWeatherFixes()}
                        className="rounded-full border border-amber-300/30 bg-amber-300/10 px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-amber-50 transition hover:bg-amber-300/20"
                      >
                        Auto-fill safe mappings
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmUnitMismatch(true)}
                        className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-slate-200 transition hover:bg-white/[0.08]"
                      >
                        I reviewed these warnings
                      </button>
                      <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-slate-300">
                        Showing {visibleWarningDetails.length} of{" "}
                        {warningDetails.length}
                      </span>
                    </div>

                    <div className="mt-3 space-y-2">
                      {visibleWarningDetails.map((warning) => (
                        <div
                          key={`${warning.column}-${warning.kind}`}
                          className="rounded-2xl border border-amber-300/20 bg-black/10 px-3 py-3"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <p className="text-sm font-semibold text-white">
                              {warning.column}
                            </p>
                            <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-amber-100">
                              {(warning.kind ?? "legacy_warning").replace(
                                /_/g,
                                " ",
                              )}
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-amber-50">
                            {warning.message}
                          </p>
                          <p className="mt-1 text-xs text-amber-100/80">
                            Suggested fix: {warning.suggestion}
                          </p>
                        </div>
                      ))}
                      {warningDetails.length > visibleWarningDetails.length ? (
                        <p className="text-xs text-amber-100/70">
                          {warningDetails.length - visibleWarningDetails.length}{" "}
                          more warnings are collapsed to keep this review
                          usable.
                        </p>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            {csvSchemas ? (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <p className="section-eyebrow">Runtime schema hints</p>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Weather
                    </p>
                    <p className="mt-2 text-xs leading-6 text-slate-300">
                      {csvSchemas.weather.recommended.join(" · ")}
                    </p>
                  </article>
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Household
                    </p>
                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {csvSchemas.household.recommended.join(" · ")}
                    </p>
                  </article>
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Market
                    </p>
                    <p className="mt-2 text-sm leading-6 text-slate-300">
                      {csvSchemas.market.recommended.join(" · ")}
                    </p>
                  </article>
                </div>
              </div>
            ) : null}

            {effectiveRole === "weather" && derivedWeather ? (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="section-eyebrow">Derived output</p>
                    <h3 className="mt-2 text-lg font-semibold text-white">
                      Normalized weather ready for reset
                    </h3>
                  </div>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                    {derivedWeather.normalization.enabled
                      ? "Normalized"
                      : "Raw"}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Rows
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {derivedWeather.rows}
                    </p>
                  </article>
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Solar scale
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {(
                        derivedWeather.normalization.irradiance_scale ??
                        derivedWeather.normalization.solar_scale ??
                        1.0
                      ).toFixed(2)}
                    </p>
                  </article>
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Wind scale
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {derivedWeather.normalization.wind_scale.toFixed(2)}
                    </p>
                  </article>
                </div>
                <p className="text-xs break-all text-white mt-4">
                  Derived Output:{" "}
                  <span className="font-mono text-emerald-100">
                    {derivedWeather.output_file_path}
                  </span>
                </p>
                <div className="flex items-center gap-2 mt-4">
                  <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                    csv derived
                  </span>
                  <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                    use the derived output path to{" "}
                    <span
                      onClick={() => void handleResetWithDerivedWeather()}
                      className=" underline cursor-pointer hover:text-emerald-200"
                    >
                      reset/simulate
                    </span>{" "}
                    the sim episode.
                  </p>
                </div>
              </div>
            ) : null}

            {effectiveRole === "household" && derivedHousehold ? (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="section-eyebrow">Derived output</p>
                    <h3 className="mt-2 text-lg font-semibold text-white">
                      Normalized household ready for reset
                    </h3>
                  </div>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                    {derivedHousehold.normalization.enabled
                      ? "Normalized"
                      : "Raw"}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Rows
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {derivedHousehold.rows}
                    </p>
                  </article>
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Consumption scale
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {(
                        derivedHousehold.normalization.consumption_scale ?? 1.0
                      ).toFixed(2)}
                    </p>
                  </article>
                </div>
                <p className="text-xs break-all text-white mt-4">
                  Derived Output:{" "}
                  <span className="font-mono text-emerald-100">
                    {derivedHousehold.output_file_path}
                  </span>
                </p>
                <div className="flex items-center gap-2 mt-4">
                  <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                    csv derived
                  </span>
                  <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                    use the derived output path to{" "}
                    <span
                      onClick={() => void handleReset()}
                      className=" underline cursor-pointer hover:text-emerald-200"
                    >
                      reset/simulate
                    </span>{" "}
                    the sim episode.
                  </p>
                </div>
              </div>
            ) : null}

            {effectiveRole === "market" && derivedMarket ? (
              <div className="panel-frame rounded-[1.35rem] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="section-eyebrow">Derived output</p>
                    <h3 className="mt-2 text-lg font-semibold text-white">
                      Normalized market ready for reset
                    </h3>
                  </div>
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                    {derivedMarket.normalization.enabled ? "Normalized" : "Raw"}
                  </span>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <article className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4">
                    <p className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Rows
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-white">
                      {derivedMarket.rows}
                    </p>
                  </article>
                </div>
                <p className="text-xs break-all text-white mt-4">
                  Derived Output:{" "}
                  <span className="font-mono text-emerald-100">
                    {derivedMarket.output_file_path}
                  </span>
                </p>
                <div className="flex items-center gap-2 mt-4">
                  <span className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-emerald-300">
                    csv derived
                  </span>
                  <p className="rounded-full border border-emerald-100 bg-emerald-100/10 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-orange-300">
                    use the derived output path to{" "}
                    <span
                      onClick={() => void handleReset()}
                      className=" underline cursor-pointer hover:text-emerald-200"
                    >
                      reset/simulate
                    </span>{" "}
                    the sim episode.
                  </p>
                </div>
              </div>
            ) : null}
          </section>
        </div>
      ) : null}

      {activeTab === "automation" ? (
        <div className="relative z-50 px-6 py-6 lg:px-8">
          <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
            <section className="panel-frame h-full rounded-[1.35rem] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-eyebrow">Faculty demo</p>
                  <h3 className="mt-2 text-lg font-semibold text-white">
                    One button that stages the whole stack.
                  </h3>
                </div>
                <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-300">
                  {isRunningDemo ? "Busy" : "Idle"}
                </span>
              </div>

              <p className="section-copy mt-3 text-sm">
                Reset the simulation, run a short episode, train PPO, compare
                against the rule baseline, and refresh the artifacts view.
              </p>

              <div className="mt-4 flex-1 flex flex-col flex-wrap gap-3">
                <button
                  type="button"
                  disabled={isBusy && !isRunningDemo}
                  onClick={() => void onRunDemoSequence()}
                  className="inline-flex items-center justify-center gap-2 rounded-[1rem] border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isRunningDemo ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  {isRunningDemo ? "Stop demo" : "Run visual demo"}
                </button>

                <div className="mt-4 rounded-[1rem] border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-200">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] uppercase tracking-[0.18em] text-slate-400">
                      Demo phase:
                    </span>
                    <span className="font-semibold text-[10px] text-white">
                      {demoPhase ?? (isRunningDemo ? "Running" : "Idle")}
                    </span>
                  </div>
                  <div className="mt-2 h-2 w-full rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-[rgba(212,175,55,0.65)]"
                      style={{
                        width: `${Math.round((demoProgress ?? 0) * 100)}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="mt-auto pt-6 flex items-center justify-between gap-3">
                  <button
                    type="button"
                    disabled={!isRunningDemo}
                    onClick={() => void onViewDemoSequence()}
                    className="mt-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <ArrowBigDownDash className="h-4 w-4" />
                    View Demo Sequence
                  </button>
                  <button
                    type="button"
                    disabled={isBusy}
                    onClick={() => void onRefresh()}
                    className="mt-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <RefreshCcw className="h-4 w-4" />
                    Refresh artifacts
                  </button>
                </div>
              </div>
            </section>

            <section className="space-y-4">
              <article className="panel-frame rounded-[1.35rem] p-5">
                <p className="section-eyebrow">Sequence</p>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {[
                    [
                      "Reset",
                      "Hydrate the grid with the chosen weather source.",
                    ],
                    [
                      "Run",
                      "Execute a short live episode for visual feedback.",
                    ],
                    ["Train", "Kick off a small PPO run for proof of life."],
                    ["Compare", "Refresh PPO-vs-rule metrics and export them."],
                  ].map(([title, description]) => (
                    <article
                      key={title}
                      className="rounded-[1.1rem] border border-white/10 bg-white/[0.03] px-4 py-4"
                    >
                      <p className="text-[11px] uppercase tracking-[0.2em] text-[#f6e7be]">
                        {title}
                      </p>
                      <p className="mt-2 text-sm leading-6 text-slate-300">
                        {description}
                      </p>
                    </article>
                  ))}
                </div>
              </article>

              <article className="panel-frame rounded-[1.35rem] p-5">
                <p className="section-eyebrow">Status note</p>
                <p className="mt-3 text-sm leading-7 text-slate-300">
                  This tab is intentionally sparse: it preserves the one-click
                  demo path without forcing the rest of the control room to stay
                  open.
                </p>
              </article>
            </section>
          </div>
        </div>
      ) : null}
    </section>
  );
}
