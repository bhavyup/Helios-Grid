"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  apiClient,
  DeriveWeatherInput,
  UploadWeatherInput,
  DeriveHouseholdInput,
  DeriveMarketInput,
  UploadHouseholdInput,
  UploadMarketInput,
} from "@/lib/api-client";
import {
  CsvProfilePayload,
  CsvRole,
  CsvPathsPayload,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  UploadedWeatherPayload,
  DerivedHouseholdPayload,
  DerivedMarketPayload,
  UploadedHouseholdPayload,
  UploadedMarketPayload,
} from "@/lib/types";

interface UseWeatherOptions {
  autoLoadSchemas?: boolean;
}

interface UseWeatherState {
  isProfilingCsv: boolean;
  isDerivingWeather: boolean;
  isUploadingWeather: boolean;
  csvError: string | null;
  csvSchemas: CsvSchemasPayload | null;
  csvPaths: CsvPathsPayload | null;
  csvProfile: CsvProfilePayload | null;
  derivedWeather: DerivedWeatherPayload | null;
  uploadedWeather: UploadedWeatherPayload | null;
  isDerivingHousehold: boolean;
  isUploadingHousehold: boolean;
  derivedHousehold: DerivedHouseholdPayload | null;
  uploadedHousehold: UploadedHouseholdPayload | null;
  isDerivingMarket: boolean;
  isUploadingMarket: boolean;
  derivedMarket: DerivedMarketPayload | null;
  uploadedMarket: UploadedMarketPayload | null;
  reloadSchemas: () => Promise<void>;
  reloadPaths: () => Promise<void>;
  analyzeCsv: (filePath: string, role: CsvRole) => Promise<void>;
  deriveWeatherFromCsv: (input: DeriveWeatherInput) => Promise<void>;
  deriveHouseholdFromCsv: (input: DeriveHouseholdInput) => Promise<void>;
  deriveMarketFromCsv: (input: DeriveMarketInput) => Promise<void>;
  uploadWeatherCsv: (input: UploadWeatherInput) => Promise<void>;
  uploadHouseholdCsv: (input: UploadHouseholdInput) => Promise<void>;
  uploadMarketCsv: (input: UploadMarketInput) => Promise<void>;
}

export function useWeather(options: UseWeatherOptions = {}): UseWeatherState {
  const { autoLoadSchemas = false } = options;
  const [isProfilingCsv, setIsProfilingCsv] = useState<boolean>(false);
  const [isDerivingWeather, setIsDerivingWeather] = useState<boolean>(false);
  const [isUploadingWeather, setIsUploadingWeather] = useState<boolean>(false);
  const [isDerivingHousehold, setIsDerivingHousehold] = useState<boolean>(false);
  const [isUploadingHousehold, setIsUploadingHousehold] = useState<boolean>(false);
  const [isDerivingMarket, setIsDerivingMarket] = useState<boolean>(false);
  const [isUploadingMarket, setIsUploadingMarket] = useState<boolean>(false);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvSchemas, setCsvSchemas] = useState<CsvSchemasPayload | null>(null);
  const [csvPaths, setCsvPaths] = useState<CsvPathsPayload | null>(null);
  const [csvProfile, setCsvProfile] = useState<CsvProfilePayload | null>(null);
  const [derivedWeather, setDerivedWeather] =
    useState<DerivedWeatherPayload | null>(null);
  const [uploadedWeather, setUploadedWeather] =
    useState<UploadedWeatherPayload | null>(null);
  const [derivedHousehold, setDerivedHousehold] =
    useState<DerivedHouseholdPayload | null>(null);
  const [uploadedHousehold, setUploadedHousehold] =
    useState<UploadedHouseholdPayload | null>(null);
    const [derivedMarket, setDerivedMarket] =
    useState<DerivedMarketPayload | null>(null);
  const [uploadedMarket, setUploadedMarket] =
    useState<UploadedMarketPayload | null>(null);

  const captureCsvError = useCallback((unknownError: unknown) => {
    if (unknownError instanceof ApiError) {
      setCsvError(`${unknownError.message}`);
      return;
    }

    if (unknownError instanceof Error) {
      setCsvError(unknownError.message);
      return;
    }

    setCsvError("An unknown CSV profiling error occurred.");
  }, []);

  const reloadSchemas = useCallback(async () => {
    try {
      const payload = await apiClient.getCsvSchemas();
      setCsvSchemas(payload);
      setCsvError(null);
    } catch (unknownError) {
      captureCsvError(unknownError);
    }
  }, [captureCsvError]);

  const reloadPaths = useCallback(async () => {
    try {
      const payload = await apiClient.getCsvPaths();
      setCsvPaths(payload);
      setCsvError(null);
    } catch (unknownError) {
      captureCsvError(unknownError);
    }
  }, [captureCsvError]);

  const analyzeCsv = useCallback(
    async (filePath: string, role: CsvRole) => {
      setIsProfilingCsv(true);
      try {
        const payload = await apiClient.profileCsvData({
          file_path: filePath,
          role,
          preview_rows: 5,
        });
        setCsvProfile(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsProfilingCsv(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const deriveWeatherFromCsv = useCallback(
    async (input: DeriveWeatherInput) => {
      setIsDerivingWeather(true);
      try {
        const payload = await apiClient.deriveWeatherCsv(input);
        setDerivedWeather(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsDerivingWeather(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const deriveHouseholdFromCsv = useCallback(
    async (input: DeriveHouseholdInput) => {
      setIsDerivingHousehold(true);
      try {
        const payload = await apiClient.deriveHouseholdCsv(input);
        setDerivedHousehold(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsDerivingHousehold(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const deriveMarketFromCsv = useCallback(
    async (input: DeriveMarketInput) => {
      setIsDerivingMarket(true);
      try {
        const payload = await apiClient.deriveMarketCsv(input);
        setDerivedMarket(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsDerivingMarket(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const uploadWeatherCsv = useCallback(
    async (input: UploadWeatherInput) => {
      setIsUploadingWeather(true);
      try {
        const payload = await apiClient.uploadWeatherCsv(input);
        setUploadedWeather(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsUploadingWeather(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const uploadHouseholdCsv = useCallback(
    async (input: UploadHouseholdInput) => {
      setIsUploadingHousehold(true);
      try {
        const payload = await apiClient.uploadHouseholdCsv(input);
        setUploadedHousehold(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsUploadingHousehold(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  const uploadMarketCsv = useCallback(
    async (input: UploadMarketInput) => {
      setIsUploadingMarket(true);
      try {
        const payload = await apiClient.uploadMarketCsv(input);
        setUploadedMarket(payload);
        void reloadPaths();
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsUploadingMarket(false);
      }
    },
    [captureCsvError, reloadPaths],
  );

  useEffect(() => {
    if (autoLoadSchemas) {
      void reloadSchemas();
      void reloadPaths();
    }
  }, [autoLoadSchemas, reloadPaths, reloadSchemas]);

  return {
    isProfilingCsv,
    isDerivingWeather,
    isUploadingWeather,
    isDerivingHousehold,
    isUploadingHousehold,
    isDerivingMarket,
    isUploadingMarket,
    csvError,
    csvSchemas,
    csvPaths,
    csvProfile,
    derivedWeather,
    uploadedWeather,
    derivedHousehold,
    uploadedHousehold,
    derivedMarket,
    uploadedMarket,
    reloadSchemas,
    reloadPaths,
    analyzeCsv,
    deriveWeatherFromCsv,
    uploadWeatherCsv,
    deriveHouseholdFromCsv,
    uploadHouseholdCsv,
    deriveMarketFromCsv,
    uploadMarketCsv
  };
}
