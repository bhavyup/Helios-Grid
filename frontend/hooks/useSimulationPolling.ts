"use client";

import { useEffect, useState } from "react";

interface UseSimulationPollingOptions {
  intervalMs?: number;
  onRefresh: () => void | Promise<void>;
  enabledByDefault?: boolean;
  autoRefresh?: boolean;
  setAutoRefresh?: (value: boolean) => void;
}

interface UseSimulationPollingState {
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
}

export function useSimulationPolling(
  options: UseSimulationPollingOptions,
): UseSimulationPollingState {
  const {
    intervalMs = 2500,
    onRefresh,
    enabledByDefault = false,
    autoRefresh: controlledAutoRefresh,
    setAutoRefresh: setControlledAutoRefresh,
  } = options;
  const [autoRefreshState, setAutoRefreshState] = useState<boolean>(enabledByDefault);
  const autoRefresh = controlledAutoRefresh ?? autoRefreshState;
  const setAutoRefresh = setControlledAutoRefresh ?? setAutoRefreshState;

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }

    const interval = setInterval(() => {
      void onRefresh();
    }, intervalMs);

    return () => {
      clearInterval(interval);
    };
  }, [autoRefresh, intervalMs, onRefresh]);

  return {
    autoRefresh,
    setAutoRefresh,
  };
}
