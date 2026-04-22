"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError, apiClient, PPOComparisonInput, PPOTrainingInput } from "@/lib/api-client";
import { PolicyComparison, RewardCurvePayload, TrainingRunPayload } from "@/lib/types";

interface UseTrainingState {
  latestRun: TrainingRunPayload | null;
  latestComparison: (PolicyComparison & { run_id?: string; created_at?: string }) | null;
  rewardCurve: RewardCurvePayload | null;
  isTraining: boolean;
  error: string | null;
  runTraining: (input: PPOTrainingInput) => Promise<void>;
  runComparison: (input: PPOComparisonInput) => Promise<void>;
  refreshArtifacts: () => Promise<void>;
}

export function useTraining(): UseTrainingState {
  const [latestRun, setLatestRun] = useState<TrainingRunPayload | null>(null);
  const [latestComparison, setLatestComparison] = useState<
    (PolicyComparison & { run_id?: string; created_at?: string }) | null
  >(null);
  const [rewardCurve, setRewardCurve] = useState<RewardCurvePayload | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const captureError = useCallback((unknownError: unknown) => {
    if (unknownError instanceof ApiError) {
      setError(unknownError.message);
      return;
    }

    if (unknownError instanceof Error) {
      setError(unknownError.message);
      return;
    }

    setError("An unknown training error occurred.");
  }, []);

  const refreshArtifacts = useCallback(async () => {
    try {
      const [runPayload, comparisonPayload, curvePayload] = await Promise.all([
        apiClient.getLatestPpoTraining(),
        apiClient.getLatestPpoComparison(),
        apiClient.getLatestRewardCurve(),
      ]);

      if ("run_id" in runPayload) {
        setLatestRun(runPayload as TrainingRunPayload);
      }

      if ("ppo" in comparisonPayload) {
        setLatestComparison(comparisonPayload as PolicyComparison & { run_id?: string; created_at?: string });
      }

      if (curvePayload.reward_curve.length > 0) {
        setRewardCurve(curvePayload);
      }

      setError(null);
    } catch (unknownError) {
      captureError(unknownError);
    }
  }, [captureError]);

  const runTraining = useCallback(
    async (input: PPOTrainingInput) => {
      setIsTraining(true);
      try {
        const payload = await apiClient.runPpoTraining(input);
        setLatestRun(payload);
        setLatestComparison(payload.comparison);
        setRewardCurve({
          run_id: payload.run_id,
          created_at: payload.created_at,
          episodes: payload.training.episodes,
          reward_curve: payload.training.reward_curve,
        });
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      } finally {
        setIsTraining(false);
      }
    },
    [captureError],
  );

  const runComparison = useCallback(
    async (input: PPOComparisonInput) => {
      try {
        const payload = await apiClient.comparePpoAndRule(input);
        setLatestComparison(payload);
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      }
    },
    [captureError],
  );

  useEffect(() => {
    void refreshArtifacts();
  }, [refreshArtifacts]);

  return {
    latestRun,
    latestComparison,
    rewardCurve,
    isTraining,
    error,
    runTraining,
    runComparison,
    refreshArtifacts,
  };
}
