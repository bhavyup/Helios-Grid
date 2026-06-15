"use client";

import { useCallback, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
  const queryClient = useQueryClient();

  const latestRunQuery = useQuery({
    queryKey: ["training", "latest"],
    queryFn: () => apiClient.getLatestPpoTraining(),
  });

  const latestComparisonQuery = useQuery({
    queryKey: ["training", "comparison"],
    queryFn: () => apiClient.getLatestPpoComparison(),
  });

  const rewardCurveQuery = useQuery({
    queryKey: ["training", "reward-curve"],
    queryFn: () => apiClient.getLatestRewardCurve(),
  });

  const latestRun = useMemo<TrainingRunPayload | null>(() => {
    const payload = latestRunQuery.data;
    if (payload && "run_id" in payload) {
      return payload as TrainingRunPayload;
    }
    return null;
  }, [latestRunQuery.data]);

  const latestComparison = useMemo<
    (PolicyComparison & { run_id?: string; created_at?: string }) | null
  >(() => {
    const payload = latestComparisonQuery.data;
    if (payload && "ppo" in payload) {
      return payload as PolicyComparison & { run_id?: string; created_at?: string };
    }
    return null;
  }, [latestComparisonQuery.data]);

  const rewardCurve = useMemo<RewardCurvePayload | null>(() => {
    const payload = rewardCurveQuery.data;
    if (payload && payload.reward_curve.length > 0) {
      return payload;
    }
    return null;
  }, [rewardCurveQuery.data]);

  const runTrainingMutation = useMutation({
    mutationFn: (input: PPOTrainingInput) =>
      apiClient.runPpoTraining({
        ...input,
        wait_for_result: input.wait_for_result ?? true,
      }),
    onSuccess: (payload) => {
      if ("run_id" in payload && "training" in payload) {
        const runPayload = payload as TrainingRunPayload;
        queryClient.setQueryData(["training", "latest"], runPayload);
        queryClient.setQueryData(["training", "comparison"], runPayload.comparison);
        queryClient.setQueryData(["training", "reward-curve"], {
          run_id: runPayload.run_id,
          created_at: runPayload.created_at,
          episodes: runPayload.training.episodes,
          reward_curve: runPayload.training.reward_curve,
        });
      }
    },
  });

  const runComparisonMutation = useMutation({
    mutationFn: (input: PPOComparisonInput) => apiClient.comparePpoAndRule(input),
    onSuccess: (payload) => {
      queryClient.setQueryData(["training", "comparison"], payload);
    },
  });

  const runTraining = useCallback(
    async (input: PPOTrainingInput) => {
      await runTrainingMutation.mutateAsync(input);
    },
    [runTrainingMutation],
  );

  const runComparison = useCallback(
    async (input: PPOComparisonInput) => {
      await runComparisonMutation.mutateAsync(input);
    },
    [runComparisonMutation],
  );

  const refreshArtifacts = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey: ["training"] });
  }, [queryClient]);

  const error = useMemo(() => {
    const candidates = [
      runTrainingMutation.error,
      runComparisonMutation.error,
      latestRunQuery.error,
      latestComparisonQuery.error,
      rewardCurveQuery.error,
    ];

    for (const candidate of candidates) {
      if (!candidate) {
        continue;
      }
      if (candidate instanceof ApiError) {
        return candidate.message;
      }
      if (candidate instanceof Error) {
        return candidate.message;
      }
      return "An unknown training error occurred.";
    }

    return null;
  }, [
    latestComparisonQuery.error,
    latestRunQuery.error,
    rewardCurveQuery.error,
    runComparisonMutation.error,
    runTrainingMutation.error,
  ]);

  return {
    latestRun,
    latestComparison,
    rewardCurve,
    isTraining: runTrainingMutation.isPending,
    error,
    runTraining,
    runComparison,
    refreshArtifacts,
  };
}
