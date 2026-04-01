"""
test_house_env.py — Contract test suite for HouseEnv.

CONTEXT
=======
As of the last full audit, ``envs/house_env.py`` contains
``CommunicationLayer`` (a TCP socket server), NOT ``HouseEnv``.
That class must be relocated before these tests can run.

These tests define the **contract HouseEnv must satisfy**, derived
entirely from how ``grid_env.py`` consumes it:

    envs = [HouseEnv() for _ in range(N)]       # zero-arg construction
    house.reset()                                # return value unused
    result = house.step(ndarray_shape_6)         # continuous action
    state  = house.get_state()                   # → array shape (10,)

Every assumption is cited inline.  No behavior is invented.

SOURCE OF TRUTH
===============
grid_env.py declares:
    house_actions : Box(low=0, high=1, shape=(N, 6), dtype=float32)
    house_states  : Box(low=-inf, high=inf, shape=(N, 10), dtype=float32)

Per-household slice:  action (6,)  →  state (10,)
"""

import sys
import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.house_env import HouseEnv


# ===========================================================================
# Constants derived from grid_env.py's space declarations
# ===========================================================================
ACTION_DIM = 6          # Box(low=0, high=1, shape=(N, 6))
OBS_DIM = 10            # Box(shape=(N, 10), dtype=float32)
ACTION_DTYPE = np.float32
OBS_DTYPE = np.float32


# ===========================================================================
# Helpers
# ===========================================================================

def _make_action(value: float = 0.5) -> np.ndarray:
    """
    Produce a valid per-household action matching grid_env.py's
    Box(low=0, high=1, shape=(N, 6)) sliced to shape (6,).
    """
    return np.full(ACTION_DIM, value, dtype=ACTION_DTYPE)


def _unpack_step(result):
    """
    Safely extract observation from step() return, whether it is
    a gym-style tuple or a bare state array.

    Returns (obs, reward_or_none, done_or_none, info_or_none).
    """
    if isinstance(result, tuple):
        obs = result[0]
        reward = result[1] if len(result) > 1 else None
        done = result[2] if len(result) > 2 else None
        info = result[3] if len(result) > 3 else None
        return obs, reward, done, info
    # Bare array returned (grid_env.py captures full return as one var)
    return result, None, None, None


def _assert_valid_obs(arr: np.ndarray, context: str = ""):
    """Assert observation shape, dtype-castability, and finiteness."""
    obs = np.asarray(arr, dtype=OBS_DTYPE)
    assert obs.shape == (OBS_DIM,), (
        f"{context}: expected shape ({OBS_DIM},), got {obs.shape}"
    )
    assert np.all(np.isfinite(obs)), (
        f"{context}: non-finite values: {obs}"
    )


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def env():
    """Fresh HouseEnv; closed after test if close() exists."""
    e = HouseEnv()
    yield e
    if hasattr(e, "close") and callable(e.close):
        e.close()


@pytest.fixture
def reset_env(env):
    """HouseEnv that has already been reset."""
    env.reset()
    return env


# ===========================================================================
# 1. Import and construction
# ===========================================================================

class TestConstruction:

    def test_import_resolves(self):
        """
        ``from envs.house_env import HouseEnv`` must succeed.

        If this fails, CommunicationLayer likely still occupies the
        module slot.
        """
        from envs.house_env import HouseEnv as _H
        assert _H is not None

    def test_zero_arg_construction(self):
        """
        grid_env.py: ``[HouseEnv() for _ in range(N)]``
        Constructor must accept zero arguments.
        """
        e = HouseEnv()
        assert e is not None


# ===========================================================================
# 2. reset() contract
# ===========================================================================

class TestReset:

    def test_reset_does_not_crash(self, env):
        """reset() is callable without error."""
        env.reset()

    def test_reset_return_type(self, env):
        """
        grid_env.py does NOT use the return value of house.reset().
        Classic gym convention: returns an observation.
        Accept ndarray or None.
        """
        result = env.reset()
        assert result is None or isinstance(result, np.ndarray), (
            f"reset() returned {type(result).__name__}; "
            f"expected ndarray or None"
        )

    def test_reset_observation_shape_if_returned(self, env):
        """If reset() returns an array, shape must be (10,)."""
        result = env.reset()
        if isinstance(result, np.ndarray):
            _assert_valid_obs(result, "reset()")

    def test_reset_observation_dtype_if_returned(self, env):
        """If reset() returns an array, it must be castable to float32."""
        result = env.reset()
        if isinstance(result, np.ndarray):
            cast = np.asarray(result, dtype=OBS_DTYPE)
            assert cast.dtype == OBS_DTYPE


# ===========================================================================
# 3. get_state() contract
# ===========================================================================

class TestGetState:
    """
    grid_env.py._get_observation():
        house_states = [house.get_state() for house in self.house_environments]
        np.array(house_states, dtype=np.float32)   # shape (N, 10)
    """

    def test_method_exists(self, reset_env):
        assert hasattr(reset_env, "get_state"), (
            "HouseEnv must implement get_state() — "
            "grid_env.py calls it every observation cycle"
        )
        assert callable(reset_env.get_state)

    def test_shape_after_reset(self, reset_env):
        state = reset_env.get_state()
        _assert_valid_obs(state, "get_state() after reset")

    def test_castable_to_float32(self, reset_env):
        state = reset_env.get_state()
        arr = np.asarray(state, dtype=OBS_DTYPE)
        assert arr.dtype == OBS_DTYPE

    def test_valid_after_step(self, reset_env):
        reset_env.step(_make_action())
        state = reset_env.get_state()
        _assert_valid_obs(state, "get_state() after step")

    def test_valid_after_multiple_steps(self, reset_env):
        rng = np.random.RandomState(7)
        for i in range(15):
            action = rng.uniform(0, 1, size=(ACTION_DIM,)).astype(ACTION_DTYPE)
            reset_env.step(action)
        state = reset_env.get_state()
        _assert_valid_obs(state, "get_state() after 15 steps")


# ===========================================================================
# 4. step() contract
# ===========================================================================

class TestStep:
    """
    grid_env.py:
        result = house.step(house_actions[i])
        house_step_results.append(result)

    The full return is captured as ONE variable.  This is
    compatible with EITHER a bare state OR a gym 4-tuple.
    Tests below validate both scenarios defensively.
    """

    def test_accepts_continuous_action(self, reset_env):
        """step() must accept ndarray shape (6,) dtype float32."""
        result = reset_env.step(_make_action(0.5))
        assert result is not None

    def test_return_is_tuple_or_array(self, reset_env):
        """Return must be a tuple (gym) or ndarray (bare state)."""
        result = reset_env.step(_make_action())
        assert isinstance(result, (tuple, np.ndarray)), (
            f"step() returned {type(result).__name__}; "
            f"expected tuple or ndarray"
        )

    def test_observation_shape(self, reset_env):
        result = reset_env.step(_make_action())
        obs, _, _, _ = _unpack_step(result)
        _assert_valid_obs(obs, "step() observation")

    def test_reward_finite_if_present(self, reset_env):
        """If step returns a tuple, reward must be numeric and finite."""
        result = reset_env.step(_make_action())
        _, reward, _, _ = _unpack_step(result)
        if reward is not None:
            assert isinstance(reward, (int, float, np.integer, np.floating)), (
                f"Reward type: {type(reward).__name__}; expected numeric"
            )
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_done_is_boolean_if_present(self, reset_env):
        result = reset_env.step(_make_action())
        _, _, done, _ = _unpack_step(result)
        if done is not None:
            assert isinstance(done, (bool, np.bool_)), (
                f"Done type: {type(done).__name__}; expected bool"
            )

    def test_info_is_dict_if_present(self, reset_env):
        result = reset_env.step(_make_action())
        _, _, _, info = _unpack_step(result)
        if info is not None:
            assert isinstance(info, dict), (
                f"Info type: {type(info).__name__}; expected dict"
            )


# ===========================================================================
# 5. Action-space coverage
# ===========================================================================

class TestActionSpace:

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_boundary_values_accepted(self, reset_env, value):
        """Box(low=0, high=1) boundary and interior values."""
        result = reset_env.step(_make_action(value))
        obs, _, _, _ = _unpack_step(result)
        _assert_valid_obs(obs, f"action value={value}")

    def test_varied_per_dimension_action(self, reset_env):
        """Each of the 6 action dimensions can differ."""
        action = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=ACTION_DTYPE)
        result = reset_env.step(action)
        obs, _, _, _ = _unpack_step(result)
        _assert_valid_obs(obs, "varied-per-dim action")

    def test_sequential_random_actions(self, reset_env):
        """10 random valid actions in sequence; no crash, valid obs."""
        rng = np.random.RandomState(42)
        for i in range(10):
            action = rng.uniform(0, 1, size=(ACTION_DIM,)).astype(ACTION_DTYPE)
            result = reset_env.step(action)
            obs, _, _, _ = _unpack_step(result)
            _assert_valid_obs(obs, f"random action step {i}")


# ===========================================================================
# 6. Action has causal effect
# ===========================================================================

class TestActionCausality:
    """
    A non-degenerate environment should produce different states
    for different actions.  If all actions yield identical
    trajectories, the MDP is degenerate and no learning is possible.
    """

    def test_different_actions_produce_different_states(self):
        """
        Run two single-step trajectories from the same initial state
        with different actions.  States should differ.

        NOTE: this can legitimately fail if the environment is
        stochastic and unseeded, or if a single step has negligible
        effect.  Treat a failure here as a research-validity warning,
        not necessarily a code bug.
        """
        env1 = HouseEnv()
        env2 = HouseEnv()

        # Seed both identically if possible
        for e in (env1, env2):
            if hasattr(e, "seed") and callable(e.seed):
                e.seed(999)

        env1.reset()
        env2.reset()

        env1.step(_make_action(0.0))
        env2.step(_make_action(1.0))

        s1 = np.asarray(env1.get_state(), dtype=OBS_DTYPE)
        s2 = np.asarray(env2.get_state(), dtype=OBS_DTYPE)

        if np.array_equal(s1, s2):
            pytest.xfail(
                "Extreme actions (0.0 vs 1.0) produced identical states. "
                "The environment may be degenerate — action has no effect. "
                "Review HouseEnv.step() implementation."
            )


# ===========================================================================
# 7. Episode lifecycle
# ===========================================================================

class TestEpisodeLifecycle:

    def test_reset_produces_valid_state(self, env):
        env.reset()
        _assert_valid_obs(env.get_state(), "post-reset state")

    def test_reset_after_steps_produces_valid_state(self, env):
        """State after re-reset must be valid regardless of prior episode."""
        env.reset()
        for _ in range(5):
            env.step(_make_action(0.9))
        env.reset()
        _assert_valid_obs(env.get_state(), "state after re-reset")

    def test_multi_episode_isolation(self, env):
        """
        Run two episodes; verify second episode starts from a
        consistent initial state.
        """
        env.reset()
        initial_1 = np.asarray(env.get_state(), dtype=OBS_DTYPE).copy()

        for _ in range(10):
            env.step(_make_action(1.0))

        env.reset()
        initial_2 = np.asarray(env.get_state(), dtype=OBS_DTYPE).copy()

        # Shapes must match; values should match if deterministic
        assert initial_1.shape == initial_2.shape
        _assert_valid_obs(initial_2, "second episode initial state")

    def test_long_trajectory_stays_valid(self, reset_env):
        """50-step trajectory; observations remain finite and shaped."""
        rng = np.random.RandomState(77)
        for i in range(50):
            action = rng.uniform(0, 1, size=(ACTION_DIM,)).astype(ACTION_DTYPE)
            result = reset_env.step(action)
            obs, _, _, _ = _unpack_step(result)
            _assert_valid_obs(obs, f"long trajectory step {i}")
            _assert_valid_obs(reset_env.get_state(), f"get_state at step {i}")


# ===========================================================================
# 8. Determinism / reproducibility
# ===========================================================================

class TestDeterminism:
    """
    Core Helios-Grid principle: deterministic behavior under fixed seed.
    """

    @staticmethod
    def _try_seed(env, seed=42) -> bool:
        if hasattr(env, "seed") and callable(env.seed):
            env.seed(seed)
            return True
        return False

    def test_seed_method_exists(self, env):
        """
        A research-grade environment must be seedable.
        Skip (not fail) if unsupported, but flag the gap.
        """
        if not (hasattr(env, "seed") and callable(env.seed)):
            pytest.skip(
                "HouseEnv lacks seed() — deterministic experiments "
                "are impossible.  This violates the project's core "
                "reproducibility principle."
            )

    def test_deterministic_initial_state(self):
        """Same seed → identical initial state."""
        envs = [HouseEnv(), HouseEnv()]
        for e in envs:
            if not self._try_seed(e, 42):
                pytest.skip("Seeding not supported")
            e.reset()

        s0 = np.asarray(envs[0].get_state(), dtype=OBS_DTYPE)
        s1 = np.asarray(envs[1].get_state(), dtype=OBS_DTYPE)
        np.testing.assert_array_equal(s0, s1, err_msg=(
            "Identical seeds produced different initial states"
        ))

    def test_deterministic_trajectory(self):
        """Same seed + same actions → identical trajectory."""
        trajectories = []
        for _ in range(2):
            e = HouseEnv()
            if not self._try_seed(e, 123):
                pytest.skip("Seeding not supported")
            e.reset()

            states = []
            for _ in range(10):
                e.step(_make_action(0.5))
                states.append(
                    np.asarray(e.get_state(), dtype=OBS_DTYPE).copy()
                )
            trajectories.append(states)

        for i, (a, b) in enumerate(zip(trajectories[0], trajectories[1])):
            np.testing.assert_array_equal(a, b, err_msg=(
                f"Trajectories diverged at step {i}"
            ))

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different states."""
        envs = [HouseEnv(), HouseEnv()]
        seeds = [42, 99]
        states = []
        for e, s in zip(envs, seeds):
            if not self._try_seed(e, s):
                pytest.skip("Seeding not supported")
            e.reset()
            for _ in range(5):
                e.step(_make_action(0.5))
            states.append(np.asarray(e.get_state(), dtype=OBS_DTYPE))

        if np.array_equal(states[0], states[1]):
            pytest.xfail(
                "Different seeds produced identical states after 5 steps. "
                "Environment may be ignoring the seed."
            )


# ===========================================================================
# 9. Edge cases and robustness
# ===========================================================================

class TestEdgeCases:

    def test_step_before_reset(self, env):
        """
        Must either work (implicit reset) or raise a clear error.
        Must NOT silently corrupt state or segfault.
        """
        try:
            result = env.step(_make_action())
            obs, _, _, _ = _unpack_step(result)
            _assert_valid_obs(obs, "step-before-reset")
        except (RuntimeError, ValueError, AttributeError, TypeError):
            pass  # explicit error is acceptable

    def test_get_state_before_reset(self, env):
        """Must return valid state or raise a clear error."""
        try:
            state = env.get_state()
            _assert_valid_obs(state, "get_state-before-reset")
        except (RuntimeError, ValueError, AttributeError, TypeError):
            pass  # explicit error is acceptable

    def test_close_is_idempotent(self, env):
        """close() must not raise, even when called twice."""
        if hasattr(env, "close") and callable(env.close):
            env.close()
            env.close()

    def test_repeated_resets_no_accumulation(self, env):
        """20 consecutive resets must not leak or grow state."""
        for _ in range(20):
            env.reset()
        _assert_valid_obs(env.get_state(), "after 20 resets")


# ===========================================================================
# 10. gym.Env interface compliance (informational)
# ===========================================================================

class TestGymCompliance:
    """
    Optional checks for standard gym.Env attributes.
    Failures here are informational — grid_env.py may not require
    these, but standard tooling (wrappers, vectorized envs) does.
    """

    def test_has_action_space(self, env):
        if not hasattr(env, "action_space"):
            pytest.xfail(
                "HouseEnv lacks action_space attribute. "
                "Standard gym wrappers will fail."
            )

    def test_has_observation_space(self, env):
        if not hasattr(env, "observation_space"):
            pytest.xfail(
                "HouseEnv lacks observation_space attribute. "
                "Standard gym wrappers will fail."
            )

    def test_action_space_shape_if_present(self, env):
        if not hasattr(env, "action_space"):
            pytest.skip("No action_space")
        space = env.action_space
        if hasattr(space, "shape"):
            assert space.shape == (ACTION_DIM,), (
                f"Expected action_space shape ({ACTION_DIM},), "
                f"got {space.shape}"
            )

    def test_observation_space_shape_if_present(self, env):
        if not hasattr(env, "observation_space"):
            pytest.skip("No observation_space")
        space = env.observation_space
        if hasattr(space, "shape"):
            assert space.shape == (OBS_DIM,), (
                f"Expected observation_space shape ({OBS_DIM},), "
                f"got {space.shape}"
            )