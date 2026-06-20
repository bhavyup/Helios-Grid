"""Compatibility wrapper for MarketAgent.

MarketAgent — Placeholder market agent scaffold for future decentralized market policies.
"""

import logging

logger = logging.getLogger(__name__)


class MarketAgent:  # noqa: F811
    """Placeholder market agent scaffold for future decentralized market policies."""

    def __init__(self, *args, **kwargs) -> None:
        self._initialized_with = {"args": args, "kwargs": kwargs}

    def act(self, *args, **kwargs):
        raise NotImplementedError(
            "MarketAgent.act() is not implemented yet. "
            "Use HouseAgent/CoordinatorAgent scaffolds until market policy integration."
        )
