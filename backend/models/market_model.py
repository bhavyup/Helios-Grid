"""
models/market_model.py
======================

Placeholder for the Helios-Grid market model.

This file previously contained a mislabeled duplicate of ``CommunicationLayer``
(which belongs in ``communication_layer.py``).  That duplicate has been removed.

``MarketModel`` is expected by other modules (e.g. ``house_agent.py``) via::

    from market_model import MarketModel

This stub satisfies that import without inventing unspecified market logic.

TODO
----
- Define the market-clearing mechanism (e.g. uniform-price auction, LMP, P2P).
- Define the pricing interface consumed by household agents.
- Decide whether this is a per-timestep pricer, a stateful auction, or a
  policy-driven component.
- Ensure deterministic behavior under fixed seed.
- Move out of ``models/`` if it becomes a service rather than a data model.
"""


class MarketModel:
    """Placeholder for the Helios-Grid energy market model.

    This class intentionally contains no logic.  All methods raise
    ``NotImplementedError`` to make accidental use visible immediately.

    When the market mechanism is designed, this stub should be replaced
    with the real implementation.
    """

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "MarketModel(stub=True)"

    def get_price(self, *args, **kwargs):
        """Return the current energy price.  Not yet implemented."""
        raise NotImplementedError(
            "MarketModel.get_price() is a stub. "
            "No market mechanism has been implemented yet."
        )

    def step(self, *args, **kwargs):
        """Advance the market by one timestep.  Not yet implemented."""
        raise NotImplementedError(
            "MarketModel.step() is a stub. "
            "No market mechanism has been implemented yet."
        )

    def reset(self, *args, **kwargs):
        """Reset the market state.  Not yet implemented."""
        raise NotImplementedError(
            "MarketModel.reset() is a stub. "
            "No market mechanism has been implemented yet."
        )