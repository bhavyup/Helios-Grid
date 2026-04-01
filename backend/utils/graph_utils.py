"""
Graph construction and query utilities for Helios-Grid.

NODE-ID CONTRACT
================
All node IDs are **contiguous integers** starting at 0.  This is
required because ``GNNCoordinator`` converts edge lists to
``torch.Tensor(dtype=torch.long)`` for message-passing layers.

Node metadata (type, label) is stored as node attributes::

    graph.nodes[0]  →  {"type": "household", "label": "House_0"}

GRAPH STRUCTURE
===============
``create_grid_graph`` builds a star topology:

* One central hub node (type ``"grid"``, label ``"Grid"``)
* ``num_households`` household nodes connected to the hub
* ``num_solar_panels`` solar nodes connected to the hub
* ``num_wind_turbines`` wind nodes connected to the hub

All edges have ``weight=1.0`` by default.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ===================================================================
# Graph construction
# ===================================================================

def create_grid_graph(
    num_households: int = 10,
    num_solar_panels: int = 5,
    num_wind_turbines: int = 3,
) -> nx.Graph:
    """
    Build a star-topology grid graph with integer node IDs.

    Args:
        num_households: Number of household nodes.
        num_solar_panels: Number of solar panel nodes.
        num_wind_turbines: Number of wind turbine nodes.

    Returns:
        ``nx.Graph`` with integer node IDs ``[0, N)`` where
        ``N = 1 + num_households + num_solar_panels + num_wind_turbines``.
        Node 0 is the central grid hub.
    """
    G = nx.Graph()
    node_id = 0

    # --- central hub -------------------------------------------------
    grid_hub_id = node_id
    G.add_node(grid_hub_id, type="grid", label="Grid")
    node_id += 1

    # --- households --------------------------------------------------
    for i in range(num_households):
        G.add_node(node_id, type="household", label=f"House_{i}")
        G.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    # --- solar panels ------------------------------------------------
    for i in range(num_solar_panels):
        G.add_node(node_id, type="solar", label=f"Solar_{i}")
        G.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    # --- wind turbines -----------------------------------------------
    for i in range(num_wind_turbines):
        G.add_node(node_id, type="wind", label=f"Wind_{i}")
        G.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    logger.debug(
        "Created grid graph: %d nodes (%d households, %d solar, "
        "%d wind, 1 hub), %d edges.",
        G.number_of_nodes(),
        num_households,
        num_solar_panels,
        num_wind_turbines,
        G.number_of_edges(),
    )

    return G


def build_grid_graph(
    topology_file: str,
    num_households: int,
) -> nx.Graph:
    """
    Build a grid graph, optionally from a topology file.

    This function exists for compatibility with ``grid_env.py``::

        from utils.graph_utils import build_grid_graph
        self.graph = build_grid_graph(grid_topology_file, num_households)

    Args:
        topology_file: Path to grid topology JSON. If provided and non-empty,
            raises NotImplementedError as file-based loading is not yet implemented.
        num_households: Number of household nodes.

    Returns:
        ``nx.Graph`` with integer node IDs.

    Raises:
        NotImplementedError: If topology_file is provided and non-empty.
    """
    if topology_file:
        raise NotImplementedError(
            f"File-based topology loading is not yet implemented. "
            f"Provided topology_file: {topology_file!r}. "
            f"Use create_grid_graph() directly or pass an empty string."
        )

    return create_grid_graph(num_households=num_households)


# ===================================================================
# Node queries
# ===================================================================

def get_node_types(graph: nx.Graph) -> Dict[int, str]:
    """
    Return a mapping of node ID → type string.

    Nodes without a ``type`` attribute are labeled ``"unknown"``.
    """
    return {
        node: data.get("type", "unknown")
        for node, data in graph.nodes(data=True)
    }


def get_node_labels(graph: nx.Graph) -> Dict[int, str]:
    """
    Return a mapping of node ID → human-readable label.
    """
    return {
        node: data.get("label", str(node))
        for node, data in graph.nodes(data=True)
    }


def get_nodes_by_type(graph: nx.Graph, node_type: str) -> List[int]:
    """
    Return a list of node IDs that have the given type.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get("type") == node_type
    ]


def get_subgraph_by_type(graph: nx.Graph, node_type: str) -> nx.Graph:
    """
    Return the subgraph induced by all nodes of the given type.
    """
    nodes = get_nodes_by_type(graph, node_type)
    return graph.subgraph(nodes)


def get_neighbors(graph: nx.Graph, node: int) -> List[int]:
    """Return the neighbor node IDs of the given node."""
    return list(graph.neighbors(node))


def get_node_attributes(graph: nx.Graph, node: int) -> Dict[str, Any]:
    """Return all attributes of a single node."""
    return dict(graph.nodes[node])


def get_all_node_attributes(
    graph: nx.Graph,
) -> Dict[int, Dict[str, Any]]:
    """Return attributes for every node."""
    return {
        node: dict(data)
        for node, data in graph.nodes(data=True)
    }


# ===================================================================
# Edge queries
# ===================================================================

def get_edges(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    """
    Return all edges as ``(src, dst, weight)`` triples.

    Edges without a ``weight`` attribute default to ``1.0``.
    """
    return [
        (u, v, data.get("weight", 1.0))
        for u, v, data in graph.edges(data=True)
    ]


def get_edge_attributes(
    graph: nx.Graph, u: int, v: int,
) -> Dict[str, Any]:
    """Return all attributes of a single edge."""
    return dict(graph.edges[u, v])


def get_all_edge_attributes(
    graph: nx.Graph,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Return attributes for every edge."""
    return {
        (u, v): dict(data)
        for u, v, data in graph.edges(data=True)
    }