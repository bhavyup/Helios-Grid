"""
Graph construction and query utilities for Helios-Grid.

NODE-ID CONTRACT
================
All node IDs are **contiguous integers** starting at 0.  This is
required because ``GNNCoordinator`` converts edge lists to
``torch.Tensor(dtype=torch.long)`` for message-passing layers.

Node metadata (type, label) is stored as node attributes::

    graph.nodes[0]  ->  {"type": "household", "label": "House_0"}

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
from typing import Any, Dict, List, Tuple
from app.core.project_config import config

import math
import networkx as nx
import random

logger = logging.getLogger(__name__)


def _apply_neighborhood_layout(graph: nx.Graph) -> nx.Graph:
    """Attach stable local coordinates so the dashboard can render a real map-like view.

    Procedural layout:
    - households placed on a grid sized to household count
    - solar placed near the first K households (roof-adjacent spots)
    - wind placed on the perimeter
    """
    # ---- collect nodes deterministically ---------------------------------
    grid_nodes: list[int] = []
    household_nodes: list[int] = []
    solar_nodes: list[int] = []
    wind_nodes: list[int] = []

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "unknown")
        if node_type == "grid":
            grid_nodes.append(node_id)
        elif node_type == "household":
            household_nodes.append(node_id)
        elif node_type == "solar":
            solar_nodes.append(node_id)
        elif node_type == "wind":
            wind_nodes.append(node_id)

    grid_nodes.sort()
    household_nodes.sort()
    solar_nodes.sort()
    wind_nodes.sort()

    # ---- layout sizing ---------------------------------------------------
    n_house = len(household_nodes)
    if n_house == 0:
        graph.graph.update({"layout": "neighborhood-grid"})
        return graph

    # Grid dimensions (near-square)
    cols = int(math.ceil(math.sqrt(n_house)))
    rows = int(math.ceil(n_house / cols))

    # Canvas spacing in "map units" (tune visually)
    dx = 110.0
    dy = 90.0

    # Neighborhood extents (house grid only)
    grid_w = (cols - 1) * dx
    grid_h = (rows - 1) * dy

    # Padding around the lots
    pad_x = 220.0
    pad_y = 180.0

    bounds_width = grid_w + 2 * pad_x
    bounds_height = grid_h + 2 * pad_y

    # Ensure a minimum canvas so small neighborhoods don't look cramped
    canvas_width = max(1200.0, bounds_width + 240.0)
    canvas_height = max(760.0, bounds_height + 200.0)

    center_x = canvas_width / 2.0
    center_y = canvas_height / 2.0

    min_x = center_x - bounds_width / 2.0
    max_x = center_x + bounds_width / 2.0
    min_y = center_y - bounds_height / 2.0
    max_y = center_y + bounds_height / 2.0

    # Where the household grid starts
    lots_origin_x = center_x - grid_w / 2.0
    lots_origin_y = center_y - grid_h / 2.0

    # ---- graph-level metadata -------------------------------------------
    graph.graph.update(
        {
            "layout": "neighborhood-grid",
            "canvas_width": float(canvas_width),
            "canvas_height": float(canvas_height),
            "bounds": {
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
                "width": float(bounds_width),
                "height": float(bounds_height),
            },
            "lot_grid": {
                "rows": int(rows),
                "cols": int(cols),
                "dx": float(dx),
                "dy": float(dy),
                "origin_x": float(lots_origin_x),
                "origin_y": float(lots_origin_y),
            },
        }
    )

    # ---- grid hub --------------------------------------------------------
    hub_id = grid_nodes[0] if grid_nodes else 0
    hub_attrs = graph.nodes[hub_id]
    hub_attrs.update(
        {
            "x": float(center_x),
            "y": float(center_y),
            "kind": "utility_hub",
            "asset_class": "grid",
        }
    )

    # ---- households: place on grid --------------------------------------
    for lot_index, node_id in enumerate(household_nodes):
        r = lot_index // cols
        c = lot_index % cols
        x = lots_origin_x + c * dx
        y = lots_origin_y + r * dy

        graph.nodes[node_id].update(
            {
                "x": float(x),
                "y": float(y),
                "kind": "house",
                "asset_class": "building",
                "lot_index": int(lot_index),
                "row": int(r),
                "col": int(c),
                "has_solar_candidate": bool((lot_index % 3) == 0),
            }
        )

        seed = int(config.get("seed", 42))
        rng = random.Random(seed + n_house * 1009 + len(solar_nodes) * 917)
        shuffled_households = list(household_nodes)
        rng.shuffle(shuffled_households)

    # ---- solar: attach near first households (roof-adjacent spots) -------
    for idx, node_id in enumerate(solar_nodes):
        served_house = shuffled_households[idx % len(shuffled_households)]
        hx = float(graph.nodes[served_house].get("x", center_x))
        hy = float(graph.nodes[served_house].get("y", center_y))
        sx = hx
        sy = hy - 42.0

        graph.nodes[node_id].update(
            {
                "x": float(sx),
                "y": float(sy),
                "kind": "roof_solar",
                "asset_class": "solar",
                "serves_household_id": int(served_house),
            }
        )

    # ---- wind: perimeter placement --------------------------------------
    perimeter_spots = [
        (min_x + 70.0, min_y + 70.0),
        (center_x, min_y + 52.0),
        (max_x - 70.0, min_y + 70.0),
        (min_x + 70.0, max_y - 70.0),
        (max_x - 70.0, max_y - 70.0),
        (min_x + 52.0, center_y),
        (max_x - 52.0, center_y),
    ]
    for idx, node_id in enumerate(wind_nodes):
        wx, wy = perimeter_spots[idx % len(perimeter_spots)]
        graph.nodes[node_id].update(
            {
                "x": float(wx),
                "y": float(wy),
                "kind": "wind_turbine",
                "asset_class": "wind",
            }
        )

    return graph


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
    graph = nx.Graph()
    node_id = 0

    # --- central hub -------------------------------------------------
    grid_hub_id = node_id
    graph.add_node(grid_hub_id, type="grid", label="Grid")
    node_id += 1

    # --- households --------------------------------------------------
    for i in range(num_households):
        graph.add_node(node_id, type="household", label=f"House_{i}")
        graph.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    # --- solar panels ------------------------------------------------
    for i in range(num_solar_panels):
        graph.add_node(node_id, type="solar", label=f"Solar_{i}")
        graph.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    # --- wind turbines -----------------------------------------------
    for i in range(num_wind_turbines):
        graph.add_node(node_id, type="wind", label=f"Wind_{i}")
        graph.add_edge(node_id, grid_hub_id, weight=1.0)
        node_id += 1

    logger.debug(
        "Created grid graph: %d nodes (%d households, %d solar, "
        "%d wind, 1 hub), %d edges.",
        graph.number_of_nodes(),
        num_households,
        num_solar_panels,
        num_wind_turbines,
        graph.number_of_edges(),
    )

    return _apply_neighborhood_layout(graph)


def build_grid_graph(
    topology_file: str,
    num_households: int,
) -> nx.Graph:
    """
    Build a grid graph, optionally from a topology file.

    This function exists for compatibility with ``grid_env.py``::

        from app.infrastructure.graph_utils import build_grid_graph
        self.graph = build_grid_graph(grid_topology_file, num_households)

    ASSUMPTION: ``grid_env.py`` passes a file path and a household
    count.  Currently the file path is **ignored** and a default
    star topology is generated.  When topology-file loading is
    implemented, this function should parse the file and construct
    the graph accordingly.

    Args:
        topology_file: Path to grid topology JSON (currently unused).
        num_households: Number of household nodes.

    Returns:
        ``nx.Graph`` with integer node IDs.
    """
    if topology_file:
        logger.info(
            "build_grid_graph called with topology_file=%r -- using generated neighborhood coordinates with %d households.",
            topology_file,
            num_households,
        )

    num_solar_panels = int(config.get("num_solar_panels", 24))  # choose your default

    num_wind_turbines = int(config.get("num_wind_turbines", 4))

    return create_grid_graph(
        num_households=num_households,
        num_solar_panels=num_solar_panels,
        num_wind_turbines=num_wind_turbines,
    )


# ===================================================================
# Node queries
# ===================================================================


def get_node_types(graph: nx.Graph) -> Dict[int, str]:
    """
    Return a mapping of node ID -> type string.

    Nodes without a ``type`` attribute are labeled ``"unknown"``.
    """
    return {node: data.get("type", "unknown") for node, data in graph.nodes(data=True)}


def get_node_labels(graph: nx.Graph) -> Dict[int, str]:
    """
    Return a mapping of node ID -> human-readable label.
    """
    return {node: data.get("label", str(node)) for node, data in graph.nodes(data=True)}


def get_nodes_by_type(graph: nx.Graph, node_type: str) -> List[int]:
    """
    Return a list of node IDs that have the given type.
    """
    return [
        node for node, data in graph.nodes(data=True) if data.get("type") == node_type
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
    return {node: dict(data) for node, data in graph.nodes(data=True)}


# ===================================================================
# Edge queries
# ===================================================================


def get_edges(graph: nx.Graph) -> List[Tuple[int, int, float]]:
    """
    Return all edges as ``(src, dst, weight)`` triples.

    Edges without a ``weight`` attribute default to ``1.0``.
    """
    return [(u, v, data.get("weight", 1.0)) for u, v, data in graph.edges(data=True)]


def get_edge_attributes(
    graph: nx.Graph,
    u: int,
    v: int,
) -> Dict[str, Any]:
    """Return all attributes of a single edge."""
    return dict(graph.edges[u, v])


def get_all_edge_attributes(
    graph: nx.Graph,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Return attributes for every edge."""
    return {(u, v): dict(data) for u, v, data in graph.edges(data=True)}
