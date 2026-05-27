"""TopologyEngine handles grid topology loading and graph construction."""

from app.infrastructure.graph_utils import build_grid_graph


class TopologyEngine:
    """Build and expose the grid topology graph and node lists."""

    def __init__(self, grid_topology_file: str, num_households: int) -> None:
        self.grid_topology_file = grid_topology_file
        self.num_households = num_households
        self.graph = build_grid_graph(grid_topology_file, num_households)
        self.nodes = list(self.graph.nodes)
        self.edges = list(self.graph.edges)
