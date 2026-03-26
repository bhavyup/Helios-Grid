
import random
import logging
from datetime import datetime
from typing import Any, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from utils.graph_utils import create_grid_graph, get_node_types, get_edges
from utils.logging_utils import log_training_data, log_simulation_data
from config import config

logger = logging.getLogger(__name__)


class GNNCoordinator:
    """
    Manages a Graph Neural Network (GNN) for coordination tasks within the grid.

    This class serves as a placeholder for a more sophisticated GNN.
    It currently implements a simple MLP per node and does not use true
    graph convolution or attention.

    It is compatible with `grid_env.py`'s expected interface for `GNNCoordinator`.
    """

    def __init__(
        self,
        graph: nx.Graph,
        seed: int = 42,
        log_dir: str = config.LOG_DIR,
    ):
        """
        Initializes the GNN Coordinator.

        Args:
            graph: The NetworkX graph representing the grid topology.
            seed: Random seed for reproducibility.
            log_dir: Directory for logging training and simulation data.
        """
        self.graph = graph
        self.log_dir = log_dir
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed_everything(self.seed)

        # Map NetworkX node IDs to contiguous 0-indexed integers for PyTorch tensors
        self.node_to_idx = {node: i for i, node in enumerate(self.graph.nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}

        self.model = self._build_model().to(self.device)
        # ASSUMPTION: config.GNN_LR exists and is a float.
        # grid_env.py uses config['key'] (dict access), here config.KEY is used.
        # This inconsistency needs to be resolved in config.py.
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.GNN_LR)
        self.criterion = nn.MSELoss()

    def seed_everything(self, seed: int) -> None:
        """Sets seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("GNNCoordinator seeded with %d", seed)

    def _build_model(self) -> nn.Module:
        """
        Builds the GNN model.

        NOTE: This is currently a placeholder MLP that processes node features
        independently. It does NOT implement graph convolution or attention.
        To enable true GNN capabilities, a library like PyTorch Geometric
        would need to be installed and its GNN layers used here.
        e.g., `from torch_geometric.nn import GATConv`
        """

        class PlaceholderGNN(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                super().__init__()
                # This is a simple MLP acting on each node's features independently.
                # It does not use the `edge_index` for message passing.
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                """
                Forward pass for the placeholder GNN.

                Args:
                    x: Node features tensor (num_nodes, input_dim).
                    edge_index: Edge index tensor (2, num_edges).
                                NOTE: Currently unused in this placeholder model.

                Returns:
                    Output tensor (num_nodes, output_dim).
                """
                # For a true GNN, layers would consume `edge_index`
                # e.g., `x = self.conv1(x, edge_index)`
                return self.layers(x)

        # Current node features are type IDs, which are floats (1.0, 2.0, 3.0),
        # so input_dim=1. Output target is also the type ID, so output_dim=1.
        input_dim = 1
        output_dim = 1
        hidden_dim = 64 # Arbitrary hidden dimension for the placeholder MLP

        return PlaceholderGNN(input_dim, hidden_dim, output_dim)

    def _get_node_features(self) -> torch.Tensor:
        """
        Extracts node features from the graph.

        Currently, features are numerical representations of node types.
        """
        node_types = get_node_types(self.graph)
        # Ensure mapping to contiguous 0-indexed IDs is used
        sorted_nodes = [self.idx_to_node[i] for i in range(len(self.graph.nodes))]

        node_type_ids = torch.tensor(
            [
                1.0
                if node_types[node] == "household"
                else 2.0
                if node_types[node] == "solar"
                else 3.0
                for node in sorted_nodes
            ],
            dtype=torch.float32,
        ).unsqueeze(1).to(self.device)
        return node_type_ids

    # NOTE: _get_edge_features is currently unused by _build_model
    def _get_edge_features(self) -> torch.Tensor:
        """
        Extracts edge features from the graph.

        ASSUMPTION: `get_edges()` returns a list of 3-tuples (u, v, weight).
        If not, `e[2]` will raise an error.
        This method is currently not used by the placeholder GNN model.
        """
        edges = get_edges(self.graph)
        edge_weights = torch.tensor([e[2] for e in edges], dtype=torch.float32).unsqueeze(1)
        return edge_weights.to(self.device)

    def _get_graph_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares node features and edge index for the GNN model.
        """
        x = self._get_node_features()

        # Build bidirectional edge index, mapping NetworkX node IDs to 0-indexed
        mapped_edges = []
        for u, v in self.graph.edges:
            mapped_edges.append((self.node_to_idx[u], self.node_to_idx[v]))

        edge_src = [u_idx for u_idx, v_idx in mapped_edges]
        edge_dst = [v_idx for u_idx, v_idx in mapped_edges]

        # Ensure edges are bidirectional for proper message passing (even if placeholder doesn't use it)
        bidirectional_src = edge_src + edge_dst
        bidirectional_dst = edge_dst + edge_src

        edge_index = torch.tensor(
            [bidirectional_src, bidirectional_dst], dtype=torch.long
        ).to(self.device)

        return x, edge_index

    def reset(self) -> None:
        """
        Resets the GNN Coordinator's internal state (e.g., model parameters).
        Placeholder for now.
        """
        # For an agent, this might reset optimizer state or model weights.
        # For now, it does nothing as model is static after init.
        logger.debug("GNNCoordinator reset called.")
        pass

    def compute_coordination_signals(
        self, house_states: List[Any], graph: nx.Graph, weather_data: Any
    ) -> Any:
        """
        Computes coordination signals for households based on current state and weather.

        This method is the primary interface for `grid_env.py` to get signals.
        Currently, it runs a forward pass of the model and returns its output.

        Args:
            house_states: List of states from household environments.
                          (Specific structure unverified)
            graph: The current grid graph (likely redundant as it's self.graph).
            weather_data: Current weather data.

        Returns:
            Any: Placeholder for actual coordination signals.
        """
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            x, edge_index = self._get_graph_data()
            # NOTE: If house_states, graph, weather_data were to be inputs for the GNN,
            # they would need to be incorporated into x or edge_attr, etc.
            # Currently, only node type features are used.
            output = self.model(x, edge_index)
        
        # Placeholder for actual coordination signal interpretation
        # e.g., output could be node-wise energy recommendations, price signals, etc.
        logger.debug("Computed coordination signals (placeholder output mean: %.4f)", output.mean().item())
        return output.cpu().numpy() # Return numpy array for compatibility

    def train(self, num_epochs: int = 100) -> None:
        """
        Trains the GNN model.

        NOTE: The current training target is a placeholder (reconstructing input features).
        This will not lead to meaningful coordination. A proper target,
        e.g., predicted load, optimized control values, etc., is needed.
        """
        x, edge_index = self._get_graph_data()
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(x, edge_index)
            loss = self.criterion(output, x)  # Placeholder: model learns to reconstruct its input

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    "GNN Train - Epoch %d/%d, Loss: %.4f",
                    epoch + 1, num_epochs, loss.item()
                )

            # NOTE: Logged training data metrics are currently placeholders
            # (all are set to the loss value). This needs to be replaced
            # with actual relevant metrics for research validity.
            log_training_data(
                log_dir=self.log_dir,
                episode=epoch + 1, # Using epoch as episode for logging
                total_reward=loss.item(),
                avg_house_reward=loss.item(),
                avg_market_reward=loss.item(),
                avg_grid_reward=loss.item(),
                step=epoch + 1,
            )

    def simulate(self) -> None:
        """
        Runs a forward pass of the GNN model and logs simulation data.
        """
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            x, edge_index = self._get_graph_data()
            output = self.model(x, edge_index)

        # NOTE: Logged simulation data metrics are currently placeholders
        # (all are set to the mean of the GNN output). This needs to be replaced
        # with actual relevant metrics for research validity.
        # The timestamp fallback also needs to be deterministic if not provided.
        log_simulation_data(
            log_dir=self.log_dir,
            timestamp="2023-01-01T00:00:00" if self.seed is not None else datetime.now().isoformat(), # Deterministic placeholder
            grid_balance=output.mean().item(),
            market_balance=output.mean().item(),
            household_consumption=output.mean().item(),
            solar_production=output.mean().item(),
            wind_production=output.mean().item(),
        )

    def run(self, num_epochs: int = 100) -> None:
        """
        Orchestrates the training and simulation sequence for the GNN.
        This method is used by `communication.py` for its training workflow.
        """
        self.train(num_epochs)
        self.simulate()