"""
CommunicationLayer -- TCP socket-based message router.
"""

import socket
import threading
import queue
import logging
from typing import Dict, Any
from datetime import datetime

from app.core.project_config import config
from app.domain.models.gnn_coordinator import GNNCoordinator
from app.infrastructure.graph_utils import create_grid_graph
from app.infrastructure.logging_utils import log_simulation_data, log_training_data

logger = logging.getLogger(__name__)


class CommunicationLayer:
    """
    TCP socket server with threaded connection acceptance and a
    queue-based message router.
    """

    _ACCEPT_TIMEOUT_S: float = 1.0
    _QUEUE_TIMEOUT_S: float = 0.5

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        gnn_coordinator: "GNNCoordinator | None" = None,
    ):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(self._ACCEPT_TIMEOUT_S)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

        self.clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
        self.message_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.running: bool = False

        self._gnn_coordinator = gnn_coordinator

        self._listener_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            logger.warning(
                "CommunicationLayer.start() called while already running"
            )
            return

        self.running = True

        self._listener_thread = threading.Thread(
            target=self._listen_for_connections,
            daemon=True,
            name="comm-listener",
        )
        self._processor_thread = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name="comm-processor",
        )
        self._listener_thread.start()
        self._processor_thread.start()
        logger.info(
            "CommunicationLayer started on %s:%d", self.host, self.port
        )

    def stop(self) -> None:
        self.running = False

        if self._listener_thread is not None:
            self._listener_thread.join(timeout=self._ACCEPT_TIMEOUT_S + 1.0)
        if self._processor_thread is not None:
            self._processor_thread.join(timeout=self._QUEUE_TIMEOUT_S + 1.0)

        with self._clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except OSError:
                    pass
            self.clients.clear()

        try:
            self.sock.close()
        except OSError:
            pass

        logger.info("CommunicationLayer stopped.")

    def _listen_for_connections(self) -> None:
        while self.running:
            try:
                client, addr = self.sock.accept()
                logger.info("Connection from %s", addr)
                with self._clients_lock:
                    self.clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                if not self.running:
                    break
                logger.exception("Unexpected OSError in listener thread")
                break

    def _process_messages(self) -> None:
        while self.running:
            try:
                message = self.message_queue.get(
                    timeout=self._QUEUE_TIMEOUT_S
                )
            except queue.Empty:
                continue

            try:
                self._route_message(message)
            except Exception:
                logger.exception("Error processing message: %s", message)
            finally:
                self.message_queue.task_done()

    def _route_message(self, message: Dict[str, Any]) -> None:
        component_type = message.get("component_type")
        if component_type == "gnn":
            self._send_to_gnn(message)
        elif component_type == "agent":
            self._send_to_agent(message)
        elif component_type == "grid":
            self._send_to_grid(message)
        else:
            logger.warning("Unknown component_type: %r", component_type)

    def _send_to_gnn(self, message: Dict[str, Any]) -> None:
        if self._gnn_coordinator is None:
            default_graph = create_grid_graph(
                num_households=int(config.get("num_households", 10)),
                num_solar_panels=int(config.get("num_solar_panels", 5)),
                num_wind_turbines=int(config.get("num_wind_turbines", 3)),
            )
            self._gnn_coordinator = GNNCoordinator(
                graph=default_graph,
                log_dir=config.LOG_DIR,
            )

        self._gnn_coordinator.run(
            num_epochs=message.get("epochs", 100)
        )

        log_training_data(
            log_dir=config.LOG_DIR,
            episode=message.get("episode", 1),
            total_reward=message.get("total_reward", 0.0),
            avg_house_reward=message.get("avg_house_reward", 0.0),
            avg_market_reward=message.get("avg_market_reward", 0.0),
            avg_grid_reward=message.get("avg_grid_reward", 0.0),
            step=message.get("step", 1),
        )

    def _send_to_agent(self, message: Dict[str, Any]) -> None:
        agent_id = message.get("agent_id")
        reward = message.get("reward", 0.0)
        action = message.get("action", "none")
        logger.info(
            "Agent %s received reward: %s, action: %s",
            agent_id, reward, action,
        )

    def _send_to_grid(self, message: Dict[str, Any]) -> None:
        log_simulation_data(
            log_dir=config.LOG_DIR,
            timestamp=message.get(
                "timestamp", datetime.now().isoformat()
            ),
            grid_balance=message.get("grid_balance", 0.0),
            market_balance=message.get("market_balance", 0.0),
            household_consumption=message.get("household_consumption", 0.0),
            solar_production=message.get("solar_production", 0.0),
            wind_production=message.get("wind_production", 0.0),
        )

    def send_message(self, message: Dict[str, Any]) -> None:
        self.message_queue.put(message)
