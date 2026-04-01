import socket
import threading
import queue
from typing import Dict, Any
from datetime import datetime

from config import config
from utils.logging_utils import log_simulation_data, log_training_data
from models.gnn_coordinator import GNNCoordinator


class CommunicationLayer:
    """
    In-process message dispatcher with an attached TCP listener socket.

    Architectural notes
    -------------------
    - This file is located in ``models/`` but is **not** a data model. It acts
      as infrastructure / service. Consider relocating to ``services/`` or
      ``infrastructure/`` in a future restructure.

    - The TCP socket layer (bind / listen / accept) is scaffolding for future
      distributed communication. Accepted sockets are stored but **never read
      from or written to**. All actual message dispatch is in-process via the
      internal ``queue.Queue``.

    - ``_send_to_gnn()`` instantiates a fresh ``GNNCoordinator`` and runs
      training epochs on every call. This is **orchestration, not
      communication**, and should be extracted in a future refactor.

    Public API
    ----------
    - ``start()``        — spawn listener and processor daemon threads.
    - ``send_message()`` — enqueue a message dict for async dispatch.
    - ``stop()``         — shut down threads and release all sockets.
    """

    _QUEUE_POLL_TIMEOUT_S = 0.1   # seconds; prevents busy-wait in processor loop
    _ACCEPT_TIMEOUT_S = 1.0       # seconds; allows listener to re-check self.running
    _THREAD_JOIN_TIMEOUT_S = 3.0  # seconds; bounded wait during stop()

    def __init__(self, host: str = "localhost", port: int = 5000, gnn_coordinator=None):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(self._ACCEPT_TIMEOUT_S)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

        self.clients: list = []           # accepted client sockets (currently unused)
        self.message_queue: queue.Queue = queue.Queue()
        self.running = False              # set True only by start()
        self._threads: list = []
        # Store long-lived coordinator to preserve trained state
        self._gnn_coordinator = gnn_coordinator

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn listener and message-processing daemon threads."""
        if self.running:
            return
        self.running = True

        listener = threading.Thread(
            target=self._listen_for_connections,
            daemon=True,
            name=f"comm-listener-{self.port}",
        )
        processor = threading.Thread(
            target=self._process_messages,
            daemon=True,
            name=f"comm-processor-{self.port}",
        )
        self._threads = [listener, processor]
        listener.start()
        processor.start()
        print(f"Communication Layer started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Shut down threads, close all sockets."""
        # Let processor threads drain queue before signaling stop.
        while True:
            try:
                self._send_message_to_component(self.message_queue.get_nowait())
            except queue.Empty:
                break
            except Exception as exc:
                print(f"[CommunicationLayer] Error processing message: {exc}")

        self.running = False

        # Close server socket to unblock accept() if it's between timeouts
        try:
            self.sock.close()
        except OSError:
            pass

        # Close every accepted client socket
        for client in self.clients:
            try:
                client.close()
            except OSError:
                pass
        self.clients.clear()

        # Wait for daemon threads to exit (bounded)
        for t in self._threads:
            t.join(timeout=self._THREAD_JOIN_TIMEOUT_S)
        self._threads.clear()

        print("Communication Layer stopped.")

    # ------------------------------------------------------------------
    # Thread targets
    # ------------------------------------------------------------------

    def _listen_for_connections(self) -> None:
        """Accept TCP connections until stopped.

        NOTE: Accepted sockets are stored but never read or written.
        """
        while self.running:
            try:
                client, addr = self.sock.accept()
                print(f"Connection from {addr}")
                self.clients.append(client)
            except socket.timeout:
                # Normal: periodically re-check self.running
                continue
            except OSError:
                # Socket closed during shutdown
                break

    def _process_messages(self) -> None:
        """Block-wait on queue and dispatch messages until stopped."""
        while self.running or not self.message_queue.empty():
            try:
                message = self.message_queue.get(
                    block=True, timeout=self._QUEUE_POLL_TIMEOUT_S
                )
            except queue.Empty:
                continue

            try:
                self._send_message_to_component(message)
            except Exception as exc:
                # Surface errors without killing the processing thread.
                print(f"[CommunicationLayer] Error processing message: {exc}")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _send_message_to_component(self, message: Dict[str, Any]) -> None:
        component_type = message.get("component_type")
        if component_type == "gnn":
            self._send_to_gnn(message)
        elif component_type == "agent":
            self._send_to_agent(message)
        elif component_type == "grid":
            self._send_to_grid(message)
        else:
            print(
                f"[CommunicationLayer] Unknown component_type: {component_type!r}, "
                f"message dropped."
            )

    def _send_to_gnn(self, message: Dict[str, Any]) -> None:
        """Orchestration stub — uses long-lived GNNCoordinator instance.

        WARNING: This is orchestration hidden inside a communication call.
        Should be extracted into a dedicated training service.
        """
        if self._gnn_coordinator is None:
            # Lazy instantiation requires graph; log error if not available
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                "GNNCoordinator not available; cannot process GNN message. "
                "Inject a coordinator instance when creating CommunicationLayer."
            )
            return

        self._gnn_coordinator.run(num_epochs=message.get("epochs", 100))
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
        """Stub: prints agent info to stdout.  Does not send over network."""
        agent_id = message.get("agent_id")
        reward = message.get("reward", 0.0)
        action = message.get("action", "none")
        price = message.get("price")
        consumption = message.get("consumption")
        energy = message.get("energy")
        print(
            f"Agent {agent_id} received reward: {reward}, action: {action}, "
            f"price: {price}, consumption: {consumption}, energy: {energy}"
        )

    def _send_to_grid(self, message: Dict[str, Any]) -> None:
        """Logs grid-level simulation data.  Does not send over network."""
        grid_balance = message.get("grid_balance", 0.0)
        market_balance = message.get("market_balance", 0.0)
        log_simulation_data(
            log_dir=config.LOG_DIR,
            # DETERMINISM NOTE: datetime.now() fallback introduces
            # non-determinism when "timestamp" key is absent.
            timestamp=message.get("timestamp", datetime.now().isoformat()),
            grid_balance=grid_balance,
            market_balance=market_balance,
            household_consumption=message.get("household_consumption", 0.0),
            solar_production=message.get("solar_production", 0.0),
            wind_production=message.get("wind_production", 0.0),
        )

    # ------------------------------------------------------------------
    # Public message interface
    # ------------------------------------------------------------------

    def send_message(self, message: Dict[str, Any]) -> None:
        """Enqueue a message for asynchronous dispatch by the processing thread.

        Messages are silently queued if ``start()`` has not been called.
        """
        self.message_queue.put(message)
