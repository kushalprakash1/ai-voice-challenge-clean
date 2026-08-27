"""Loopback-only AudioSocket dispatcher for concurrent VoiceProbe workers.

Asterisk continues to connect to the original trusted AudioSocket endpoint on
127.0.0.1:9019.  The dispatcher consumes only the first UUID frame, resolves a
pre-registered isolated worker port, forwards that exact frame unchanged, and
then proxies bytes bidirectionally for the lifetime of the call.

The dispatcher never dials, interprets speech, mutates patient state, or makes
scenario decisions.  It is a transport demultiplexer only.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from uuid import UUID

from voiceprobe.media.live_asr import TYPE_UUID, recv_exact

DISPATCH_HOST = "127.0.0.1"
DISPATCH_PORT = 9019
WORKER_PORT_MIN = 9020
WORKER_PORT_MAX = 9999
DEFAULT_ACCEPT_POLL_SECONDS = 0.20
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_REGISTERED_SESSIONS = 128


class AudioSocketDispatcherError(RuntimeError):
    """Raised when the local concurrent-media routing contract is violated."""


@dataclass(frozen=True, slots=True)
class AudioSocketRoute:
    """One one-shot UUID to isolated loopback-worker mapping."""

    call_id: UUID
    worker_port: int


def validate_worker_port(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AudioSocketDispatcherError("Worker port must be an integer.")
    if not WORKER_PORT_MIN <= value <= WORKER_PORT_MAX:
        raise AudioSocketDispatcherError(
            f"Worker port must be between {WORKER_PORT_MIN} and {WORKER_PORT_MAX}."
        )
    if value == DISPATCH_PORT:
        raise AudioSocketDispatcherError("Worker port cannot equal dispatcher port.")
    return value


class AudioSocketDispatcher:
    """Route concurrent Asterisk AudioSocket sessions by their UUID frame."""

    def __init__(
        self,
        *,
        host: str = DISPATCH_HOST,
        port: int = DISPATCH_PORT,
        max_registered_sessions: int = DEFAULT_MAX_REGISTERED_SESSIONS,
    ) -> None:
        if host != DISPATCH_HOST:
            raise AudioSocketDispatcherError(
                "Concurrent AudioSocket dispatcher must remain on 127.0.0.1."
            )
        if port != DISPATCH_PORT:
            raise AudioSocketDispatcherError(
                f"Concurrent AudioSocket dispatcher must remain on port {DISPATCH_PORT}."
            )
        if (
            isinstance(max_registered_sessions, bool)
            or not isinstance(max_registered_sessions, int)
            or not 1 <= max_registered_sessions <= 1024
        ):
            raise AudioSocketDispatcherError(
                "max_registered_sessions must be an integer between 1 and 1024."
            )

        self.host = host
        self.port = port
        self.max_registered_sessions = max_registered_sessions
        self._routes: dict[UUID, AudioSocketRoute] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._session_threads: set[threading.Thread] = set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def registered_count(self) -> int:
        with self._lock:
            return len(self._routes)

    def register(self, call_id: UUID, worker_port: int) -> AudioSocketRoute:
        """Register a one-shot route before its AMI Originate may occur."""

        if not isinstance(call_id, UUID):
            raise AudioSocketDispatcherError("call_id must be a UUID.")
        port = validate_worker_port(worker_port)

        with self._lock:
            if call_id in self._routes:
                raise AudioSocketDispatcherError(
                    f"AudioSocket UUID {call_id} is already registered."
                )
            if len(self._routes) >= self.max_registered_sessions:
                raise AudioSocketDispatcherError(
                    "AudioSocket dispatcher registration limit reached."
                )
            if any(route.worker_port == port for route in self._routes.values()):
                raise AudioSocketDispatcherError(
                    f"Worker port {port} is already registered."
                )

            route = AudioSocketRoute(call_id=call_id, worker_port=port)
            self._routes[call_id] = route
            return route

    def unregister(self, call_id: UUID) -> None:
        """Remove an unconsumed route, for example after worker startup failure."""
        with self._lock:
            self._routes.pop(call_id, None)

    def start(self) -> None:
        """Bind the trusted endpoint and begin accepting concurrent sessions."""
        if self._accept_thread is not None:
            raise AudioSocketDispatcherError("AudioSocket dispatcher already started.")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            server.bind((self.host, self.port))
            server.listen(self.max_registered_sessions)
            server.settimeout(DEFAULT_ACCEPT_POLL_SECONDS)
        except BaseException:
            server.close()
            raise

        self._server = server
        self._stop.clear()
        self._ready.set()
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="voiceprobe-audiosocket-dispatcher",
            daemon=True,
        )
        self._accept_thread.start()

    def close(self) -> None:
        """Stop accepting new sessions and close the dispatcher listener."""
        self._stop.set()
        self._ready.clear()

        server = self._server
        self._server = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

        thread = self._accept_thread
        self._accept_thread = None
        if thread is not None:
            thread.join(timeout=2.0)

        with self._lock:
            session_threads = tuple(self._session_threads)
            self._routes.clear()

        for session_thread in session_threads:
            session_thread.join(timeout=2.0)

    def __enter__(self) -> AudioSocketDispatcher:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            server = self._server
            if server is None:
                return

            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue

            thread = threading.Thread(
                target=self._handle_session,
                args=(connection,),
                name="voiceprobe-audiosocket-route",
                daemon=True,
            )
            with self._lock:
                self._session_threads.add(thread)
            thread.start()

    def _handle_session(self, downstream: socket.socket) -> None:
        current = threading.current_thread()
        upstream: socket.socket | None = None

        try:
            header = recv_exact(downstream, 3)
            if header is None:
                return

            message_type = header[0]
            payload_length = int.from_bytes(header[1:3], "big")
            payload = recv_exact(downstream, payload_length)
            if payload is None:
                return

            if message_type != TYPE_UUID or len(payload) != 16:
                return

            call_id = UUID(bytes=payload)

            with self._lock:
                route = self._routes.pop(call_id, None)

            # Unknown or already-consumed UUIDs are never forwarded.
            if route is None:
                return

            upstream = socket.create_connection(
                (DISPATCH_HOST, route.worker_port),
                timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS,
            )
            upstream.settimeout(None)
            downstream.settimeout(None)

            # Preserve the exact first AudioSocket frame for the existing
            # worker.  Downstream v2/v3 UUID validation therefore stays intact.
            upstream.sendall(header + payload)

            left = threading.Thread(
                target=self._pump,
                args=(downstream, upstream),
                name="voiceprobe-audiosocket-downstream-upstream",
                daemon=True,
            )
            right = threading.Thread(
                target=self._pump,
                args=(upstream, downstream),
                name="voiceprobe-audiosocket-upstream-downstream",
                daemon=True,
            )
            left.start()
            right.start()
            left.join()
            right.join()
        except (OSError, ValueError):
            return
        finally:
            for connection in (downstream, upstream):
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
            with self._lock:
                self._session_threads.discard(current)

    @staticmethod
    def _pump(source: socket.socket, destination: socket.socket) -> None:
        try:
            while True:
                chunk = source.recv(65536)
                if not chunk:
                    break
                destination.sendall(chunk)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass
