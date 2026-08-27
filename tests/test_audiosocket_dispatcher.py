from __future__ import annotations

import socket
import threading
from uuid import uuid4

import pytest

from voiceprobe.media.live_asr import TYPE_UUID, recv_exact
from voiceprobe.telephony.audiosocket_dispatcher import (
    DISPATCH_HOST,
    DISPATCH_PORT,
    AudioSocketDispatcher,
    AudioSocketDispatcherError,
    validate_worker_port,
)


def _uuid_frame(call_id) -> bytes:
    payload = call_id.bytes
    return bytes((TYPE_UUID,)) + len(payload).to_bytes(2, "big") + payload


def _available_worker_port() -> int:
    for port in range(9500, 9600):
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            candidate.bind((DISPATCH_HOST, port))
        except OSError:
            candidate.close()
            continue
        candidate.close()
        return port
    raise RuntimeError("No free test worker port found.")


def test_worker_port_must_remain_in_reserved_loopback_range() -> None:
    assert validate_worker_port(9500) == 9500

    for invalid in (0, 9019, 10000, True, "9500"):
        with pytest.raises(AudioSocketDispatcherError):
            validate_worker_port(invalid)  # type: ignore[arg-type]


def test_dispatcher_rejects_duplicate_uuid_and_worker_port_routes() -> None:
    dispatcher = AudioSocketDispatcher()
    first = uuid4()
    second = uuid4()

    dispatcher.register(first, 9500)

    with pytest.raises(AudioSocketDispatcherError, match="already registered"):
        dispatcher.register(first, 9501)

    with pytest.raises(AudioSocketDispatcherError, match="already registered"):
        dispatcher.register(second, 9500)


def test_dispatcher_routes_by_uuid_and_preserves_first_frame() -> None:
    call_id = uuid4()
    worker_port = _available_worker_port()
    worker_ready = threading.Event()
    observed: dict[str, bytes] = {}

    def worker() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((DISPATCH_HOST, worker_port))
            server.listen(1)
            worker_ready.set()
            connection, _ = server.accept()
            with connection:
                first = recv_exact(connection, 19)
                payload = recv_exact(connection, 5)
                observed["first"] = first or b""
                observed["payload"] = payload or b""
                connection.sendall(b"reply")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    assert worker_ready.wait(1.0)

    with AudioSocketDispatcher() as dispatcher:
        dispatcher.register(call_id, worker_port)

        with socket.create_connection((DISPATCH_HOST, DISPATCH_PORT), timeout=1.0) as client:
            frame = _uuid_frame(call_id)
            client.sendall(frame + b"hello")
            assert recv_exact(client, 5) == b"reply"

        assert dispatcher.registered_count == 0

    worker_thread.join(timeout=1.0)
    assert not worker_thread.is_alive()
    assert observed["first"] == _uuid_frame(call_id)
    assert observed["payload"] == b"hello"


def test_unknown_uuid_is_not_forwarded_to_registered_worker() -> None:
    expected = uuid4()
    unexpected = uuid4()
    worker_port = _available_worker_port()

    with AudioSocketDispatcher() as dispatcher:
        dispatcher.register(expected, worker_port)

        with socket.create_connection((DISPATCH_HOST, DISPATCH_PORT), timeout=1.0) as client:
            client.sendall(_uuid_frame(unexpected))
            client.settimeout(1.0)
            assert client.recv(1) == b""

        # Unknown traffic cannot consume another call's one-shot route.
        assert dispatcher.registered_count == 1
