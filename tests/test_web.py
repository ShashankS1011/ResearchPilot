import json
from http.client import HTTPConnection
from threading import Thread

from research.web import create_handler


def test_health_endpoint_is_public(monkeypatch):
    monkeypatch.setenv("WEB_AUTH_TOKEN", "test-token")
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_research_endpoint_requires_authentication(monkeypatch):
    monkeypatch.setenv("WEB_AUTH_TOKEN", "test-token")
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler())
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("POST", "/research", body='{"query":"test"}')
        response = connection.getresponse()
        assert response.status == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
