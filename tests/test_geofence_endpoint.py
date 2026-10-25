import pytest
from fastapi.testclient import TestClient

import app as service


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc")
    return TestClient(service.app)


H = {"x-webhook-secret": "s3cret"}


def test_requires_the_shared_secret(client):
    assert client.post("/geofence/sync/gdacs").status_code == 401
    assert client.post("/geofence/sync/gdacs", headers={"x-webhook-secret": "wrong"}).status_code == 401


def test_runs_a_sync_and_returns_the_summary(client, monkeypatch):
    monkeypatch.setattr(service, "sync_gdacs", lambda store: {"ingested": 2, "expired": 1})
    r = client.post("/geofence/sync/gdacs", headers=H)
    assert r.status_code == 200 and r.json() == {"ingested": 2, "expired": 1}


def test_failure_is_a_502_and_does_not_leak_details(client, monkeypatch):
    def boom(store): raise RuntimeError("secret internals")
    monkeypatch.setattr(service, "sync_gdacs", boom)
    r = client.post("/geofence/sync/gdacs", headers=H)
    assert r.status_code == 502 and "secret internals" not in r.text and "RuntimeError" in r.text


def test_overlapping_runs_are_skipped_and_the_lock_is_released(client, monkeypatch):
    monkeypatch.setattr(service, "sync_gdacs", lambda store: {"ok": True})
    assert service._GDACS_LOCK.acquire(blocking=False)
    try:
        assert client.post("/geofence/sync/gdacs", headers=H).json() == {"skipped": "sync_in_progress"}
    finally:
        service._GDACS_LOCK.release()
    assert client.post("/geofence/sync/gdacs", headers=H).json() == {"ok": True}
    def boom(store): raise ValueError("x")
    monkeypatch.setattr(service, "sync_gdacs", boom)
    client.post("/geofence/sync/gdacs", headers=H)
    assert service._GDACS_LOCK.acquire(blocking=False)      # released even after a failure
    service._GDACS_LOCK.release()


def test_needs_supabase_configured(client, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL")
    assert client.post("/geofence/sync/gdacs", headers=H).status_code == 503
