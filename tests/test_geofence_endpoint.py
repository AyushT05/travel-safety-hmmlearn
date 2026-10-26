import pytest

# These tests import the whole service, which needs the full runtime stack from requirements.txt.
# Without it they are skipped (not a collection error), so the pure geofence tests still run.
for _mod in ("fastapi", "httpx", "hmmlearn", "pymupdf"):
    pytest.importorskip(_mod)

from fastapi.testclient import TestClient  # noqa: E402

import app as service  # noqa: E402


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


# ------------------------------------------------------------------ the live geofence hook on /score
LOCATION = {"type": "INSERT", "table": "locations", "schema": "public",
            "record": {"id": 1, "user_id": "u1", "travel_card_id": "c1", "lat": 12.3, "lon": 76.65, "accuracy": 5,
                       "speed": 1, "created_at": "2026-10-24T15:30:00+00:00", "observed_at": "2026-10-24T15:29:58+00:00"}}


@pytest.fixture
def spy(monkeypatch):
    calls = []
    monkeypatch.setattr(service._GEOFENCE_POOL, "submit", lambda fn, rec: calls.append(rec))
    return calls


def test_hook_is_off_unless_explicitly_enabled(client, monkeypatch, spy):
    monkeypatch.delenv("GEOFENCE_ALERTS", raising=False)
    assert client.post("/score", json=LOCATION, headers=H).status_code == 200
    assert spy == []


def test_hook_passes_the_real_observation_time_when_enabled(client, monkeypatch, spy):
    monkeypatch.setenv("GEOFENCE_ALERTS", "on")
    assert client.post("/score", json=LOCATION, headers=H).status_code == 200
    assert len(spy) == 1 and spy[0]["observed_at"] == "2026-10-24T15:29:58+00:00" and spy[0]["user_id"] == "u1"


def test_a_broken_geofence_never_breaks_scoring(client, monkeypatch):
    monkeypatch.setenv("GEOFENCE_ALERTS", "on")

    def boom(fn, rec): raise RuntimeError("pool exploded")
    monkeypatch.setattr(service._GEOFENCE_POOL, "submit", boom)
    r = client.post("/score", json={**LOCATION, "record": {**LOCATION["record"], "id": 2, "created_at": "2026-10-24T15:31:00+00:00"}}, headers=H)
    assert r.status_code == 200 and "trip_id" in r.json()


def test_evaluation_errors_are_swallowed(monkeypatch):
    class Boom:
        def handle_location(self, rec): raise RuntimeError("db down")
    monkeypatch.setattr(service, "_geofence_monitor", lambda: Boom())
    service._geofence_run({"user_id": "u"})              # must not raise
