# SafeTrail reliability changes

This branch contains the backend/migration portion of three changes:

1. **Stage 1 — freshness:** `latest_locations` now exposes `travel_card_id`, `location_age_seconds`, and `freshness_status` (`LIVE` <=20s, `DELAYED` <=90s, otherwise `STALE`). It is computed at query time, so status ages even if no new row arrives. The mobile/web frontend repository was not included in the uploaded ZIP, so a drop-in `frontend-integration/useDevices.js` and `LocationFreshnessBadge.jsx` are provided separately for integration into the frontend.
2. **Stage 2 — dropout alerts:** `check_location_dropouts()` checks active travel cards every minute, creates a single unresolved `dropout` alert after 3 minutes without a fix (critical after 10 minutes), and resolves the active dropout alert when fresh location data resumes. It only considers trips with at least one recorded location. This is a communication-loss alert, not a claim that the traveller is in danger.
3. **Concurrent scoring:** the `/score` handler acquires a PostgreSQL-backed per-trip lease before load -> inference -> save. This serializes requests for the same trip across Render workers/instances. The state records the latest processed location ID and timestamp to skip duplicate/out-of-order webhook deliveries. Different trips can score concurrently.

## Apply in this order

1. Back up the current SQL/migration state and confirm `WEBHOOK_SECRET`, `SUPABASE_URL`, and `SUPABASE_SERVICE_ROLE_KEY` are set on Render. The service now fails closed if the webhook secret is missing in a deployed/Supabase-configured environment.
2. Run `supabase_migrations/location_freshness_and_dropout_alerts.sql` in the Supabase SQL Editor. If `pg_cron` cannot be created, enable it in Database > Extensions and rerun the migration. Review the output for errors; SQL Editor does not make the whole script safe to blindly rerun if a middle statement failed.
3. Deploy this folder to the existing Render scoring service.
4. Verify freshness in Supabase:

   ```sql
   select user_id, travel_card_id, recorded_at, location_age_seconds, freshness_status
   from public.latest_locations
   order by recorded_at desc limit 20;
   ```

5. Manually test the scheduled function:

   ```sql
   select public.check_location_dropouts();
   select id, user_id, travel_card_id, type, severity, status, created_at, message
   from public.alerts where type = 'dropout' order by created_at desc limit 20;
   ```

6. Confirm the scheduled job exists:

   ```sql
   select jobid, jobname, schedule, command, active
   from cron.job where jobname = 'safetrail-location-dropout-check';
   ```

7. Copy `frontend-integration/useDevices.js` into the frontend's matching hook location (adjust the Supabase import path if needed), then import `LocationFreshnessBadge.jsx` into the device/traveller card and render `<LocationFreshnessBadge device={devices[userId]} />`. The uploaded archive contains only the Python scoring service, so these are integration-ready frontend files, not an automatic patch to the separate frontend repository.

## Thresholds and assumptions

- Dashboard freshness: LIVE <=20 seconds, DELAYED <=90 seconds, STALE >90 seconds.
- Dropout alert: active card's last recorded location older than 3 minutes; critical after 10 minutes.
- Active trip: `travel_cards.status = 'active'` and start/end dates include the current database date.
- No alert is created for a trip that has never emitted a location fix.
- The lock lease is 300 seconds; acquisition waits up to 25 seconds before returning HTTP 503. A worker crash releases the lock by lease expiry.

## Security note

A webhook secret was pasted in the earlier SQL in the conversation. Treat it as compromised and rotate it: update Render's `WEBHOOK_SECRET` and the secret used by the Supabase trigger together. Do not commit the real value into source control.

## Dynamic geo-fencing (steps 1 to 3, plus GDACS ingestion)

Code: `geofence/` (pure scoring, route sampling, alert state machine, GDACS parser, Supabase store).
Tests: `tests/test_geofence.py`, `tests/test_gdacs.py`, `tests/test_geofence_endpoint.py`
(fixture `tests/fixtures/gdacs_list_sample.json` is trimmed from a real GDACS list response).

Deploy order:
1. Main repo, Supabase SQL editor: `20261024_risk_zones_and_geofence_state.sql`, then
   `20261025_risk_zones_upsert_fix.sql` (the first created a partial unique index that breaks upserts).
2. Deploy this service (adds `POST /geofence/sync/gdacs`, protected by the same X-Webhook-Secret).
3. Trigger once by hand: `curl.exe -X POST https://travel-safety-hmmlearn.onrender.com/geofence/sync/gdacs -H "x-webhook-secret: <WEBHOOK_SECRET>"`
   and check `select name, hazard_type, severity, radius_m, expires_at from risk_zones;`.
4. Run `supabase_migrations/20261025_geofence_gdacs_cron.sql` (put the real secret in only in the SQL editor).

Behaviour to remember:
- Only GDACS events with iscurrent = "true" inside the India-plus-margin box are stored. Override the box
  with env `GEOFENCE_INTEREST_BBOX="min_lat,min_lon,max_lat,max_lon"`.
- Cyclones use real wind-field polygons from `properties.url.geometry` (one zone per time step and wind band:
  60 km/h medium, 90 high, 120 critical). Each is valid from the previous step to the next, so a moving storm
  leaves no gap, and `Zone.valid_from` keeps a forecast field from applying before its time. Forecast
  confidence falls 0.1 per day ahead (floor 0.4). Used from the geometry response: only features with
  featuretype WindRadii. Ignored on purpose: the cumulative whole-storm swaths, the uncertainty cone, track
  lines, point circles, and degenerate zero-area polygons.
- If the geometry call fails the cyclone keeps its approximate disc (confidence 0.4); if it succeeds but holds
  no wind-field area the event is dropped. Floods, earthquakes and wildfires are still approximate discs.
- Discs live 6 hours from the last sync; wind fields until the end of their validity window; earthquakes 72 hours.
  `RiskZoneStore.zones_in_bbox(..., gdacs_fresh_after_iso=...)` ignores GDACS rows not refreshed recently, so a
  stopped sync cannot keep a stale forecast alive.
- A failed or empty GDACS response raises before any write: it never retires live hazards.
- Nothing in the live scoring or alert path calls the geofence evaluator yet (step 5 is not wired).
