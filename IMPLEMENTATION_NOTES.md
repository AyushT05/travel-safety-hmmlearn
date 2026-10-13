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
