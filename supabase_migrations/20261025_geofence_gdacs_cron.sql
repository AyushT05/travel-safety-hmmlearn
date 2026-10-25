-- Hourly GDACS sync for dynamic geo-fencing. Mirrors the existing safetrail-agent-tick job.
-- Run AFTER the risk_zones migrations (20261024 and 20261025_risk_zones_upsert_fix) in the main repo.
--
-- Replace <YOUR_WEBHOOK_SECRET> with your WEBHOOK_SECRET before running, and do not commit the real value.
-- Zones expire on their own 6 hours after the last successful sync, so if this job stops, stale
-- disaster zones disappear instead of staying active.
--
-- timeout 55000 ms: the service may need to wake up (Render cold start) and then call GDACS.

create extension if not exists pg_cron;
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname = 'margarakshak-gdacs-sync' loop
    perform cron.unschedule(j.jobid);
  end loop;
  perform cron.schedule('margarakshak-gdacs-sync', '7 * * * *', $cron$
    select net.http_post(
      url     := 'https://travel-safety-hmmlearn.onrender.com/geofence/sync/gdacs',
      headers := jsonb_build_object('Content-Type', 'application/json', 'X-Webhook-Secret', '<YOUR_WEBHOOK_SECRET>'),
      body    := '{}'::jsonb,
      timeout_milliseconds := 55000
    );
  $cron$);
end $$;

-- Check it afterwards:
--   select jobname, schedule, active from cron.job where jobname = 'margarakshak-gdacs-sync';
--   select status_code, left(content::text, 200), created from net._http_response order by created desc limit 3;
