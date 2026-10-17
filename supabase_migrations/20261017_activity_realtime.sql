-- 20261017_activity_realtime.sql
-- Idempotent. Lets the dashboard receive trip_activity_state changes over
-- Realtime (the dashboard also polls every 15 s, so this is for latency, not
-- correctness) and speeds up the trail-history query.

do $$
begin
  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'trip_activity_state'
  ) then
    alter publication supabase_realtime add table public.trip_activity_state;
  end if;
end $$;

create index if not exists idx_locations_recorded_at
  on public.locations (recorded_at desc);
