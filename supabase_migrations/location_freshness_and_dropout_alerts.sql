-- SafeTrail: location freshness (Stage 1) + scheduled dropout alerts (Stage 2)
-- Review before applying to production. Uses existing locations, travel_cards,
-- and alerts tables; adds one nullable trip reference and supporting indexes.

-- Stage 1: extend the existing latest_locations view without changing its first
-- six columns, so existing clients keep working and can opt into the new fields.
create or replace view public.latest_locations as
select distinct on (user_id)
  user_id, lat, lon, accuracy, speed, recorded_at,
  travel_card_id,
  greatest(0, extract(epoch from (now() - recorded_at)))::bigint as location_age_seconds,
  case
    when now() - recorded_at <= interval '20 seconds' then 'LIVE'
    when now() - recorded_at <= interval '90 seconds' then 'DELAYED'
    else 'STALE'
  end as freshness_status
from public.locations
order by user_id, recorded_at desc, id desc;

alter view public.latest_locations set (security_invoker = on);
grant select on public.latest_locations to authenticated;

-- A trip-specific variant is useful for audit/debug views. Freshness is computed
-- at query time, so it naturally changes even when no new GPS row arrives.
create or replace view public.latest_locations_status as
select
  ll.user_id,
  ll.lat,
  ll.lon,
  ll.accuracy,
  ll.speed,
  ll.recorded_at,
  ll.travel_card_id,
  greatest(0, extract(epoch from (now() - ll.recorded_at)))::bigint
    as location_age_seconds,
  case
    when now() - ll.recorded_at <= interval '20 seconds' then 'LIVE'
    when now() - ll.recorded_at <= interval '90 seconds' then 'DELAYED'
    else 'STALE'
  end as freshness_status
from (
  select distinct on (l.user_id, coalesce(l.travel_card_id::text, 'legacy'))
    l.user_id, l.lat, l.lon, l.accuracy, l.speed, l.recorded_at, l.travel_card_id
  from public.locations l
  order by l.user_id, coalesce(l.travel_card_id::text, 'legacy'), l.recorded_at desc, l.id desc
) ll;

alter view public.latest_locations_status set (security_invoker = on);
grant select on public.latest_locations_status to authenticated;

-- Stage 2: associate dropout alerts with the trip that lost contact.
alter table public.alerts
  add column if not exists travel_card_id uuid references public.travel_cards(id) on delete set null;

create index if not exists alerts_dropout_trip_idx
  on public.alerts (user_id, travel_card_id, created_at desc)
  where type = 'dropout';

-- At most one unresolved dropout alert per trip. A resolved incident can recur.
create unique index if not exists alerts_one_open_dropout_per_trip_idx
  on public.alerts (user_id, travel_card_id)
  where type = 'dropout'
    and travel_card_id is not null
    and status in ('open', 'acknowledged');

-- Stage 3: distributed per-trip lock to serialize concurrent HSMM requests across
-- Render workers/instances. Expiring leases recover if a worker crashes.
create table if not exists public.trip_scoring_locks (
  trip_id text primary key,
  lock_token uuid not null,
  locked_until timestamptz not null,
  updated_at timestamptz not null default now()
);
alter table public.trip_scoring_locks enable row level security;

create or replace function public.try_acquire_trip_scoring_lock(
  p_trip_id text,
  p_lock_token uuid,
  p_lease_seconds integer default 300
) returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_acquired boolean := false;
begin
  if p_trip_id is null or length(p_trip_id) = 0 or p_lock_token is null then
    raise exception 'trip id and lock token are required';
  end if;
  if p_lease_seconds < 30 or p_lease_seconds > 900 then
    raise exception 'lease must be between 30 and 900 seconds';
  end if;

  insert into public.trip_scoring_locks as l (trip_id, lock_token, locked_until, updated_at)
  values (p_trip_id, p_lock_token, now() + make_interval(secs => p_lease_seconds), now())
  on conflict (trip_id) do update
    set lock_token = excluded.lock_token,
        locked_until = excluded.locked_until,
        updated_at = now()
    where l.locked_until < now()
  returning true into v_acquired;

  return coalesce(v_acquired, false);
end;
$$;

create or replace function public.release_trip_scoring_lock(
  p_trip_id text,
  p_lock_token uuid
) returns boolean
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_deleted integer;
begin
  delete from public.trip_scoring_locks
  where trip_id = p_trip_id and lock_token = p_lock_token;
  get diagnostics v_deleted = row_count;
  return v_deleted = 1;
end;
$$;

-- The Python scoring service uses the service-role key to call these RPCs.
revoke all on function public.try_acquire_trip_scoring_lock(text, uuid, integer) from public, anon, authenticated;
revoke all on function public.release_trip_scoring_lock(text, uuid) from public, anon, authenticated;
grant execute on function public.try_acquire_trip_scoring_lock(text, uuid, integer) to service_role;
grant execute on function public.release_trip_scoring_lock(text, uuid) to service_role;

-- Run from pg_cron every minute. A location older than 3 minutes on an active
-- trip creates a dropout alert; a fresh ping automatically resolves it. No
-- location history means no dropout alert until the trip has sent its first fix.
create or replace function public.check_location_dropouts()
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_resolved integer := 0;
  v_created integer := 0;
begin
  -- Close dropout incidents when fresh data resumes.
  with latest as (
    select distinct on (l.travel_card_id)
      l.travel_card_id, l.user_id, l.recorded_at
    from public.locations l
    where l.travel_card_id is not null
    order by l.travel_card_id, l.recorded_at desc, l.id desc
  ), resolved as (
    update public.alerts a
       set status = 'resolved'
      from latest l
      join public.travel_cards tc on tc.id = l.travel_card_id
     where a.type = 'dropout'
       and a.travel_card_id = l.travel_card_id
       and a.user_id = l.user_id
       and a.status in ('open', 'acknowledged')
       and (tc.status <> 'active' or tc.end_date < current_date
            or l.recorded_at >= now() - interval '3 minutes')
    returning a.id
  )
  select count(*) into v_resolved from resolved;

  -- Create an alert for active cards with an existing but stale last location.
  with latest as (
    select distinct on (l.travel_card_id)
      l.travel_card_id, l.user_id, l.lat, l.lon, l.recorded_at
    from public.locations l
    where l.travel_card_id is not null
    order by l.travel_card_id, l.recorded_at desc, l.id desc
  ), inserted as (
    insert into public.alerts (
      user_id, travel_card_id, type, severity, lat, lon, message, status
    )
    select
      tc.user_id,
      tc.id,
      'dropout',
      case when now() - latest.recorded_at > interval '10 minutes' then 'critical' else 'warning' end,
      latest.lat,
      latest.lon,
      format(
        'No location update for %s seconds. Last known location recorded at %s.',
        floor(extract(epoch from (now() - latest.recorded_at)))::bigint,
        latest.recorded_at
      ),
      'open'
    from public.travel_cards tc
    join latest on latest.travel_card_id = tc.id and latest.user_id = tc.user_id
    where tc.status = 'active'
      and tc.start_date <= current_date
      and tc.end_date >= current_date
      and latest.recorded_at < now() - interval '3 minutes'
      and not exists (
        select 1 from public.alerts existing
        where existing.type = 'dropout'
          and existing.travel_card_id = tc.id
          and existing.user_id = tc.user_id
          and existing.status in ('open', 'acknowledged')
      )
    on conflict do nothing
    returning id
  )
  select count(*) into v_created from inserted;

  return jsonb_build_object('resolved', v_resolved, 'created', v_created, 'checked_at', now());
end;
$$;

revoke all on function public.check_location_dropouts() from public, anon, authenticated;
grant execute on function public.check_location_dropouts() to service_role;

-- pg_cron is available on Supabase projects where the extension is enabled.
-- If CREATE EXTENSION is disallowed, enable pg_cron in Database > Extensions,
-- then rerun this section.
create extension if not exists pg_cron;
do $$
declare
  existing_job record;
begin
  for existing_job in select jobid from cron.job where jobname = 'safetrail-location-dropout-check'
  loop
    perform cron.unschedule(existing_job.jobid);
  end loop;
  perform cron.schedule(
    'safetrail-location-dropout-check',
    '* * * * *',
    $job$select public.check_location_dropouts();$job$
  );
end;
$$;

-- Make the newly added RPC functions visible to PostgREST without waiting for
-- the schema cache's periodic refresh.
notify pgrst, 'reload schema';
