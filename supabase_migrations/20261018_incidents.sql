-- 20261018_incidents.sql
-- Incident layer for the safety agent. Requires public.is_admin(uuid) (from your Phase 1 migration).
-- Before running, replace <YOUR_WEBHOOK_SECRET> (appears twice, last section) with the same value as
-- WEBHOOK_SECRET on Render. The Render URL is already filled in. Safe to re-run.

create table if not exists public.incidents (
  id               uuid primary key default gen_random_uuid(),
  dedupe_key       text not null,                      -- trip:<card> | panic:<alert> | cluster:<zone>
  kind             text not null check (kind in ('trip_risk', 'panic', 'zone_cluster')),
  user_id          uuid references auth.users(id),
  travel_card_id   uuid references public.travel_cards(id) on delete set null,
  zone_id          bigint,
  alert_id         bigint references public.alerts(id) on delete set null,   -- alerts.id is bigint identity
  level            text not null check (level in ('low', 'medium', 'high', 'critical')),
  peak_level       text check (peak_level in ('low', 'medium', 'high', 'critical')),
  score            integer not null,
  status           text not null default 'open' check (status in ('open', 'acknowledged', 'resolved', 'dismissed')),
  resolution       text,                               -- auto_cleared | trip_ended | admin reason code
  reasons          jsonb not null default '[]'::jsonb, -- one line per scoring rule that fired
  evidence         jsonb not null default '[]'::jsonb, -- the signals, each with evidence ids and timestamps
  last_lat         double precision,
  last_lon         double precision,
  last_fix_at      timestamptz,
  brief            jsonb,
  brief_source     text check (brief_source in ('template', 'llm')),
  brief_generated_at timestamptz,
  brief_error      text,
  clear_streak     integer not null default 0,
  opened_at        timestamptz not null default now(),
  updated_at       timestamptz not null default now(),
  resolved_at      timestamptz,
  resolved_by      uuid references auth.users(id)
);

create unique index if not exists incidents_one_open_per_key
  on public.incidents (dedupe_key) where status in ('open', 'acknowledged');
create index if not exists incidents_status_level_idx on public.incidents (status, level, updated_at desc);
create index if not exists incidents_key_idx on public.incidents (dedupe_key, opened_at desc);

create table if not exists public.incident_actions (
  id           bigserial primary key,
  incident_id  uuid not null references public.incidents(id) on delete cascade,
  actor        uuid not null default auth.uid() references auth.users(id),
  action       text not null check (action in ('acknowledged', 'resolved', 'dismissed', 'contacted', 'note')),
  reason       text check (reason in ('expected_behaviour', 'known_dead_zone', 'false_positive',
                                      'duplicate', 'tourist_confirmed_safe', 'other')),
  note         text check (char_length(note) <= 500),
  created_at   timestamptz not null default now()
);
create index if not exists incident_actions_incident_idx on public.incident_actions (incident_id, created_at);

-- ---------------------------------------------------------------- RLS (admins only)
alter table public.incidents enable row level security;
alter table public.incident_actions enable row level security;

drop policy if exists incidents_admin_select on public.incidents;
create policy incidents_admin_select on public.incidents for select to authenticated
  using (public.is_admin(auth.uid()));

drop policy if exists incidents_admin_update on public.incidents;
create policy incidents_admin_update on public.incidents for update to authenticated
  using (public.is_admin(auth.uid()))
  with check (public.is_admin(auth.uid()));

drop policy if exists incidents_service_all on public.incidents;
create policy incidents_service_all on public.incidents for all to service_role using (true) with check (true);

drop policy if exists incident_actions_admin_select on public.incident_actions;
create policy incident_actions_admin_select on public.incident_actions for select to authenticated
  using (public.is_admin(auth.uid()));

drop policy if exists incident_actions_admin_insert on public.incident_actions;
create policy incident_actions_admin_insert on public.incident_actions for insert to authenticated
  with check (actor = auth.uid()
              and public.is_admin(auth.uid()));

drop policy if exists incident_actions_service_all on public.incident_actions;
create policy incident_actions_service_all on public.incident_actions for all to service_role using (true) with check (true);

-- RLS cannot restrict columns. Admins may only change status/resolution; everything the
-- engine computes (score, evidence, brief, ...) is protected from client-side edits.
create or replace function public.incidents_guard_admin_update()
returns trigger language plpgsql as $$
declare
  v_status text; v_resolution text;
begin
  -- PostgREST runs each request under SET ROLE (authenticated / service_role), so current_user
  -- is the caller's database role. Only end-user (authenticated) edits are restricted; the
  -- scoring service (service_role) and SQL-editor/migration roles are not.
  if current_user <> 'authenticated' then
    return NEW;
  end if;
  v_status := NEW.status;
  v_resolution := NEW.resolution;
  NEW := OLD;
  NEW.status := v_status;
  NEW.resolution := v_resolution;
  if v_status in ('resolved', 'dismissed') and OLD.status not in ('resolved', 'dismissed') then
    NEW.resolved_at := now();
    NEW.resolved_by := auth.uid();
  end if;
  NEW.updated_at := now();
  return NEW;
end;
$$;

drop trigger if exists incidents_guard_admin_update on public.incidents;
create trigger incidents_guard_admin_update before update on public.incidents
  for each row execute function public.incidents_guard_admin_update();

-- ---------------------------------------------------------------- Realtime
do $$
begin
  if not exists (select 1 from pg_publication_tables
                 where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'incidents') then
    alter publication supabase_realtime add table public.incidents;
  end if;
end $$;

-- ---------------------------------------------------------------- Triggers and schedule
create extension if not exists pg_net;

-- Panic alerts: notify the agent. This is fire-and-forget and independent of the existing
-- alert/SMS flow; if it fails, the next /agent/tick creates the incident anyway.
create or replace function public.notify_agent_panic()
returns trigger language plpgsql security definer as $$
begin
  perform net.http_post(
    url     := 'https://travel-safety-hmmlearn.onrender.com/agent/panic',
    headers := jsonb_build_object('Content-Type', 'application/json', 'X-Webhook-Secret', '<YOUR_WEBHOOK_SECRET>'),
    body    := jsonb_build_object('alert_id', NEW.id),
    timeout_milliseconds := 30000
  );
  return NEW;
exception when others then
  return NEW;      -- never let the agent hook block an SOS insert
end;
$$;

drop trigger if exists alerts_panic_agent_hook on public.alerts;
create trigger alerts_panic_agent_hook after insert on public.alerts
  for each row when (NEW.type = 'panic') execute function public.notify_agent_panic();

create extension if not exists pg_cron;
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname = 'safetrail-agent-tick' loop
    perform cron.unschedule(j.jobid);
  end loop;
  perform cron.schedule('safetrail-agent-tick', '* * * * *', $cron$
    select net.http_post(
      url     := 'https://travel-safety-hmmlearn.onrender.com/agent/tick',
      headers := jsonb_build_object('Content-Type', 'application/json', 'X-Webhook-Secret', '<YOUR_WEBHOOK_SECRET>'),
      body    := '{}'::jsonb,
      timeout_milliseconds := 55000
    );
  $cron$);
end $$;
