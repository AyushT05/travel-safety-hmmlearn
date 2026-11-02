-- 20261101_trip_insights_and_notifications.sql
-- AI trip analyst: one live insight row per trip (admin sees it) and a notification inbox per traveller
-- (the mobile app subscribes to it). Requires public.is_admin(uuid). Safe to re-run.

-- ------------------------------------------------------------------ trip_insights (one row per trip)
create table if not exists public.trip_insights (
  travel_card_id   uuid primary key references public.travel_cards(id) on delete cascade,
  user_id          uuid not null references auth.users(id) on delete cascade,
  status           text not null default 'normal' check (status in ('normal', 'watch', 'alert')),
  headline         text not null,
  admin_summary    text not null,
  user_message     text,                                -- what the traveller was told (null = nothing sent)
  situation        jsonb not null default '{}'::jsonb,  -- where, heading, route adherence, movement, connectivity
  abnormal_events  jsonb not null default '[]'::jsonb,  -- deterministic detections, each with evidence
  explanations     jsonb not null default '[]'::jsonb,
  admin_actions    jsonb not null default '[]'::jsonb,
  source           text not null default 'template' check (source in ('template', 'llm')),
  llm_error        text,
  event_signature  text,                                -- hash of the event set; the LLM is re-asked only when it changes
  generated_at     timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);
create index if not exists trip_insights_status_idx on public.trip_insights (status, updated_at desc);
create index if not exists trip_insights_user_idx on public.trip_insights (user_id);

alter table public.trip_insights enable row level security;
drop policy if exists trip_insights_select on public.trip_insights;
create policy trip_insights_select on public.trip_insights for select to authenticated
  using (
    public.is_admin(auth.uid())
    or user_id = auth.uid()
    or exists (select 1 from public.tracking_requests r
               where r.requester_id = auth.uid() and r.traveler_id = trip_insights.user_id and r.status = 'approved')
  );
drop policy if exists trip_insights_service on public.trip_insights;
create policy trip_insights_service on public.trip_insights for all to service_role using (true) with check (true);

-- ------------------------------------------------------------------ user_notifications (traveller inbox)
create table if not exists public.user_notifications (
  id              bigserial primary key,
  user_id         uuid not null references auth.users(id) on delete cascade,
  travel_card_id  uuid references public.travel_cards(id) on delete set null,
  kind            text not null,                         -- route_deviation | abrupt_stop | signal_restored | hazard_ahead | check_in | ...
  severity        text not null default 'notice' check (severity in ('info', 'notice', 'warning', 'urgent')),
  title           text not null,
  body            text not null,
  actions         jsonb not null default '[]'::jsonb,    -- [{id, label}] shown as buttons: confirm_ok | need_help
  data            jsonb not null default '{}'::jsonb,
  dedupe_key      text,
  source          text not null default 'agent',
  created_at      timestamptz not null default now(),
  read_at         timestamptz,
  response        text check (response in ('ok', 'need_help')),
  responded_at    timestamptz
);
create index if not exists user_notifications_user_idx on public.user_notifications (user_id, created_at desc);
create index if not exists user_notifications_trip_idx on public.user_notifications (travel_card_id, created_at desc);

alter table public.user_notifications enable row level security;
drop policy if exists user_notifications_select_own on public.user_notifications;
create policy user_notifications_select_own on public.user_notifications for select to authenticated
  using (user_id = auth.uid() or public.is_admin(auth.uid()));
drop policy if exists user_notifications_update_own on public.user_notifications;
create policy user_notifications_update_own on public.user_notifications for update to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());
drop policy if exists user_notifications_service on public.user_notifications;
create policy user_notifications_service on public.user_notifications for all to service_role using (true) with check (true);

-- RLS cannot restrict columns: a traveller may only set read_at / response / responded_at.
create or replace function public.user_notifications_guard_update()
returns trigger language plpgsql as $$
declare r_read timestamptz; r_resp text; r_at timestamptz;
begin
  if current_user <> 'authenticated' then return NEW; end if;
  r_read := NEW.read_at; r_resp := NEW.response; r_at := NEW.responded_at;
  NEW := OLD;
  NEW.read_at := coalesce(OLD.read_at, r_read);
  if OLD.response is null and r_resp is not null then
    NEW.response := r_resp;
    NEW.responded_at := coalesce(r_at, now());
  end if;
  return NEW;
end $$;
drop trigger if exists user_notifications_guard_update on public.user_notifications;
create trigger user_notifications_guard_update before update on public.user_notifications
  for each row execute function public.user_notifications_guard_update();

-- ------------------------------------------------------------------ Realtime
do $$
begin
  if not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'trip_insights') then
    alter publication supabase_realtime add table public.trip_insights;
  end if;
  if not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'user_notifications') then
    alter publication supabase_realtime add table public.user_notifications;
  end if;
end $$;

-- ------------------------------------------------------------------ incidents: allow the AI analyst's kind
-- (the check constraint on incidents.kind only lists trip_risk / panic / zone_cluster; no change needed:
--  abnormal events feed the EXISTING trip_risk incident through the existing scoring.)

notify pgrst, 'reload schema';
