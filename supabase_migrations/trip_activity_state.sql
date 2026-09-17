-- trip_activity_state.sql
--
-- Persists the HSMM pipeline's per-trip state so it survives the scoring
-- service restarting (Render's free tier sleeps after ~15 min idle, which
-- would otherwise silently wipe any in-memory state). `state` holds the
-- full pipeline.py state dict as jsonb; the other columns duplicate just
-- the latest score out of that blob so the dashboard can query or
-- subscribe to them directly without unpacking jsonb on the frontend.
--
-- Place this alongside the existing files in supabase/migrations/.

create table if not exists trip_activity_state (
  trip_id text primary key,
  state jsonb not null,
  current_state text,
  duration_minutes numeric,
  tail_probability numeric,
  llr numeric,
  updated_at timestamptz default now()
);

-- keep updated_at current on every upsert
create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger trip_activity_state_updated_at
before update on trip_activity_state
for each row execute function set_updated_at();

-- RLS: the scoring service writes via the service-role key (bypasses RLS
-- automatically), but the dashboard should be able to read this table
-- through the anon/authenticated key for realtime subscriptions.
alter table trip_activity_state enable row level security;

create policy "authenticated users can read trip activity state"
  on trip_activity_state for select
  to authenticated
  using (true);
