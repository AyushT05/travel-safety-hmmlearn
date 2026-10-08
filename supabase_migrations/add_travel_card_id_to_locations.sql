-- add_travel_card_id_to_locations.sql
--
-- Root cause of the "old trip's trail/state bleeds into the new trip"
-- bug: `locations` only ever stored `user_id`, never which travel card
-- (trip) a given ping belonged to. Since a user can create, finish, delete,
-- and start a brand new travel card while keeping the same account, every
-- system keyed purely on user_id (the map's trail, and the HSMM activity
-- pipeline's trip_id) had no way to tell "this is a new trip" from "this
-- is more of the same trip" — they're structurally the same thing without
-- this column.
--
-- This is additive and safe to run on a live table: existing rows get
-- travel_card_id = null (they predate this fix, nothing to backfill them
-- with), new rows get it once LocationShare's ActiveTracking.js is updated
-- to send it (see the accompanying diff to that file).

alter table public.locations
  add column if not exists travel_card_id uuid references public.travel_cards(id);

create index if not exists idx_locations_travel_card_id
  on public.locations(travel_card_id);

-- Also index the pg_net trigger's payload correctly picks this up for
-- free: to_jsonb(NEW) in locations_webhook_trigger.sql automatically
-- includes every column on the row, so once this migration runs and
-- ActiveTracking.js starts sending travel_card_id, it starts appearing in
-- record.travel_card_id with zero changes needed to the trigger itself.
