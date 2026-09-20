-- locations_webhook_trigger.sql
--
-- Workaround for Supabase's Database Webhooks dashboard feature failing
-- with "schema supabase_functions does not exist" (a known, recurring
-- provisioning gap on some projects, see supabase/supabase#20056,
-- #32700, #12447 on GitHub). Rather than depend on that feature, this
-- writes the trigger directly using pg_net, which lives in its own `net`
-- schema and isn't affected by the missing supabase_functions schema.
--
-- Replace the two placeholders below before running this:
--   <YOUR_RENDER_URL>     e.g. https://your-service.onrender.com/score
--   <YOUR_WEBHOOK_SECRET> the same value set as WEBHOOK_SECRET on Render

create extension if not exists pg_net;

create or replace function public.notify_location_insert()
returns trigger
language plpgsql
security definer
as $$
begin
  perform net.http_post(
    url     := '<YOUR_RENDER_URL>',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'X-Webhook-Secret', '<YOUR_WEBHOOK_SECRET>'
    ),
    body    := jsonb_build_object(
      'type', 'INSERT',
      'table', 'locations',
      'schema', 'public',
      'record', to_jsonb(NEW) || jsonb_build_object('created_at', now()),
      'old_record', null
    ),
    timeout_milliseconds := 30000
  );
  return NEW;
end;
$$;

drop trigger if exists locations_insert_webhook on public.locations;

create trigger locations_insert_webhook
after insert on public.locations
for each row execute function public.notify_location_insert();

-- NOTE: this embeds the webhook secret in plain text inside the function
-- body, visible to anyone with SQL access to your project (which, for a
-- solo/small-team student project, is just you). If you want to avoid
-- that later, move the secret into Supabase Vault and read it with
-- `vault.decrypted_secrets` inside the function instead of hardcoding it
-- here. Not necessary to do now, worth knowing it's the next step up.
