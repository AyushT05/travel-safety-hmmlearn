# Incident agent

Correlates signals into prioritised incidents and writes evidence-cited briefs for administrators.

## What decides what

| Step | Who | Notes |
|---|---|---|
| Is there an incident, and how urgent? | Deterministic code (`signals.py`, `scoring.py`) | Same input always gives the same output. Thresholds are hand-set in `config.py`. |
| Brief text | Template first (`brief.template_brief`), LLM wording second | The template is always written first. The LLM can only reword and choose steps from a closed list. |
| Anything that contacts someone or escalates | A human | Steps marked "needs authorised approval" are flagged by code; the model cannot waive that. |

Invariants (enforced by tests): automatic detection never reaches `critical` (only a panic alert does);
one kind of evidence never reaches `high`; an SOS incident and template brief exist before any
network call other than the database; LLM output that asserts harm, cites unknown evidence, or uses
an unknown action is discarded; the model never receives phone numbers or personal names.

## Signals (v1)
zone_inside (verified zones only), signal_lost, prolonged_rest (HSMM duration tail), poor_gps (confidence modifier),
isolated_area (context only, lower trust because the assessment is client-triggered), panic, and cluster (3+ travellers in one zone within 30 min).

## Not implemented yet
Route deviation and route suggestions (itinerary stops are not geocoded), tourist-facing companion and multilingual
warnings, battery level as a signal, automated check-in ("are you OK?") workflow, outbound SMS/calls.

## Deploy
1. Run `supabase_migrations/20261017_activity_realtime.sql`, then `20261018_incidents.sql` (replace `<YOUR_RENDER_URL>` and `<YOUR_WEBHOOK_SECRET>` first).
2. Render env: `LLM_API_KEY` (or the existing `GROQ_API_KEY`) and `LLM_MODEL` (a current model slug from your provider).
   If either is unset the agent still works and briefs stay template-only. Optional: `AGENT_LOCAL_TZ` (default Asia/Kolkata, used for the night rule).
3. Redeploy the service. Check: `select jobname from cron.job;` lists `safetrail-agent-tick`; `curl -X POST -H "X-Webhook-Secret: ..." <url>/agent/tick` returns counters.
4. Copy the dashboard files and use an admin account: the Incidents button appears in the header.

## Operations
- `/agent/tick` runs every minute. It also creates incidents for any open panic alert whose webhook was lost.
- Every admin action is stored in `incident_actions` with a reason; dismissal reasons are the labels to tune thresholds from.
- `incidents.brief_error` says why LLM wording was not used (unavailable, or rejected and why).
- Scale: one query per active trip per tick. Fine for tens of trips; batch the snapshot queries before going far beyond that.

## Tests
`pip install -r requirements-dev.txt && python -m pytest tests`
