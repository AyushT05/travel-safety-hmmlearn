-- ============================================================================
-- 32. Planned Routes and Route Deviations Migration
-- ============================================================================
-- Adds planned_route jsonb column to travel_cards for storing full road polylines
-- and route legs for AI route deviation and corridor monitoring. Safe to re-run.

ALTER TABLE public.travel_cards
  ADD COLUMN IF NOT EXISTS planned_route jsonb;

-- Index for open deviation alerts per trip/user
CREATE INDEX IF NOT EXISTS idx_alerts_deviation_open
  ON public.alerts (user_id, travel_card_id, created_at desc)
  WHERE type = 'deviation' AND status = 'open';

NOTIFY pgrst, 'reload schema';
