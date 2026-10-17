"""
Tunable thresholds for the incident engine. Every number here is a HAND-SET
heuristic, not a learned value. They are collected in one place so they can be
reviewed, challenged and (later) tuned from admin feedback recorded in
incident_actions.
"""
import os

# Signal extraction
SIGNAL_LOST_SEC = 180            # no fix for this long counts as signal lost
GOOD_FIX_ACCURACY_M = 100        # fixes worse than this are not used for position claims
POOR_FIX_ACCURACY_M = 50         # "poor GPS" if most recent fixes are worse than this
ZONE_EXIT_TOLERANCE_M = 25       # hysteresis when walking back through a zone's edge
REST_MIN_DURATION_MIN = 15       # RESTING shorter than this is never unusual
REST_TAIL_P = 0.10               # HSMM duration tail probability considered unusual
ISOLATED_SCORE = 0.75            # safety_assessments.isolation_score; 0.5 means "lookup failed"
NIGHT_START_HOUR, NIGHT_END_HOUR = 21, 5
LOCAL_TZ = os.environ.get("AGENT_LOCAL_TZ", "Asia/Kolkata")

# Levels (automatic detection never exceeds "high"; "critical" is reserved for a panic alert)
LOW_MIN, MEDIUM_MIN, HIGH_MIN = 20, 40, 60
LEVEL_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
AUTO_CLEAR_TICKS = 5             # consecutive quiet ticks before an incident auto-resolves
DISMISS_SUPPRESS_SEC = 3600      # a dismissed incident is not re-raised at the same level for this long
UPDATE_SCORE_DELTA = 8           # avoid a DB write per tick for tiny score wiggles

# Cluster (several travellers in the same verified zone)
CLUSTER_MIN_TRAVELLERS = 3
CLUSTER_WINDOW_SEC = 1800

# Briefs
BRIEF_MIN_LEVEL = "medium"
BRIEF_COOLDOWN_SEC = 300
MAX_LLM_BRIEFS_PER_TICK = 5
SNAPSHOT_LOCATION_ROWS = 400     # newest rows per trip (about 33 min at 5 s pings)
