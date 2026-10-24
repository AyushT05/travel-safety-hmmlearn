"""Per (user, zone) alert state machine: one alert per approach, one per entry, no GPS-flicker spam.

Pure: takes the stored row (or None) and the newest proximity state, returns the row to store and
which alert, if any, to raise. The caller persists the row and inserts the alert.

Approach and entry have separate cooldowns. An approach alert must never suppress the escalation
to an entry alert a few seconds later.
"""
from dataclasses import dataclass
from typing import Optional

CLEAR_AFTER = 3              # consecutive "away" observations before a zone counts as left
APPROACH_COOLDOWN_S = 900.0
ENTRY_COOLDOWN_S = 300.0

_AWAY = {"outside", "near"}


@dataclass
class Decision:
    row: dict
    emit: Optional[str]      # None | "approach" | "entry" | "exit"
    reason: str


def _row(state, streak, first, last_approach, last_entry):
    return {"state": state, "clear_streak": streak, "first_seen_at": first,
            "last_approach_at": last_approach, "last_entry_at": last_entry}


def transition(prev, prox_state, now_s, approach_cooldown_s=APPROACH_COOLDOWN_S, entry_cooldown_s=ENTRY_COOLDOWN_S):
    """prev: stored row or None. prox_state: from scoring.zone_proximity. now_s: epoch seconds."""
    p = prev or _row("clear", 0, now_s, None, None)
    state = p["state"]
    la, le = p.get("last_approach_at"), p.get("last_entry_at")
    first = p.get("first_seen_at") or now_s

    def cooled(last, window):
        return last is not None and now_s - last < window

    if prox_state == "inside":
        if state == "inside":
            return Decision(_row("inside", 0, first, la, le), None, "still inside")
        if cooled(le, entry_cooldown_s):
            return Decision(_row("inside", 0, now_s, la, le), None, "entry suppressed by cooldown")
        return Decision(_row("inside", 0, now_s, la, now_s), "entry", "entered zone")

    if prox_state in ("approaching", "uncertain_boundary"):
        if state == "inside":                         # edge flicker while inside: still inside
            return Decision(_row("inside", 0, first, la, le), None, "boundary uncertainty while inside")
        if state == "approaching":
            return Decision(_row("approaching", 0, first, la, le), None, "still approaching")
        if cooled(la, approach_cooldown_s):
            return Decision(_row("approaching", 0, now_s, la, le), None, "approach suppressed by cooldown")
        return Decision(_row("approaching", 0, now_s, now_s, le), "approach", "approaching zone")

    if prox_state in _AWAY:
        if state == "clear":
            return Decision(_row("clear", 0, first, la, le), None, "no change")
        streak = p.get("clear_streak", 0) + 1
        if streak < CLEAR_AFTER:
            return Decision(_row(state, streak, first, la, le), None, f"away {streak}/{CLEAR_AFTER}")
        emit = "exit" if state == "inside" else None
        return Decision(_row("clear", 0, first, la, le), emit, "left zone" if emit else "no longer approaching")

    return Decision(_row(state, p.get("clear_streak", 0), first, la, le), None, "unknown state ignored")
