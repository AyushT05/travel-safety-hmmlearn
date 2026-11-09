"""
Incident engine. Orchestrates: snapshot -> signals -> score -> incident row
(upsert, escalate, auto-clear, suppress after dismissal) -> brief.

Design guarantees:
  * Decisions (is this an incident? what level?) are deterministic; the LLM only
    words the brief and can never raise, lower or close an incident.
  * A panic alert gets an incident with a complete template brief before any
    network call other than the database; services lookup and LLM run after.
  * tick() also re-checks open panic alerts, so a lost webhook is self-healing.
"""
import logging
import threading

from . import config
from .context import likely_issues, weather_now
from .services import emergency_numbers
from .brief import (SYSTEM_PROMPT, BriefRejected, build_packet, build_user_message, playbook,
                    template_brief, validate_llm_brief)
from .llm import LLMUnavailable
from .scoring import score_signals
from .signals import Signal, extract_signals, last_known, _fixes as _signal_fixes
from .util import iso, parse_ts, hhmm

log = logging.getLogger("agent")


def _evidence(signals):
    return [{"kind": s.kind, "at": iso(s.at), "text": s.text, "evidence_ids": s.evidence_ids, "detail": s.detail}
            for s in sorted(signals, key=lambda s: s.at)]


def _kinds(evidence):
    return {e["kind"] for e in evidence or [] if e["kind"] in ("zone_inside", "signal_lost", "prolonged_rest", "cluster_member")}


def _lk_dict(fix, now, source="location"):
    if not fix:
        return None
    return {"lat": fix["lat"], "lon": fix["lon"], "at": iso(fix["t"]), "age_s": max(0, round(now - fix["t"])),
            "accuracy_m": fix.get("acc"), "source": source}


def _run_inline(fn, *a, **kw):
    fn(*a, **kw)


class Engine:
    def __init__(self, store, llm, find_services, clock, submit=None, insights=None, geocode=None):
        self.store, self.llm, self.find_services, self.clock = store, llm, find_services, clock
        self.geocode = geocode                         # (lat, lon) -> {name, kind, ...} | None
        self.insights = insights                       # optional InsightService (AI trip analyst)
        self.submit = submit or _run_inline          # production passes a thread-pool submit

    # ------------------------------------------------------------------ tick
    def tick(self):
        now = self.clock()
        stats = {"trips": 0, "created": 0, "updated": 0, "cleared": 0, "suppressed": 0,
                 "panic_recovered": 0, "errors": 0, "llm_queued": 0}
        self._llm_budget = config.MAX_LLM_BRIEFS_PER_TICK
        self._recover_panics(now, stats)
        zones = self.store.zones(now)
        trips = self.store.active_trips(now)
        by_zone = {}
        if self.insights:
            self.insights.begin_tick()

        for trip in trips:
            stats["trips"] += 1
            try:
                snap = self.store.snapshot(trip, now)
                signals, ctx = extract_signals(snap, zones, now)
                signals = [s for s in signals if s.kind != "panic"]      # panic has its own incident
                if self.insights:
                    signals += self._analyst(trip, snap, zones, now, stats)
                res = score_signals(signals, ctx)
                lk = _lk_dict(ctx.get("last_known"), now)
                self._sync(f"trip:{trip['id']}", "trip_risk", res, signals, ctx, trip, now, stats, lk,
                           base={"user_id": trip["user_id"], "travel_card_id": trip["id"]})
                for s in signals:
                    if s.kind == "zone_inside" and now - s.detail["last_seen"] <= config.CLUSTER_WINDOW_SEC:
                        by_zone.setdefault(s.detail["zone_id"], []).append((trip["id"], s))
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
                log.exception("trip %s failed", trip.get("id"))

        try:
            self._clusters(by_zone, zones, now, stats)
            self._close_orphans(trips, stats, now)
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
            log.exception("tick tail failed")
        return stats

    def _recover_panics(self, now, stats):
        """An SOS whose webhook was lost (Render asleep, secret mismatch, pg_net timeout) still becomes an incident
        within one tick. Runs FIRST and on its own so a failure anywhere else in tick() can never block it."""
        try:
            for alert in self.store.open_panic_alerts(now):
                try:
                    if not self.store.find_incident(f"panic:{alert['id']}"):
                        self.handle_panic(alert["id"])
                        stats["panic_recovered"] += 1
                except Exception:  # noqa: BLE001
                    stats["errors"] += 1
                    log.exception("panic recovery failed for alert %s", alert.get("id"))
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
            log.exception("could not list open panic alerts")

    def _analyst(self, trip, snap, zones, now, stats):
        """Run the AI trip analyst (never raises); its watch/alert stop and loop findings also feed incident scoring."""
        try:
            _, events, _ = self.insights.process(trip, snap, zones, now, stats)
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
            log.exception("insights failed for trip %s", trip.get("id"))
            return []
        return [Signal(e.kind, e.evidence_ids, e.at, e.text, {**e.detail, "severity": e.severity})
                for e in events if e.kind in ("abrupt_stop", "circling", "unplanned_stop") and e.severity != "info"]

    # --------------------------------------------------------- incident sync
    def _sync(self, key, kind, res, signals, ctx, trip, now, stats, last_known_d, base, cluster=None):
        existing = self.store.open_incident(key)
        if res.level is None:
            if existing:
                streak = (existing.get("clear_streak") or 0) + 1
                if streak >= config.AUTO_CLEAR_TICKS:
                    self.store.update_incident(existing["id"], {"status": "resolved", "resolution": "auto_cleared",
                                                                "resolved_at": iso(now), "clear_streak": 0})
                    stats["cleared"] += 1
                else:
                    self.store.update_incident(existing["id"], {"clear_streak": streak})
            return

        evidence = _evidence(signals)
        row = {**base, "dedupe_key": key, "kind": kind, "level": res.level, "score": res.score,
               "reasons": res.reasons, "evidence": evidence,
               "last_lat": last_known_d["lat"] if last_known_d else None,
               "last_lon": last_known_d["lon"] if last_known_d else None,
               "last_fix_at": last_known_d["at"] if last_known_d else None, "clear_streak": 0,
               "peak_level": res.level, "updated_at": iso(now)}
        rank = config.LEVEL_RANK

        if not existing:
            dismissed = self.store.last_dismissed(key, now - config.DISMISS_SUPPRESS_SEC)
            if dismissed and rank[res.level] <= rank[dismissed["level"]]:
                stats["suppressed"] += 1
                return
            inc = self.store.insert_incident(row)
            stats["created"] += 1
            if inc:
                self._brief(inc, kind, res, signals, ctx, trip, now, last_known_d, cluster, force=True, stats=stats)
            return

        escalated = rank[res.level] > rank[existing["level"]]
        kinds_changed = _kinds(evidence) != _kinds(existing.get("evidence"))
        changed = (res.level != existing["level"] or kinds_changed or existing.get("clear_streak")
                   or abs(res.score - existing["score"]) >= config.UPDATE_SCORE_DELTA)
        if not changed:
            return
        patch = {k: row[k] for k in ("level", "score", "reasons", "evidence", "last_lat", "last_lon",
                                     "last_fix_at", "clear_streak", "updated_at")}
        patch["peak_level"] = max(existing.get("peak_level") or res.level, res.level, key=lambda l: rank[l])
        inc = self.store.update_incident(existing["id"], patch) or {**existing, **patch}
        stats["updated"] += 1
        self._brief(inc, kind, res, signals, ctx, trip, now, last_known_d, cluster,
                    force=escalated or kinds_changed, stats=stats)

    # ----------------------------------------------------------------- briefs
    def _brief(self, inc, kind, res, signals, ctx, trip, now, lk, cluster, force, stats, nearby=None, queue_llm=True):
        if config.LEVEL_RANK[res.level] < config.LEVEL_RANK[config.BRIEF_MIN_LEVEL]:
            return
        last = inc.get("brief_generated_at")
        if not force and last and now - parse_ts(last) < config.BRIEF_COOLDOWN_SEC:
            return
        n_contacts = len((trip or {}).get("emergency_contacts") or [])
        packet = build_packet(kind, res.level, res.score, res.reasons, signals, ctx, trip, now,
                              last_known=lk, nearby=nearby, cluster=cluster)
        steps = playbook(kind, res.level, set(res.kinds), n_contacts)
        tmpl = template_brief(packet, steps, now)
        self.store.update_incident(inc["id"], {"brief": tmpl, "brief_source": "template",
                                               "brief_generated_at": tmpl["generated_at"], "brief_error": None})
        if queue_llm and kind == "trip_risk" and res.level == "high" and lk:
            # An automatically detected high-priority situation is treated like an SOS: place name, location-based
            # likely causes, shared contacts and nearby services are attached off the tick thread, then the LLM wording runs.
            self.submit(self._panic_phase2, inc, res, signals, ctx, trip, now, lk, None, None, "trip_risk")
        elif queue_llm:
            self._queue_llm(inc["id"], packet, tmpl, now, stats)

    def _queue_llm(self, incident_id, packet, tmpl, now, stats=None):
        if not self.llm.enabled:
            return
        if getattr(self, "_llm_budget", 1) <= 0:
            return
        self._llm_budget = getattr(self, "_llm_budget", 1) - 1
        if stats is not None:
            stats["llm_queued"] += 1
        self.submit(self._polish, incident_id, packet, tmpl, now)

    def _polish(self, incident_id, packet, tmpl, now):
        guard = {"brief_generated_at": tmpl["generated_at"]}     # never overwrite a newer brief with a stale one
        try:
            raw = self.llm.generate_json(SYSTEM_PROMPT, build_user_message(packet))
            brief = validate_llm_brief(raw, packet, tmpl, now)
            self.store.update_incident(incident_id, {"brief": brief, "brief_source": "llm", "brief_error": None}, where=guard)
        except (LLMUnavailable, BriefRejected) as e:
            self.store.update_incident(incident_id, {"brief_error": f"{type(e).__name__}: {e}"[:200]}, where=guard)
        except Exception as e:  # noqa: BLE001
            log.exception("polish failed")
            self.store.update_incident(incident_id, {"brief_error": f"unexpected: {type(e).__name__}"}, where=guard)

    # -------------------------------------------------------------- clusters
    def _clusters(self, by_zone, zones, now, stats):
        zmap = {z["id"]: z for z in zones}
        seen = set()
        for zid, members in by_zone.items():
            distinct = {cid for cid, _ in members}
            key = f"cluster:{zid}"
            seen.add(key)
            n = len(distinct)
            z = zmap.get(zid, {})
            if n < config.CLUSTER_MIN_TRAVELLERS:
                self._sync(key, "zone_cluster", _Res(0, None), [], {}, None, now, stats, None, base={"zone_id": zid})
                continue
            score = min(40 + 5 * (n - config.CLUSTER_MIN_TRAVELLERS), 75)
            level = "high" if score >= config.HIGH_MIN else "medium"
            sigs = [Signal("cluster_member", [f"zone:{zid}", f"card:{cid}"], s.at,
                           f"Traveller {i + 1} inside the zone since {hhmm(s.at)}.", {})
                    for i, (cid, s) in enumerate(sorted(members, key=lambda m: m[1].at))]
            res = _Res(score, level, [f"+{score}: {n} travellers inside the same verified "
                                      f"{z.get('category', 'hazard')} zone within {config.CLUSTER_WINDOW_SEC // 60} min"],
                       ["cluster_member"])
            cluster = {"travellers": n, "window_min": config.CLUSTER_WINDOW_SEC // 60,
                       "zone": {"category": z.get("category"), "severity": z.get("severity")}}
            self._sync(key, "zone_cluster", res, sigs, {"night": False}, None, now, stats,
                       {"lat": z["lat"], "lon": z["lon"], "at": iso(now), "age_s": 0, "accuracy_m": None, "source": "zone_centre"} if z else None,
                       base={"zone_id": zid}, cluster=cluster)
        for inc in self.store.open_incidents("zone_cluster"):
            if inc["dedupe_key"] not in seen:
                self._sync(inc["dedupe_key"], "zone_cluster", _Res(0, None), [], {}, None, now, stats, None,
                           base={"zone_id": inc.get("zone_id")})

    def _close_orphans(self, trips, stats, now):
        active = {t["id"] for t in trips}
        for inc in self.store.open_incidents("trip_risk"):
            if inc.get("travel_card_id") not in active:
                self.store.update_incident(inc["id"], {"status": "resolved", "resolution": "trip_ended", "resolved_at": iso(now)})
                stats["cleared"] += 1

    # ----------------------------------------------------------------- panic
    def handle_panic(self, alert_id):
        now = self.clock()
        alert = self.store.get_alert(alert_id)
        if not alert or alert.get("type") != "panic":
            return {"skipped": "not_a_panic_alert"}
        key = f"panic:{alert['id']}"
        if self.store.find_incident(key):
            return {"skipped": "already_handled"}

        try:
            trip = self.store.active_trip_for_user(alert["user_id"], now)
        except Exception:  # noqa: BLE001
            log.exception("panic %s: trip lookup failed", alert_id)
            trip = None
        # Prefer the card the SOS was raised on (the phone sends it) over "whatever trip is active today".
        trip = trip or {"id": alert.get("travel_card_id"), "user_id": alert["user_id"], "places": [], "emergency_contacts": []}
        # The incident itself needs only the alert row. Trail, activity state and zones are context: if any of those
        # reads fails (missing table, bad migration, timeout) we still open the incident with what we have.
        try:
            snap = self.store.snapshot(trip, now)
        except Exception:  # noqa: BLE001
            log.exception("panic %s: snapshot failed, continuing without trail", alert_id)
            snap = {"trip": trip, "locations": [], "activity": None, "alerts": [alert], "assessment": None}
        try:
            zones_now = self.store.zones(now)
        except Exception:  # noqa: BLE001
            log.exception("panic %s: zones failed", alert_id)
            zones_now = []
        try:
            signals, ctx = extract_signals(snap, zones_now, now)
        except Exception:  # noqa: BLE001
            log.exception("panic %s: signal extraction failed", alert_id)
            signals, ctx = [], {"night": None, "has_route": False}
        res = score_signals(signals, ctx)                                # -> critical (panic present)

        # Latest reliable location: newest of the alert's own GPS fix and the newest usable tracked fix.
        cands = []
        if alert.get("lat") is not None and alert.get("lon") is not None:
            cands.append({"lat": alert["lat"], "lon": alert["lon"], "t": parse_ts(alert["created_at"]), "acc": None, "src": "alert"})
        tracked = last_known(_signal_fixes(snap.get("locations")))
        if tracked:
            cands.append({**tracked, "src": "location"})
        best = max(cands, key=lambda f: f["t"]) if cands else None
        lk = _lk_dict(best, now, best["src"]) if best else None

        row = {"dedupe_key": key, "kind": "panic", "user_id": alert["user_id"], "travel_card_id": trip["id"],
               "alert_id": alert["id"], "level": "critical", "score": 100, "peak_level": "critical",
               "reasons": res.reasons, "evidence": _evidence(signals),
               "last_lat": lk["lat"] if lk else None, "last_lon": lk["lon"] if lk else None,
               "last_fix_at": lk["at"] if lk else None, "updated_at": iso(now)}
        inc = self.store.insert_incident(row)
        if not inc:
            return {"skipped": "race"}
        # Phase 1: complete template brief, DB only. No external call has happened yet.
        self._extra = {"contacts_shared": _contacts(trip), "emergency_numbers": emergency_numbers(),
                       "sos_source": alert.get("source") or "app"}
        self._brief(inc, "panic", res, signals, ctx, trip, now, lk, None, force=True, stats=None, queue_llm=False)
        self.store.update_incident(inc["id"], {"brief": {**((self.store.find_incident(key) or {}).get("brief") or {}), **self._extra}})
        # Phase 2 (never blocks the above): nearby services, then LLM polish.
        if lk:
            self.submit(self._panic_phase2, inc, res, signals, ctx, trip, now, lk, alert, snap)
        return {"incident_id": inc["id"], "level": "critical"}

    def _panic_phase2(self, inc, res, signals, ctx, trip, now, lk, alert=None, snap=None, kind="panic"):
        def safe(fn, default, *a):
            try:
                return fn(*a) or default
            except Exception:  # noqa: BLE001
                return default
        services = safe(self.find_services, [], lk["lat"], lk["lon"])
        place = safe(self.geocode, None, lk["lat"], lk["lon"]) if self.geocode else None
        weather = safe(weather_now, {}, lk["lat"], lk["lon"])
        zones = safe(self.store.zones, [], now)
        events = [{"kind": s.kind, "detail": s.detail} for s in signals]
        issues = likely_issues(place, ctx.get("night"), zones, lk["lat"], lk["lon"], (snap or {}).get("assessment"), events,
                               weather, (alert or {}).get("source"))
        lk = {**lk, "location_name": (place or {}).get("name"), "place_kind": (place or {}).get("kind")}
        n_contacts = len((trip or {}).get("emergency_contacts") or [])
        packet = build_packet(kind, res.level, res.score, res.reasons, signals, ctx, trip, now,
                              last_known=lk, nearby=services)
        packet["likely_issues"] = [{"id": i["id"], "issue": i["issue"], "why": i["why"]} for i in issues]
        # History the admin needs to judge WHY the SOS was pressed: what else happened in the last 6 hours, and the
        # trip's activity state. (Types/times only: no names or phone numbers ever go into the packet.)
        packet["recent_alerts"] = [{"type": a.get("type"), "severity": a.get("severity"), "status": a.get("status"),
                                    "at": a.get("created_at")} for a in ((snap or {}).get("alerts") or [])
                                   if not (alert and str(a.get("id")) == str(alert.get("id")))][:8]
        act = (snap or {}).get("activity") or {}
        if act.get("current_state"):
            packet["activity"] = {"state": act.get("current_state"), "minutes": act.get("duration_minutes")}
        tmpl = template_brief(packet, playbook(kind, res.level, set(res.kinds), n_contacts), now)
        tmpl.update({"likely_issues": issues, "contacts_shared": _contacts(trip), "emergency_numbers": emergency_numbers(),
                     "sos_source": ((alert or {}).get("source") or "app") if kind == "panic" else "auto", "last_known": lk,
                     "conditions": {"weather": {k: weather.get(k) for k in ("weather_code", "precipitation", "wind_gusts_10m", "apparent_temperature", "visibility")} if weather else None,
                                    "night": ctx.get("night")}})
        tmpl["generated_at"] = iso(now + 0.001)         # newer than phase 1, so it replaces it
        self.store.update_incident(inc["id"], {"brief": tmpl, "brief_source": "template",
                                               "brief_generated_at": tmpl["generated_at"], "brief_error": None})
        self._llm_budget = 1
        self._queue_llm(inc["id"], packet, tmpl, now)


def _contacts(trip):
    """Numbers the traveller shared on their travel card (shown to the admin; never sent to the LLM)."""
    out = []
    for c in (trip or {}).get("emergency_contacts") or []:
        if isinstance(c, dict) and (c.get("phone") or c.get("mobile") or c.get("number")):
            out.append({"role": "emergency_contact", "name": str(c.get("name") or "")[:60] or None,
                        "relation": str(c.get("relation") or c.get("relationship") or "")[:30] or None,
                        "phone": str(c.get("phone") or c.get("mobile") or c.get("number"))[:30]})
    if (trip or {}).get("mobile"):
        out.insert(0, {"role": "traveller", "name": (trip.get("full_name") or "")[:60] or None, "relation": None, "phone": str(trip["mobile"])[:30]})
    for c in (trip or {}).get("companions") or []:
        if isinstance(c, dict) and (c.get("phone") or c.get("mobile")):
            out.append({"role": "companion", "name": str(c.get("name") or "")[:60] or None, "relation": None,
                        "phone": str(c.get("phone") or c.get("mobile"))[:30]})
    return out


class _Res:
    def __init__(self, score, level, reasons=None, kinds=None):
        self.score, self.level, self.reasons, self.kinds = score, level, reasons or [], kinds or []
