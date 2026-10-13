import { useEffect, useRef, useState } from "react";
import { supabase } from "../lib/supabase";

const FRESHNESS_REFRESH_MS = 5000;

function getFreshness(recordedAt) {
  const ageSeconds = Math.max(0, Math.floor((Date.now() - new Date(recordedAt).getTime()) / 1000));
  const freshnessStatus = ageSeconds <= 20 ? "LIVE" : ageSeconds <= 90 ? "DELAYED" : "STALE";
  return { locationAgeSeconds: ageSeconds, freshnessStatus };
}

function toDevice(row, previous = null) {
  const timestamp = new Date(row.recorded_at).getTime();
  return {
    lastLatlng: [row.lat, row.lon],
    accuracy: row.accuracy,
    speed: row.speed,
    timestamp: timestamp / 1000,
    updates: (previous?.updates || 0) + 1,
    travelCardId: row.travel_card_id || null,
    ...getFreshness(row.recorded_at),
  };
}

// Drop-in replacement for the previously shared hook. It keeps the existing
// `devices` shape/API and adds freshnessStatus + locationAgeSeconds per device.
export default function useDevices() {
  const [devices, setDevices] = useState({});
  const devicesRef = useRef({});

  useEffect(() => {
    let channel;
    let cancelled = false;

    function publish(next) {
      devicesRef.current = next;
      setDevices(next);
    }

    function applyRow(row) {
      if (!row?.user_id || !row?.recorded_at) return;
      const prev = devicesRef.current[row.user_id];
      publish({ ...devicesRef.current, [row.user_id]: toDevice(row, prev) });
    }

    async function bootstrap() {
      const { data, error } = await supabase
        .from("latest_locations")
        .select("user_id, lat, lon, accuracy, speed, recorded_at, travel_card_id");

      if (!cancelled && !error && data) {
        const initial = {};
        for (const row of data) {
          initial[row.user_id] = toDevice(row, null);
        }
        publish(initial);
      } else if (error) {
        console.error("Failed to load latest_locations:", error);
      }

      if (cancelled) return;
      channel = supabase
        .channel("locations-live")
        .on("postgres_changes", {
          event: "INSERT", schema: "public", table: "locations",
        }, (payload) => applyRow(payload.new))
        .subscribe();
    }

    bootstrap();

    // Freshness changes even when no Realtime event arrives. Recompute ages
    // locally so a device transitions to DELAYED/STALE without new GPS data.
    const freshnessTimer = setInterval(() => {
      if (cancelled) return;
      const current = devicesRef.current;
      const next = {};
      let changed = false;
      for (const [userId, device] of Object.entries(current)) {
        const recordedAt = new Date(device.timestamp * 1000).toISOString();
        const freshness = getFreshness(recordedAt);
        next[userId] = { ...device, ...freshness };
        if (freshness.freshnessStatus !== device.freshnessStatus ||
            freshness.locationAgeSeconds !== device.locationAgeSeconds) changed = true;
      }
      if (changed) publish(next);
    }, FRESHNESS_REFRESH_MS);

    return () => {
      cancelled = true;
      clearInterval(freshnessTimer);
      if (channel) supabase.removeChannel(channel);
    };
  }, []);

  return {
    devices,
    clearTrails: () => {},
    toggleFollow: () => true,
    fitAll: () => {},
  };
}
