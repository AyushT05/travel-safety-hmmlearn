import React from "react";

const LABELS = {
  LIVE: "Live location",
  DELAYED: "Location delayed",
  STALE: "Location stale",
};

// Use inside a traveller/device card: <LocationFreshnessBadge device={device} />
export default function LocationFreshnessBadge({ device }) {
  const status = device?.freshnessStatus || "STALE";
  const age = Math.max(0, Number(device?.locationAgeSeconds || 0));
  const ageLabel = age < 60 ? `${age}s ago` : `${Math.floor(age / 60)}m ago`;

  const styles = {
    LIVE: { background: "#dcfce7", color: "#166534", dot: "#16a34a" },
    DELAYED: { background: "#fef3c7", color: "#92400e", dot: "#d97706" },
    STALE: { background: "#fee2e2", color: "#991b1b", dot: "#dc2626" },
  }[status] || { background: "#e5e7eb", color: "#374151", dot: "#6b7280" };

  return (
    <span
      role="status"
      aria-label={`${LABELS[status] || LABELS.STALE}; last update ${ageLabel}`}
      title={`Last GPS update ${ageLabel}`}
      style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        padding: "4px 8px", borderRadius: 999, fontSize: 12,
        fontWeight: 600, background: styles.background, color: styles.color,
      }}
    >
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: styles.dot }} />
      {LABELS[status] || LABELS.STALE} · {ageLabel}
    </span>
  );
}
