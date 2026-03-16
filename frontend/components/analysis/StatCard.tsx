"use client";

export default function StatCard({
  label,
  value,
  hint,
  color,
}: {
  label: string;
  value: any;
  hint?: string;
  color?: string;
}) {
  return (
    <div className="stat-card" style={color ? { borderLeft: `3px solid ${color}` } : undefined}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? "—"}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}
