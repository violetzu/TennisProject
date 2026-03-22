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
      <div className="text-base text-gray-500 dark:text-gray-400 mb-0.5">{label}</div>
      <div className="text-xl font-bold">{value ?? "—"}</div>
      {hint && <div className="text-base text-gray-400 dark:text-gray-500 mt-0.5">{hint}</div>}
    </div>
  );
}
