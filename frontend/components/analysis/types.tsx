export const TABS = [
  { id: "rally",  label: "回合分析" },
  { id: "player", label: "球員統計" },
  { id: "depth",  label: "站位分析" },
  { id: "speed",  label: "速度統計" },
  { id: "court",  label: "落點圖"   },
] as const;

export type TabId = (typeof TABS)[number]["id"];

export const SHOT_TYPE_LABEL: Record<string, string> = {
  serve: "發球", overhead: "高壓", swing: "揮拍", unknown: "未知",
};

export function EmptyState({ message }: { message: string }) {
  return <div className="px-3 py-3.5 text-base text-gray-500 dark:text-gray-400">{message}</div>;
}
