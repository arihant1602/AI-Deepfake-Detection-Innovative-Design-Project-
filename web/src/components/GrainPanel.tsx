import type { ReactNode } from "react";
import "./GrainPanel.css";

/** The blue, crumpled-paper block from the reference. Children sit on top of it. */
export function GrainPanel({ children, className = "" }: { children?: ReactNode; className?: string }) {
  return <div className={`grain-panel ${className}`}>{children}</div>;
}
