import { useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { NavLink } from "react-router-dom";
import { applyTheme, currentTheme, storedTheme, type Theme } from "../lib/theme";
import "./Header.css";

const LINKS = [
  { to: "/", label: "Live", end: true },
  { to: "/file", label: "File", end: false },
  { to: "/device", label: "Device", end: false },
];

const SLOT_ID = "topbar-controls";

/** Lets a page put its own controls (camera, start/stop, status) into the top bar. */
export function TopBarControls({ children }: { children: ReactNode }) {
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  useEffect(() => setSlot(document.getElementById(SLOT_ID)), []);
  return slot ? createPortal(children, slot) : null;
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(currentTheme);
  useEffect(() => applyTheme(storedTheme()), []);
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      className="btn btn--ghost topbar__theme"
      onClick={() => {
        applyTheme(next);
        setTheme(next);
      }}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor" />
      </svg>
    </button>
  );
}

export function Header() {
  return (
    <header className="topbar">
      <NavLink to="/" className="topbar__mark" aria-label="Grain, live video verification">
        Grain
      </NavLink>
      <nav aria-label="Pages">
        <ul className="topbar__nav">
          {LINKS.map((l) => (
            <li key={l.to}>
              <NavLink to={l.to} end={l.end} className="topbar__link">
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <div className="topbar__controls" id={SLOT_ID} />
      <ThemeToggle />
    </header>
  );
}
