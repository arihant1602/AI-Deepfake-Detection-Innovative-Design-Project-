import { NavLink } from "react-router-dom";
import "./Header.css";

const LINKS = [
  { to: "/", label: "Live", end: true },
  { to: "/file", label: "Analyze a file", end: false },
  { to: "/device", label: "Device", end: false },
];

export function Header() {
  return (
    <header className="header">
      <NavLink to="/" className="header__mark" aria-label="Grain, live video verification">
        Grain
      </NavLink>
      <nav aria-label="Pages">
        <ul className="header__nav">
          {LINKS.map((l) => (
            <li key={l.to}>
              <NavLink to={l.to} end={l.end} className="header__link">
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <p className="header__tag">Is this video coming from a real camera?</p>
    </header>
  );
}
