import type { Status } from "../lib/gates";
import "./StackWord.css";

/** The verdict as one word set three times: solid, outlined, solid, after the reference's BLOOM stack. */
export function StackWord({ word, status, size = "hero" }: { word: string; status: Status; size?: "hero" | "page" }) {
  return (
    <h1 className={`stack stack--${status} stack--${size}`} aria-label={word} key={word}>
      <span aria-hidden="true">{word}</span>
      <span aria-hidden="true" className="stack__outline">
        {word}
      </span>
      <span aria-hidden="true">{word}</span>
    </h1>
  );
}
