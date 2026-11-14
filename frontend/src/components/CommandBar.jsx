import { useEffect, useMemo, useRef, useState } from "react";
import { interpret } from "../lib/places";
import Icon from "./Icon";

const SHOWN = 7;

/**
 * The command line. Type a place, coordinates, a season, a day or a sector; Enter runs the
 * first reading of it. What was run stays on the line, so a viewer can see it, and the up
 * arrow recalls earlier commands. "/" focuses it from anywhere.
 */
export default function CommandBar({ context, onRun }) {
  const [query, setQuery] = useState("");
  const [ran, setRan] = useState(false); // the text on the line is the last command, not a draft
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const history = useRef([]);
  const cursor = useRef(-1);
  const input = useRef(null);
  // one more than is shown, to know whether the list was cut
  const found = useMemo(() => (ran ? [] : interpret(query, context, SHOWN + 1)), [query, context, ran]);
  const results = found.slice(0, SHOWN);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      e.preventDefault();
      input.current?.focus();
      input.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const run = (action) => {
    if (!action) return;
    onRun(action);
    const text = action.label.toUpperCase();
    history.current = [text, ...history.current.filter((h) => h !== text)].slice(0, 20);
    cursor.current = -1;
    setQuery(text);
    setRan(true);
    setOpen(false);
    input.current?.blur();
  };
  const recall = (step) => {
    const list = history.current;
    if (!list.length) return;
    cursor.current = Math.max(-1, Math.min(list.length - 1, cursor.current + step));
    setQuery(cursor.current < 0 ? "" : list[cursor.current]);
    setRan(false);
    setActive(0);
    setOpen(true);
  };
  const onKeyDown = (e) => {
    const showing = open && results.length > 0;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      // with suggestions open the arrows walk them; on an empty or recalled line they walk history
      if (showing && query && cursor.current < 0) setActive((a) => (a + (e.key === "ArrowDown" ? 1 : results.length - 1)) % results.length);
      else recall(e.key === "ArrowUp" ? 1 : -1);
    } else if (e.key === "Enter") run(results[Math.min(active, results.length - 1)]);
    else if (e.key === "Escape") {
      setQuery("");
      setRan(false);
      input.current?.blur();
    }
  };
  const draft = query.trim();
  // numbers still being typed could become a year or coordinates: say how, not "no match"
  const numeric = /^[\d\s.,;°+\-nsew/]*$/i.test(draft) && /\d/.test(draft);
  const showing = open && !ran && draft.length > 0;
  const years = context.years;
  const empty =
    results.length > 0
      ? null
      : numeric
        ? `Coordinates as “58.4 -112.1”${years.length ? `, or a season from ${years[0]} to ${years[years.length - 1]}` : ""}.`
        : draft.length >= 2
          ? "Not in the place list. Coordinates work anywhere: “58.4 -112.1”."
          : null;
  return (
    <div className={`command${ran ? " ran" : ""}`}>
      <span className="prompt" aria-hidden="true"><Icon name="search" size="1rem" /></span>
      <input
        ref={input}
        type="text"
        value={query}
        spellCheck="false"
        autoComplete="off"
        placeholder="Place, lat lon or year"
        aria-label="Search: place, coordinates, season, day or region"
        role="combobox"
        aria-expanded={showing}
        aria-controls="command-results"
        onChange={(e) => { setQuery(e.target.value); setRan(false); cursor.current = -1; setActive(0); setOpen(true); }}
        onFocus={(e) => { setOpen(true); e.target.select(); }}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      <kbd>/</kbd>
      {showing && (results.length > 0 || empty) && (
        <ul id="command-results" className="command-results" role="listbox">
          {empty && <li className="none">{empty}</li>}
          {results.map((r, i) => (
            <li key={`${r.type}-${r.label}`} role="option" aria-selected={i === active}>
              {/* mousedown, not click: the input's blur would close the list first */}
              <button className={i === active ? "on" : undefined} onMouseDown={(e) => { e.preventDefault(); run(r); }} onMouseEnter={() => setActive(i)}>
                <span>{r.label}</span>
                <em>{r.hint}</em>
              </button>
            </li>
          ))}
          {found.length > SHOWN && <li className="none">More matches: keep typing to narrow them.</li>}
        </ul>
      )}
    </div>
  );
}
