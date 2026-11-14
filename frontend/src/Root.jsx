import { lazy, Suspense, useCallback, useState } from "react";
import App from "./App.jsx";

// Response scenarios are a second mode of the same instrument. It is loaded only when opened.
const Ops = lazy(() => import("./ops/Ops.jsx"));

const readMode = () => (new URLSearchParams(window.location.search).get("mode") === "ops" || new URLSearchParams(window.location.search).get("mode") === "scenarios" ? "ops" : "analysis");

export default function Root() {
  const [mode, setMode] = useState(readMode);
  const [initial] = useState(() => {
    const q = new URLSearchParams(window.location.search);
    return { theatre: q.get("theatre") || "", paused: q.get("paused") === "1", speed: Number(q.get("speed")) || undefined, advance: Number(q.get("advance")) || 0, fire: q.get("fire") || "" };
  });
  const change = useCallback((next) => {
    // each mode keeps its own address: analysis writes its view to the query string, ops its theatre
    window.history.replaceState(null, "", next === "ops" ? "?mode=scenarios" : window.location.pathname);
    setMode(next);
  }, []);
  if (mode === "ops")
    return (
      <Suspense fallback={<div className="console" />}>
        <Ops onMode={change} initial={initial} />
      </Suspense>
    );
  return <App onMode={change} />;
}

