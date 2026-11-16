import { lazy, Suspense, useCallback, useState } from "react";
import App from "./App.jsx";

// Response scenarios are a second mode of the same instrument. It is loaded only when opened.
const Ops = lazy(() => import("./ops/Ops.jsx"));
const MLViewer = lazy(() => import("./ml/MLViewer.jsx"));

const readMode = () => {
  const mode = new URLSearchParams(window.location.search).get("mode");
  return mode === "ml" ? "ml" : mode === "ops" || mode === "scenarios" ? "ops" : "analysis";
};

export default function Root() {
  const [mode, setMode] = useState(readMode);
  const [initial] = useState(() => {
    const q = new URLSearchParams(window.location.search);
    return { theatre: q.get("theatre") || "", paused: q.get("paused") === "1", speed: Number(q.get("speed")) || undefined, advance: Number(q.get("advance")) || 0, fire: q.get("fire") || "" };
  });
  const change = useCallback((next) => {
    // each mode keeps its own address: analysis writes its view to the query string, ops its theatre
    window.history.replaceState(null, "", next === "ops" ? "?mode=scenarios" : next === "ml" ? "?mode=ml" : window.location.pathname);
    setMode(next);
  }, []);
  if (mode === "ml") return <Suspense fallback={<div className="console" role="status">Opening MLP research…</div>}><MLViewer onMode={change} /></Suspense>;
  if (mode === "ops")
    return (
      <Suspense fallback={<div className="console" />}>
        <Ops onMode={change} initial={initial} />
      </Suspense>
    );
  return <App onMode={change} />;
}

