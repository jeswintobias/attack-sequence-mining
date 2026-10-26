import { useState } from "react";
import { Header, TABS, type TabId } from "./components/Header";
import { ModelStrip } from "./components/ModelStrip";
import { PredictView } from "./views/PredictView";
import { ExploreView } from "./views/ExploreView";

export default function App() {
  const [tab, setTab] = useState<TabId>("predict");

  return (
    <div className="min-h-dvh bg-plane">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-30 focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-[13px] focus:text-accent-ink"
      >
        Skip to content
      </a>
      <Header active={tab} onChange={setTab} />
      <ModelStrip />
      <main id="main" className="mx-auto max-w-6xl px-6 py-6">
        {TABS.map((entry) => (
          <div
            key={entry.id}
            role="tabpanel"
            id={`panel-${entry.id}`}
            aria-labelledby={`tab-${entry.id}`}
            hidden={entry.id !== tab}
          >
            {entry.id === tab &&
              (entry.id === "predict" ? <PredictView /> : <ExploreView />)}
          </div>
        ))}
      </main>
      <footer className="mx-auto max-w-6xl px-6 pb-8 text-[12px] leading-relaxed text-ink-muted">
        Predictions rank the next technique in <em>reported</em> order, which is not
        established to be execution order. Several next steps are usually plausible:
        read the shortlist, not the top answer.
      </footer>
    </div>
  );
}
