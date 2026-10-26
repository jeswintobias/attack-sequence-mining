import { useEffect, useState } from "react";
import { Compass, RotateCcw } from "lucide-react";
import { api } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import type { PredictResponse, ReconstructResponse, StageResponse } from "../lib/types";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { EmptyState, ErrorState, LoadingState } from "../components/states/States";
import { Skeleton } from "../components/ui/Skeleton";
import { TechniquePicker } from "../components/predict/TechniquePicker";
import { CandidateList } from "../components/predict/CandidateList";
import { KillChainStepper } from "../components/predict/KillChainStepper";
import { ContextPanel } from "../components/predict/ContextPanel";
import { PathReconstruction } from "../components/predict/PathReconstruction";
import { Caveats } from "../components/predict/Caveats";

/**
 * A worked example is loaded on first visit so the application demonstrates
 * itself without anyone having to type an ATT&CK identifier from memory.
 * Valid Accounts followed by Command and Scripting Interpreter is a realistic
 * opening pair and has enough training support to exercise the learned model
 * rather than the fallback.
 */
const WORKED_EXAMPLE = ["T1078", "T1059"];

export function PredictView() {
  const [observed, setObserved] = useState<string[]>(WORKED_EXAMPLE);
  const catalogue = useAsync(() => api.catalogue(), []);

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [stage, setStage] = useState<StageResponse | null>(null);
  const [paths, setPaths] = useState<ReconstructResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const key = observed.join(",");

  useEffect(() => {
    if (!observed.length) {
      setPrediction(null);
      setStage(null);
      setPaths(null);
      return;
    }
    let active = true;
    setBusy(true);
    setError(null);
    // Reconstruction is capped at eight techniques by the contract, so it is
    // requested only when the input is within range.
    const canReconstruct = observed.length >= 2 && observed.length <= 8;
    Promise.all([
      api.predict(observed, 5),
      api.stage(observed),
      canReconstruct ? api.reconstruct(observed, 3) : Promise.resolve(null),
    ])
      .then(([predicted, located, ordered]) => {
        if (!active) return;
        setPrediction(predicted);
        setStage(located);
        setPaths(ordered);
      })
      .catch((cause) => active && setError(cause))
      .finally(() => active && setBusy(false));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (catalogue.loading) return <LoadingState label="Loading technique catalogue" />;
  if (catalogue.error) return <ErrorState error={catalogue.error} />;

  const isExample = key === WORKED_EXAMPLE.join(",");

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader
          title="What has been observed?"
          description="Add techniques in the order they were seen. Search by name or identifier."
          aside={
            !isExample && (
              <button
                type="button"
                onClick={() => setObserved(WORKED_EXAMPLE)}
                className="inline-flex items-center gap-1.5 rounded-md border border-hairline px-2.5 py-1.5 text-[12px] font-medium text-ink-secondary hover:bg-plane hover:text-ink"
              >
                <RotateCcw aria-hidden className="size-3.5" />
                Reset to example
              </button>
            )
          }
        />
        <CardBody>
          <TechniquePicker
            catalogue={catalogue.data ?? []}
            observed={observed}
            onChange={setObserved}
          />
        </CardBody>
      </Card>

      {error != null && <ErrorState error={error} />}

      {!observed.length && (
        <Card>
          <EmptyState title="Add a technique to begin" icon={Compass}>
            Start with something you have seen — PowerShell, Valid Accounts, or any of
            the {catalogue.data?.length ?? 0} techniques the model knows.
          </EmptyState>
        </Card>
      )}

      {observed.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="space-y-5">
            <Card>
              <CardHeader
                title="What may come next"
                description="Ranked shortlist. Read all five, not just the first."
              />
              {busy && !prediction ? (
                <CardBody className="space-y-3">
                  {[0, 1, 2, 3, 4].map((row) => (
                    <Skeleton key={row} className="h-12" />
                  ))}
                </CardBody>
              ) : prediction ? (
                <div className={busy ? "opacity-60 transition-opacity" : undefined}>
                  <CandidateList candidates={prediction.candidates} />
                </div>
              ) : null}
            </Card>

          </div>

          <div className="space-y-5">
            {prediction && (
              <Card>
                <CardHeader title="Confidence" />
                <CardBody>
                  <ContextPanel context={prediction.context} />
                </CardBody>
              </Card>
            )}

            {paths && paths.orderings.length > 1 && (
              <Card>
                <CardHeader
                  title="Possible orderings"
                  description="If the sequence of what you saw is uncertain."
                />
                <CardBody>
                  <PathReconstruction result={paths} />
                </CardBody>
              </Card>
            )}

            {prediction && <Caveats caveats={prediction.caveats} />}
          </div>
        </div>
      )}

      {stage && observed.length > 0 && (
        <Card>
          <CardHeader
            title="Kill chain"
            description="Filled nodes are tactics the prefix has touched; bars show where predicted activity lands."
          />
          <CardBody>
            <KillChainStepper stage={stage} />
          </CardBody>
        </Card>
      )}
    </div>
  );
}
