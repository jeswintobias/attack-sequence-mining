import { Compass } from "lucide-react";
import { Card, CardBody, CardHeader } from "../components/ui/Card";
import { EmptyState } from "../components/states/States";

/** Filled in Stage 4: technique picker, ranked candidates, kill-chain stepper. */
export function PredictView() {
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
      <Card>
        <CardHeader
          title="Forecast"
          description="Enter the techniques observed so far to see what may follow."
        />
        <CardBody className="p-0">
          <EmptyState title="Technique picker arrives in Stage 4" icon={Compass}>
            The API behind this view is live: nine endpoints covering ranked
            candidates, kill-chain localisation and path reconstruction. Browse
            them at <code className="font-mono">/docs</code>.
          </EmptyState>
        </CardBody>
      </Card>
      <Card>
        <CardHeader title="Kill chain" description="Where the campaign appears to be." />
        <CardBody>
          <p className="text-[13px] text-ink-secondary">
            The fourteen-step stepper renders here once the picker exists.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}
