import { useMemo, useState } from "react";
import { ChartHost, SERIES, Tooltip, compact, useTooltip } from "./primitives";

export interface LinePoint {
  label: string;
  x: number;
  y: number;
}

/**
 * Single-series line with a crosshair.
 *
 * 2px stroke with round joins, an end marker of at least 8px carrying a 2px
 * surface ring so it stays legible where it crosses the line, and a 10% area
 * wash beneath. Gridlines are hairline and recessive. The crosshair snaps to
 * the nearest point rather than tracking the raw pointer, so reading a value is
 * exact.
 *
 * A log y-scale is offered because some of these series (technique frequency
 * against rank) span orders of magnitude and are unreadable linearly.
 */
export function LineChart({
  points,
  yLabel,
  logY = false,
  height = 220,
}: {
  points: LinePoint[];
  yLabel: string;
  logY?: boolean;
  height?: number;
}) {
  const { tip, show, hide } = useTooltip();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { top: 12, right: 16, bottom: 26, left: 46 };
  const width = 720;

  const scaled = useMemo(() => {
    if (!points.length) return null;
    const ys = points.map((p) => p.y);
    const toY = (value: number) => (logY ? Math.log10(Math.max(value, 0.1)) : value);
    const minY = logY ? Math.min(...ys.map(toY)) : 0;
    const maxY = Math.max(...ys.map(toY), minY + 1);
    const minX = Math.min(...points.map((p) => p.x));
    const maxX = Math.max(...points.map((p) => p.x));
    const px = (x: number) =>
      pad.left + ((x - minX) / Math.max(maxX - minX, 1)) * (width - pad.left - pad.right);
    const py = (y: number) =>
      height - pad.bottom - ((toY(y) - minY) / Math.max(maxY - minY, 1e-9)) * (height - pad.top - pad.bottom);
    const coords = points.map((p) => ({ ...p, cx: px(p.x), cy: py(p.y) }));
    const ticks = Array.from({ length: 4 }, (_, i) => {
      const value = minY + ((maxY - minY) * i) / 3;
      return { y: py(logY ? 10 ** value : value), label: compact(logY ? 10 ** value : value) };
    });
    return { coords, ticks };
  }, [points, logY, height]);

  if (!scaled) return null;
  const { coords, ticks } = scaled;
  const line = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.cx},${c.cy}`).join(" ");
  const area = `${line} L${coords.at(-1)!.cx},${height - pad.bottom} L${coords[0]!.cx},${height - pad.bottom} Z`;
  const last = coords.at(-1)!;
  const active = hover !== null ? coords[hover] : null;

  function onMove(event: React.MouseEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * width;
    let nearest = 0;
    for (let i = 1; i < coords.length; i++) {
      if (Math.abs(coords[i]!.cx - x) < Math.abs(coords[nearest]!.cx - x)) nearest = i;
    }
    setHover(nearest);
    const point = coords[nearest]!;
    show(event, (
      <>
        <span className="font-mono">{point.label}</span>
        <br />
        {yLabel}: <strong>{point.y.toLocaleString()}</strong>
      </>
    ));
  }

  return (
    <ChartHost>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label={`${yLabel} across ${points.length} points, from ${points[0]?.label} to ${points.at(-1)?.label}`}
        onMouseMove={onMove}
        onMouseLeave={() => { setHover(null); hide(); }}
      >
        {ticks.map((tick, i) => (
          <g key={i}>
            <line
              x1={pad.left} x2={width - pad.right} y1={tick.y} y2={tick.y}
              stroke="var(--ac-hairline)" strokeWidth={1}
            />
            <text
              x={pad.left - 6} y={tick.y + 3} textAnchor="end"
              className="fill-[var(--ac-ink-muted)] text-[10px]"
            >
              {tick.label}
            </text>
          </g>
        ))}
        <path d={area} fill={SERIES[0]} opacity={0.1} />
        <path
          d={line} fill="none" stroke={SERIES[0]}
          strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
        />
        {active && (
          <line
            x1={active.cx} x2={active.cx} y1={pad.top} y2={height - pad.bottom}
            stroke="var(--ac-axis)" strokeWidth={1}
          />
        )}
        {/* End marker: >= 8px diameter with a 2px surface ring */}
        <circle cx={last.cx} cy={last.cy} r={4} fill={SERIES[0]}
                stroke="var(--ac-surface)" strokeWidth={2} />
        {active && (
          <circle cx={active.cx} cy={active.cy} r={4} fill={SERIES[0]}
                  stroke="var(--ac-surface)" strokeWidth={2} />
        )}
        <line
          x1={pad.left} x2={width - pad.right}
          y1={height - pad.bottom} y2={height - pad.bottom}
          stroke="var(--ac-axis)" strokeWidth={1}
        />
        <text x={pad.left} y={height - 8} className="fill-[var(--ac-ink-muted)] text-[10px]">
          {points[0]?.label}
        </text>
        <text
          x={width - pad.right} y={height - 8} textAnchor="end"
          className="fill-[var(--ac-ink-muted)] text-[10px]"
        >
          {points.at(-1)?.label}
        </text>
      </svg>
      {logY && (
        <p className="mt-1 text-[11px] text-ink-muted">
          Logarithmic vertical scale; this series spans several orders of magnitude.
        </p>
      )}
      <Tooltip tip={tip} />
    </ChartHost>
  );
}
