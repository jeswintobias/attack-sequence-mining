import { useState } from "react";

/**
 * The table view.
 *
 * Present for every chart, not as a fallback but as a peer. It discharges the
 * relief obligation wherever a mark sits below 3:1 against the surface, it is
 * the accessible path to any value, and it is how a reader checks a number they
 * do not believe.
 */
export function DataTable({
  columns,
  records,
  index,
  indexLabel = "",
  pageSize = 25,
}: {
  columns: string[];
  records: Record<string, string>[];
  index?: string[] | null;
  indexLabel?: string;
  pageSize?: number;
}) {
  const [limit, setLimit] = useState(pageSize);
  const shown = records.slice(0, limit);

  return (
    <div>
      <div className="max-h-[26rem] overflow-auto rounded-lg border border-hairline">
        <table className="w-full border-collapse text-[12px]">
          <thead className="sticky top-0 bg-surface">
            <tr className="border-b border-hairline">
              {index && (
                <th scope="col" className="px-3 py-2 text-left font-medium text-ink-secondary">
                  {indexLabel || "Row"}
                </th>
              )}
              {columns.map((column) => (
                <th
                  key={column}
                  scope="col"
                  className="px-3 py-2 text-right font-medium text-ink-secondary"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((record, rowIndex) => (
              <tr key={rowIndex} className="border-b border-hairline last:border-0 hover:bg-plane">
                {index && (
                  <th
                    scope="row"
                    className="px-3 py-1.5 text-left font-mono font-normal text-ink"
                  >
                    {index[rowIndex]}
                  </th>
                )}
                {columns.map((column) => (
                  <td key={column} className="px-3 py-1.5 text-right font-mono text-ink-secondary">
                    {record[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {limit < records.length && (
        <button
          type="button"
          onClick={() => setLimit((value) => value + pageSize * 2)}
          className="mt-2 rounded-md border border-hairline px-2.5 py-1 text-[12px] text-ink-secondary hover:bg-plane hover:text-ink"
        >
          Show more ({records.length - limit} remaining)
        </button>
      )}
    </div>
  );
}
