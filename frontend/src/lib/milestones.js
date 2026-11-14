import { seasonDate, medianCrossing } from "./seasons.js";
import { formatDate } from "./chart.js";

/** The day a season's running total passed what a median season totals by the end of August. */
export function medianReached(seasons, curves, year) {
  if (!curves.target || curves.reference < 3) return null;
  const i = medianCrossing(curves);
  if (i === null) return { v: "not reached", unit: "" };
  return { v: formatDate(seasonDate(seasons, year, i)), unit: `day ${i + 1} of ${curves.days}` };
}
