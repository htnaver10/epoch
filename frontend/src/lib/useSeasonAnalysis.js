import { useMemo } from "react";
import {
  yearTotals, harmonizedSeries, assessSeason, validateBridge, fitBridge, weeklyCalendar,
  trend, splicedTrend, seasonCurves, quantile, median,
} from "./seasons";

/**
 * Everything the interface shows about a region's record, computed once per region and
 * period. A region is a built-in sector or an area of interest; a period is a span of the
 * season's days { from, to }, or null for the whole season.
 */
export function useSeasonAnalysis(seasons, region, period = null) {
  const from = period?.from ?? null, to = period?.to ?? null;
  return useMemo(() => {
    if (!seasons || !seasons.series[region]) return null;
    const span = from === null ? null : { from, to };
    const totals = span ? yearTotals(seasons, region, to, from) : yearTotals(seasons, region);
    const modis = totals.filter((t) => t.modis !== null).map((t) => t.modis);
    const years = totals.map((t) => t.year);
    return {
      totals,
      years,
      series: harmonizedSeries(totals),
      verdicts: new Map(years.map((y) => [y, assessSeason(seasons, region, y, span)])),
      bridge: fitBridge(totals),
      validation: validateBridge(totals),
      calendar: weeklyCalendar(seasons, region),
      trend: trend(totals),
      spliced: splicedTrend(totals),
      band: modis.length ? [quantile(modis, 0.1), quantile(modis, 0.9)] : null,
      base: median(modis),
      first: years[0],
      last: years[years.length - 1],
      viirsFirst: totals.find((t) => t.viirsDays > 0)?.year ?? null,
    };
  }, [seasons, region, from, to]);
}

/** Cumulative curves for one season against the envelope of the others. */
export function useSeasonCurves(seasons, region, year) {
  return useMemo(() => (seasons && year && seasons.series[region] ? seasonCurves(seasons, region, year) : null), [seasons, region, year]);
}
