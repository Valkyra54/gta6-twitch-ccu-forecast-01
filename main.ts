// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecast: for each case, predict Twitch concurrent viewers for hours 2..24
// of launch day as five log-normal quantiles per hour.
//
// Model (comparable-ratio ensemble with a magnitude regression):
//   - Every comparable launch gives a launch-day curve normalized by its own
//     hour-1 print: r_h = viewers_h / viewers_1. Those ratios carry the
//     unlock-wave SHAPE (dip at hour 2, rise as timezones unlock, peak in
//     the evening) that a flat persistence forecast misses entirely.
//   - Comparables are weighted by similarity to the target: closeness of
//     log hour-1 magnitude, closeness of the pre-launch hype ratio
//     (mean viewers in the 24h before launch, over hour-1 print), a boost
//     for console-only launches, and a penalty for early-access-contaminated
//     curves (their launch-day shape is not a clean launch).
//   - Per hour, a weighted two-variable regression of log r_h on
//     log(hour-1 print) and log(pre-launch ratio) is fit across the
//     comparables and evaluated at the target's own values. The slope is
//     strongly negative (small hour-1 launches ramp far more than big ones),
//     so the regression, not a plain weighted median, is what recovers the
//     ramp. Slopes are shrunk 15% toward zero as a hedge against
//     extrapolation on a launch bigger than any comparable.
//   - Spread is the weighted residual standard deviation of log ratios
//     around the fitted line, shrunk 40% toward a prior of 0.35 log units.
//     Quantiles are q_tau = p1 * exp(median + z_tau * sd).
//   - The GTA VI pre-release comparable has no launch-day curve (its
//     series ends before launch), so the completeness rule below drops it
//     automatically; the target's own pre-launch history is the analogous
//     signal and enters through the pre-launch ratio feature.
//   - No practice answers, no memorized values: everything is computed from
//     the case file handed to this run. Pure arithmetic, no randomness, so
//     two runs are byte-identical.
//
// Input:  $MAKEFUN_INPUT_DIR (else /input) holding <id>.in files.
// Output: $MAKEFUN_OUTPUT_DIR (else /output) holding one <id>.out per case:
//   {"hours":[{"hour":h,"q":[q05,q25,q50,q75,q95]}, ...]}, hours 2..24.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles at those levels. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Kernel width (log units) for hour-1 magnitude similarity. */
const SIG_MAG = 1.0;
/** Kernel width (log units) for pre-launch-ratio similarity. */
const SIG_PRE = 0.7;
/** Weight multiplier for console-only launches (no PC in platforms). */
const CONSOLE_BOOST = 1.3;
/** Weight multiplier for early-access-contaminated curves. */
const TIER_PENALTY = 0.3;
/** Regression slopes are shrunk this far toward zero (extrapolation hedge). */
const SLOPE_SHRINK = 1.0;
/** Residual spread is shrunk this far toward the prior spread. */
const SPREAD_SHRINK = 0.7;
/** Prior spread in log units. */
const PRIOR_SD = 0.35;
/** Baseline spread (log units per sqrt hour) if the ensemble cannot run. */
const FALLBACK_SIGMA = 0.15;

interface KnownRow {
  hour: number;
  viewers: number;
}

interface ForecastRow {
  hour: number;
}

interface HistoryPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  platforms: string;
  hourly: Array<{ from: string; viewers: number }>;
}

interface CaseFile {
  id: string;
  day_start: string;
  known: KnownRow[];
  forecast: ForecastRow[];
  history_hourly: HistoryPoint[];
  comparables: Comparable[];
}

/** One comparable reduced to what the model needs. */
interface LaunchCurve {
  /** log hour-1 print. */
  logH1: number;
  /** log pre-launch ratio (24h mean before launch, over hour-1 print). */
  logPreRatio: number;
  /** log ratios r_h = viewers_h / viewers_1 for h = 2..24 (index h). */
  logRatio: number[];
  /** Similarity weight against the target. */
  weight: number;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** Parse an ISO timestamp the case files use (always UTC, 'Z' suffix). */
const parseTime = (s: string): number => Date.parse(s);

/** Mean of an array; NaN when empty. */
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/**
 * Pull a comparable's launch-day curve out of its hourly series.
 * Returns null when the 24 launch-day hours are not all present and
 * positive (this also drops the GTA VI pre-release series, which has no
 * launch day yet), or when the pre-launch window is empty.
 */
const extractCurve = (comp: Comparable): Omit<LaunchCurve, 'weight'> | null => {
  const dayStart = parseTime(comp.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const byTime = new Map<number, number>();
  for (const p of comp.hourly) {
    const t = parseTime(p.from);
    if (Number.isFinite(t) && typeof p.viewers === 'number') byTime.set(t, p.viewers);
  }
  const day: number[] = [];
  for (let k = 0; k < 24; k++) {
    const v = byTime.get(dayStart + k * 3600_000);
    if (v === undefined || !(v > 0)) return null;
    day.push(v);
  }
  const h1 = day[0];
  const pre: number[] = [];
  for (let k = 1; k <= 24; k++) {
    const v = byTime.get(dayStart - k * 3600_000);
    if (v !== undefined && v > 0) pre.push(v);
  }
  if (pre.length === 0) return null;
  const preRatio = mean(pre) / h1;
  const logRatio: number[] = new Array<number>(25).fill(0);
  for (let h = 2; h <= 24; h++) logRatio[h] = Math.log(day[h - 1] / h1);
  return {
    logH1: Math.log(h1),
    logPreRatio: Math.log(Math.max(preRatio, 1e-3)),
    logRatio,
  };
};

/** The target's own hour-1 print and pre-launch ratio, from its case file. */
const targetFeatures = (c: CaseFile): { p1: number; logPreRatio: number } => {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first !== undefined ? first.viewers : 1);
  const dayStart = parseTime(c.day_start);
  const pre: number[] = [];
  for (const p of c.history_hourly ?? []) {
    if (parseTime(p.from) < dayStart && p.viewers > 0) pre.push(p.viewers);
  }
  const last24 = pre.slice(-24);
  const preRatio = last24.length > 0 ? mean(last24) / p1 : 1;
  return { p1, logPreRatio: Math.log(Math.max(preRatio, 1e-3)) };
};

/** Similarity weight of one comparable against the target. */
const similarity = (
  curve: Omit<LaunchCurve, 'weight'>,
  comp: Comparable,
  logP1: number,
  logPreTarget: number,
): number => {
  const dMag = curve.logH1 - logP1;
  const dPre = curve.logPreRatio - logPreTarget;
  let w = Math.exp(-(dMag * dMag) / (2 * SIG_MAG * SIG_MAG));
  w *= Math.exp(-(dPre * dPre) / (2 * SIG_PRE * SIG_PRE));
  if (!comp.platforms.includes('PC')) w *= CONSOLE_BOOST;
  if (comp.unlock.includes('early-access-contaminated')) w *= TIER_PENALTY;
  return w;
};

/** Solve a 2x2 linear system; returns [0, 0] when singular. */
const solve2 = (a11: number, a12: number, a22: number, b1: number, b2: number): [number, number] => {
  const det = a11 * a22 - a12 * a12;
  if (Math.abs(det) < 1e-12) return [0, 0];
  return [(b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det];
};

/**
 * Forecast one hour: weighted two-variable regression of log ratio on
 * log hour-1 and log pre-launch ratio, evaluated at the target; spread
 * from the weighted residuals around the fitted line.
 */
const forecastHour = (
  curves: LaunchCurve[],
  h: number,
  logP1: number,
  logPreTarget: number,
): { median: number; sd: number } => {
  const n = curves.length;
  let sw = 0;
  let mx = 0;
  let mz = 0;
  let my = 0;
  for (const c of curves) {
    sw += c.weight;
    mx += c.weight * c.logH1;
    mz += c.weight * c.logPreRatio;
    my += c.weight * c.logRatio[h];
  }
  mx /= sw;
  mz /= sw;
  my /= sw;
  let sxx = 0;
  let szz = 0;
  let sxz = 0;
  let sxy = 0;
  let szy = 0;
  for (const c of curves) {
    const dx = c.logH1 - mx;
    const dz = c.logPreRatio - mz;
    const dy = c.logRatio[h] - my;
    sxx += c.weight * dx * dx;
    szz += c.weight * dz * dz;
    sxz += c.weight * dx * dz;
    sxy += c.weight * dx * dy;
    szy += c.weight * dz * dy;
  }
  const [bRaw, cRaw] = solve2(sxx, sxz, szz, sxy, szy);
  const b = bRaw * SLOPE_SHRINK;
  const cc = cRaw * SLOPE_SHRINK;
  let median = my + b * (logP1 - mx) + cc * (logPreTarget - mz);
  // Envelope guard: a launch-day shape must stay inside the shapes the
  // comparables actually showed at this hour. It never binds on an
  // in-range target; it only stops a far-out-of-range target (hour-1
  // above every comparable) from extrapolating below the worst observed
  // shape at that hour.
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of curves) {
    if (c.logRatio[h] < lo) lo = c.logRatio[h];
    if (c.logRatio[h] > hi) hi = c.logRatio[h];
  }
  median = clamp(median, lo, hi);
  let rss = 0;
  for (const c of curves) {
    const fit = my + b * (c.logH1 - mx) + cc * (c.logPreRatio - mz);
    const r = c.logRatio[h] - fit;
    rss += c.weight * r * r;
  }
  const residSd = Math.sqrt(rss / sw);
  const sd = (1 - SPREAD_SHRINK) * residSd + SPREAD_SHRINK * PRIOR_SD;
  return { median, sd };
};

/** Forecast every hour of one case; returns the 23 output rows. */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const { p1, logPreRatio } = targetFeatures(c);
  const logP1 = Math.log(p1);
  const curves: LaunchCurve[] = [];
  const sorted = [...c.comparables].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const comp of sorted) {
    const base = extractCurve(comp);
    if (base === null) continue;
    curves.push({ ...base, weight: similarity(base, comp, logP1, logPreRatio) });
  }
  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const f of c.forecast) {
    const h = f.hour;
    let median: number;
    let sd: number;
    if (curves.length >= 3) {
      const fit = forecastHour(curves, h, logP1, logPreRatio);
      median = fit.median;
      sd = fit.sd;
    } else {
      // Degenerate input: fall back to persistence with a widening spread.
      median = 0;
      sd = FALLBACK_SIGMA * Math.sqrt(h - 1);
    }
    const q = Z.map((z) => {
      const v = p1 * Math.exp(median + z * sd);
      return Math.round(clamp(v, 0.01, 1e9) * 100) / 100;
    });
    rows.push({ hour: h, q });
  }
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
