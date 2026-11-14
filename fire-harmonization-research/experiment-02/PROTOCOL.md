# Experiment 02: signal-conditioned kernel occupancy calibration

## Frozen decision before executing this experiment

The first logistic model fits daily binary labels but has a geographic background
floor, an unbounded linear dependence on latitude/longitude, and past-day features
that can change the calendar peak. The simple power law is already a strong
calendar benchmark. Therefore the next experiment constrains the response and
fits the quantity the calendar actually reports.

We considered native-resolution image super-resolution (original imagery and
authenticated downloads unavailable), another flexible classifier (the same weak
features and packages unavailable), isotonic density calibration (a plausible
alternative, but unconstrained extrapolation and sparse high-density bins need
care), and a bounded local-signal model. We selected the last option as the most
plausible executable improvement, not as a guaranteed winner.

## Known building blocks, explicit adaptation

Use a spatial box kernel over the current day's MODIS-occupied cells. Let s be
the fraction of eligible neighboring grid cells with a MODIS detection. Predict
VIIRS occupancy as:

    p = 1 - exp(-alpha * s**beta)

This is a complementary-log-log / Poisson-occupancy response applied to a local
intensity estimate. It is a constrained calibration model using established
components, not a newly established satellite detection algorithm. All outputs
remain at the available **0.25-degree grid**, not 375-m super-resolution.

- Zero local MODIS signal produces zero estimated activity: no latitude-only floor.
- No past-day features or temporal smoothing; same-day input alone determines the output.
- Probabilities remain in [0,1], with a monotone response to local signal.
- Fit alpha by matching training expected area-weighted cell-days to observed
  VIIRS cell-days, using exact sufficient-statistic histograms. This is a
  method-of-moments count calibration, not logistic likelihood training.
- Regional alpha values are shrunk in log-space toward a pooled alpha. A region
  excluded from fitting uses the pooled value; no numerical longitude extrapolation.
- The model can redistribute activity around MODIS support. It cannot locate
  isolated missed fires outside that support; report that reference activity explicitly.

## Fixed candidate set and selection

Neighborhood radii: 1, 2, 4 source-grid cells (0.25, 0.5, 1 degree).
Response beta: 0.5, 0.75, 1.0. Regional pooling weight: 0, 0.5, 1.0, where 0 is
fully pooled and 1 is the fitted regional response. Exactly 27 configurations.

Select the minimum validation score:

    0.5 * country weekly WAPE + 0.5 * mean(sector weekly WAPE)

WAPE = sum absolute weekly errors / sum observed VIIRS activity, over the same
matched dates. This targets useful calendar totals while preventing the large
western record from hiding eastern errors. Ties prefer more pooling, smaller
support and beta nearer 1. No selection uses a test-year score or peak timing.
Report peak timing separately; absence of temporal features does not guarantee
that nonlinear aggregate corrections preserve peaks.

## Comparisons and validation

1. Comparable development benchmark: train 2012-2019, select on 2020-2022,
   evaluate 2023-2025. These test years were already inspected in Experiment 01;
   never call this a fresh blind test. Use the same training-only static mask,
   dates, grid, latitude weights and reference as the prior experiment.
2. Rolling-origin backtests for 2020-2025: for each year Y, inner training is
   2012..Y-3 and selection uses Y-2,Y-1. Then freeze settings, refit on 2012..Y-1,
   and score Y. Refit the simple baselines on the same history. Learn masks within
   each fit. This prevents within-run temporal leakage but does not erase prior
   researcher exposure to the public dataset.
3. Cold east: fit/select west of 92W, predict east of 90W in 2023-2025, with no
   eastern labels used for fitting, masks or selection. Use pooled alpha there.
4. Retain the exact old logistic predictions for comparison in the fixed split.
   Do not compare them as if they had the rolling model's longer training history.

Use matched-day weekly MAE, WAPE, seasonal errors, regional errors, peak-bin shifts,
and support-miss share. Generate the same five figure families as Experiment 01.
No synthetic result, fine-resolution claim, main-app write, or automatic promotion.

## Sources consulted

- [Complementary-log-log response](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.CLogLog.html)
- [Kernel intensity estimation](https://cran.stat.auckland.ac.nz/web/packages/spatstat.explore/spatstat.explore.pdf)
- [Monotone regression alternative](https://scikit-learn.org/stable/modules/isotonic.html)
- [MODIS/VIIRS observation operators](https://gmd.copernicus.org/articles/19/10123/2026/)

The first three references establish mathematical components, not evidence that
this particular adaptation improves MODIS/VIIRS harmonization. That is tested here.
