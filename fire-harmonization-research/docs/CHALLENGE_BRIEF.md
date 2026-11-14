# Challenge interpretation supplied in the conversation

Develop a harmonized burning-activity calendar in a web application, combining
MODIS's extensive historical series with VIIRS continuity and greater detail to
provide a consistent time series of hotspot activity.

The first part is harmonization: account for sensor differences so that patterns,
trends and anomalies become the focus. The second part is the application: let a
user select a region/date range and download analytics and visualizations.

The calendar should show recurring active periods, years with greater burning
activity, and how a current period compares with the history of the same area.
Accessible color ramps can communicate differences across months. A downloadable
CSV of detected hotspots is an additional useful output.

## Additional user decisions recorded after that brief

- NASA-distributed data for detection/history; the response simulator stays separate.
- Practical harmonization first, rather than a full latent-fire research model.
- ML should eventually be the primary harmonizer and support all date selections.
- The final desired calibration direction is **MODIS -> VIIRS reference**, improving
  MODIS-derived activity estimates from overlap-period learning.
- For now, create a separate folder, preserve the ideas and attempt preliminary
  tests. **Do not implement or integrate anything in the main app.**

This is a discussion summary, not a claim that the inaccessible challenge Resources
tab was retrieved or that the project already meets every challenge requirement.
