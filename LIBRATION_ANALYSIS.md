# Lunar Libration Analysis: Optimal Doppler Equator Coverage

To improve the SNR and surface detail of the stacked Moon-radar reflectivity map, we want to collect new datasets at epochs where the mapping degeneracy is shifted away from previous runs. 

This document details the physical background, the top predicted future observation windows for the Dwingeloo (TX) $\rightarrow$ Stockert (RX) baseline, and the predictive planning tool.

---

## 1. Physical Background & Optimal Geometry

In delay-Doppler radar mapping, the coordinate system degenerates near the **Doppler equator** (where the contours of constant delay and constant Doppler are parallel on the Moon's surface). In this zone, a single delay-Doppler cell maps to a long surface arc (the "bright stripe artifact"), carrying very little per-pixel selenographic information. Consequently, these regions must be masked out during stacking.

The Doppler equator is a great circle on the Moon's surface whose orientation is determined by the **Doppler axis of rotation** relative to the observer. 

* **Existing Sessions:** In all four previous sessions (June and September 2025), the Doppler axis was tilted by $30^\circ$ to $45^\circ$ relative to the Moon's polar (Z) axis. This meant their degenerate stripes cut diagonally across the Moon's coordinate grid.
* **The Opportunity:** To cover these gaps, we want to observe when the Doppler axis is aligned as closely as possible with the Moon's polar (Z) axis (i.e. Y component of the axis is near zero). This shifts the Doppler equator directly to the Moon's selenographic equator, exposing the diagonal regions that were previously masked.
* **Doppler Bandwidth Constraint:** To get a high-quality delay-Doppler image, we also require a high **lateral libration velocity** (the tangent velocity of the specular point on the Moon). If this velocity is near-zero, the Moon's rotation is very slow relative to the radar, causing the echo energy to compress into a very narrow frequency range (poor Doppler resolution).

---

## 2. Top Predicted Opportunities (TX Elevation $\ge 45^\circ$, Doppler Span $\ge 5$ Hz)

Using our predictive tool, we scanned the future time window from **June 18, 2026, to September 10, 2026**. The results are filtered to require:
1. **High Elevation:** At least **45°** for the Dwingeloo transmitter and **10°** for the Stockert receiver.
2. **High Doppler Bandwidth:** An estimated L-band (1299.5 MHz) Doppler bandwidth/span of **at least 5.0 Hz** across the Moon's disk to ensure high Doppler resolution.

Here are the top 10 optimal observation opportunities, ranked by their angular displacement from the existing datasets:

| Rank | Peak Date & Time (UTC) | Min Displacement | Dwingeloo Elev | Stockert Elev | Est. Doppler Span | Planning Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **2026-08-08T08:00:00** | **51.21°** | 63.4° | 65.5° | **14.2 Hz** | **Highly Recommended** (Polar-aligned, Max Elev) |
| **2** | 2026-09-03T07:00:00 | 50.99° | 48.1° | 48.9° | **12.4 Hz** | Recommended |
| **3** | 2026-08-09T07:00:00 | 50.04° | 60.5° | 62.4° | **12.1 Hz** | Recommended (High Elevation) |
| **4** | 2026-09-06T05:00:00 | 49.92° | 54.1° | 55.4° | **12.3 Hz** | Recommended |
| **5** | 2026-07-11T10:00:00 | 49.88° | 58.7° | 60.2° | **11.2 Hz** | Recommended (High Elevation) |
| **6** | 2026-09-05T06:00:00 | 49.13° | 64.7° | 67.0° | **16.0 Hz** | Recommended (Max Elevation & Bandwidth) |
| **7** | 2026-09-04T06:00:00 | 46.78° | 62.5° | 64.5° | **15.9 Hz** | High Elevation |
| **8** | 2026-07-12T09:00:00 | 46.46° | 63.9° | 66.1° | **11.0 Hz** | High Elevation |
| **9** | 2026-08-07T09:00:00 | 45.78° | 49.6° | 50.5° | **11.6 Hz** | Visible |
| **10** | 2026-07-13T08:00:00 | 44.13° | 51.1° | 52.2° | **6.1 Hz** | Lower Bandwidth |

> [!TIP]
> The **August 8, 2026 at 08:00:00 UTC** opportunity remains the optimal window. It provides both the largest displacement ($51.21^\circ$), high elevation ($>63^\circ$), and a wide Doppler span ($14.2$ Hz) that guarantees excellent delay-Doppler resolution.

---

## 3. Doppler Equator Shift Visualization

The plot below shows the location of the Doppler equators on the Moon's surface (latitude vs. longitude). The dashed gray/blue curves show where the mapping degeneracy occurred in our existing datasets, while the solid orange curve shows the optimal target of **August 8, 2026**, which cleanly bisects the previous runs.

![Doppler Equator Shift on Lunar Surface](results/ERRORS/libration_opportunities.png)

---

## 4. The Predictive Planning Tool (`predict_libration_opportunities.py`)

A planning script, [predict_libration_opportunities.py](file:///home/than/code/moon-radar/predict_libration_opportunities.py), is in the repository root.

### Features
1. **Dynamic Dataset Loading:** Automatically reads `results/LOLA_DEM_REGISTRATION/registration_runs.csv` to extract previous observation epochs.
2. **Asymmetric Horizon Gating:** Filters results to ensure the Moon is above `--min-tx-elevation` (default 45°) at Dwingeloo and `--min-rx-elevation` (default 10°) at Stockert.
3. **Doppler Span Filtering:** Automatically computes the linear speed of the specular point and filters out epochs below `--min-doppler-span` (default 5.0 Hz).
4. **High-Precision/Low-Precision Fallback:** To allow planning deep into the future, it queries the Earth rotation binary PCK. If the search date falls outside the high-precision coverage (currently expires on Sept 11, 2026), it automatically switches to a geocentric model in the `IAU_EARTH` analytical frame (precision error $< 1^\circ$).
5. **Logical Pass Grouping:** Groups contiguous visible steps into daily "passes" and extracts only the local peak window per day.

### Example Commands
Run the default search (current date to Sept 10, 2026 with default constraints):
```bash
PYTHONPATH=. .conda/bin/python predict_libration_opportunities.py
```

Search deep into 2027 (automatically switches to geocentric fallback):
```bash
PYTHONPATH=. .conda/bin/python predict_libration_opportunities.py --start-date 2026-09-01 --end-date 2027-09-01
```

---

## Appendix: Metrics for Existing Sessions

Below is a summary of the Moon elevation angles and estimated L-band (1299.5 MHz) Doppler bandwidths/spans at the midpoint of each of the four previously collected sessions:

| Session Date | TX (Dwingeloo) Elev | RX (Stockert) Elev | SRP Speed | Est. Doppler Span | Session Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **2025-06-21** | 44.7° | 46.0° | 0.880 m/s | **15.3 Hz** | Good elevation, wide Doppler bandwidth |
| **2025-09-10** | 14.5° | 14.5° | 0.959 m/s | **16.6 Hz** | Low elevation (setting), wide Doppler bandwidth |
| **2025-09-11** | 29.6° | 29.8° | 0.333 m/s | **5.8 Hz** | Moderate elevation, very narrow Doppler bandwidth |
| **2025-09-16** | 15.5° | 14.4° | 0.929 m/s | **16.1 Hz** | Low elevation (rising), wide Doppler bandwidth |

