# From a radar recording to a lunar map

The receiver records a mixture of echoes from across the Moon. The pipeline
first sorts that signal by travel time and Doppler shift, then uses the
observing geometry to place the echo strength on the lunar surface. Repeating
this for many captures reduces speckle and fills gaps left by ambiguous views.

One capture produces one **look**: a delay–Doppler (DD) image and a surface map.
This guide follows that path; the [report](REPORT.md) holds the measurements
and conventions, and the [README](README.md#running-the-pipeline) has batch commands.

## 1. Read the recording and establish the geometry

`load_observation` in [doppler_equator_alignment.py](doppler_equator_alignment.py)
reads the received SigMF samples and the TX waveform named in their metadata.
Production Stockert captures have 250,000 samples/s and a 1299.5 MHz carrier.
The nominal transmission starts one second after the RX recording starts.
The TX file's timestamp is its generation time, so it cannot set the emission time.

[spice_setup.py](spice_setup.py) loads the geometry kernels through
`furnsh_kernels()`. [doppler_equator.py](doppler_equator.py) then predicts the
transmitter→Moon→receiver travel time. Its anchor is the **specular reference
point (SRP)**: the shortest-path point on a smooth lunar ellipsoid. SPICE's
Moon ellipsoid (`pck00011.tpc`) has all three radii 1737.4 km, so it is a
sphere, the one the LOLA heights are measured from.

## 2. Measure timing and frequency offsets

[registration_stability.py](registration_stability.py) calls
`compensated_tx` and `measure_offset` in [freq_offset_hunt.py](freq_offset_hunt.py).
Multiplying the RX samples by a compensated TX reference concentrates the
specular echo into a narrow tone. A ±40-sample search finds the timing
offset at the first peak of tone strength versus shift after the tone
reaches the strength the stacker requires (the specular point is the
nearest surface, so nothing arrives before it; later echo can peak higher),
refined to a fraction of a sample; looks whose first peak is not resolved
are recorded as such and not stacked. The tone's centroid estimates the
remaining frequency offset between the transmit and receive chains.

Timing corrections are usually tens of microseconds. The tone centroid is
less reliable when the echo fades, so a second frequency measurement follows
in step 4.

Each capture has two polarization channels, named relative to the
transmitted circular wave: the **opposite-sense** channel (`chan1`), which
holds the strong mirror-like echo, and the weaker **same-sense** channel
(`chan0`); see the [report's terms](REPORT.md#1-terms-and-resolution).
Same-sense captures inherit their paired opposite-sense capture's
corrections when the batch uses `--corrections-from`.

## 3. Form the delay–Doppler image

`compute_dd_image` in the imaging module resamples the TX reference onto the
RX timeline, interpolating unwrapped phase to preserve amplitude. It removes
the SRP Doppler, its linear drift, and the measured chain-frequency offset.
GPU correlation then measures echo strength across trial delays and Dopplers
(`correlate_blocks`). The capture is cut into 4.7 ms blocks. One short FFT
correlation per block gives every delay at once, and a Fourier transform
across the blocks gives the Doppler rows. The phase change inside each block
is kept as a short power series. The Zadoff-Chu codes need that: plain block
sums fold echo from other delays into the image.

The resulting image has 3000 Doppler rows and roughly 2920 delay columns.
Its bright outline follows the lunar echo's two extreme-Doppler edges, or
**rims**. The grid samples the signal more finely than its physical resolution:
4 µs delay columns do not imply 4 µs resolution; the chip rate gives about
20 µs at 50,000 chips/s (most 2025 looks) and 40 µs at 25,000 (most
2026-10-11 looks). Doppler resolution is roughly the inverse integration time.

## 4. Align the echo rims

`measure_rim_offset` compares the measured edges with the predicted rims.
Their mean offset refines the frequency calibration; their difference helps
identify distorted or smeared echoes.

The current `process_file` loop uses a 55 mHz search-window anchor and up to
12 refinement passes. If the initial search misses the echo, `rim_seed_search`
tries coarse offsets first; its offset is used only if the rims then pass
their gates there. These July changes recovered looks whose faded tone had
put the image far from its predicted position. Each look records whether the
calibration converged (`rim_status`), and only converged looks are stacked.
Same-sense looks use the opposite-sense rim correction because their own rims are
usually too diffuse.

The rims measure only a Doppler offset. A delay offset also moves both
edges, inward or outward together, most near the SRP where an edge's Doppler
changes fastest with delay; the timing search leaves lag 0 a few
microseconds off (2026-10-11: median +6 µs), enough to put pixels along the
Doppler equator beyond the echo's edge (a dark band) or inside it (a bright
one). `fit_edge_offsets` therefore fits the image near both edges, from 10°
to 80° incidence, against a model image built from SPICE geometry, the LOLA
terrain and a generic scattering law, with the brightness left free along
each limb, and returns the delay and Doppler offsets together (with
jackknife errors). The projection applies them when the fit succeeds
(`edge_status` `fitted`); only those looks are stacked, and same-sense looks
take their twin's.

## 5. Sample the image onto the Moon

`lunar_projection` predicts the delay and window-averaged Doppler for 4×4
points spread across each HEALPix surface pixel and averages the DD image
cells they fall in. Its Doppler model uses the same SRP drift removed during
imaging.

Surface points include LOLA terrain heights (the DEM is required); the SRP
and rim reference stay on the smooth ellipsoid. Delays are measured from the
terrain's light time at the SRP, since that is the echo the timing offset
was measured on. The current map spacing is about 4.5 km. The projection also
saves, per pixel, the ground area one DD cell covers, and the image's noise
floor goes in the runs CSV.

Several surface points can share one DD cell. The projection saves these
counts alongside the map so stacking can mask the most ambiguous region,
including the stripe through the reference point.

## 6. Select, align, and average the maps

The batch saves per-look maps, ambiguity counts, cell areas and calibration
CSVs, and lists every capture it did not map with the reason.
[stack_maps.py](stack_maps.py) rejects weak echoes, delay estimates at the
search limit, a tone at the edge of the frequency search, rim calibrations
that did not converge, and captures with abnormal rim spread, and writes
every decision to a CSV. The July 2025 run retained 107 of 111 looks per
channel; one no-signal capture and three captures with frequency drift were
excluded from each.

The stacker turns each look into brightness per unit area with its noise
floor removed, scales it to a mean of 1 at 20–50° incidence, and averages
the looks with equal weights. It fits a mean brightness-versus-incidence
curve for each polarization, and solves session shifts from correlations
between larger surface features, stopping if they fail their checks.
Different viewing geometries rotate the ambiguous stripe, allowing other
looks to supply information there. The curve is fitted in 1° steps of
incidence: it falls about fivefold within 12° of normal incidence, and the
coarser steps used before 2026-10-25 left a glow at each session's
specular point.

Looks from one night share one geometry, so they repeat the same blur and
mirror ghosts. The main stack therefore averages each night on its own and
then weights the nights at each place and feature size by their expected
error: geometry error from passing a synthetic surface through every
look's own geometry, plus noise measured from the night's own looks. Where
one night sees a place much better, as 2026 does at the 2025 nights'
specular point, it gets most of the weight there. Within a night, a look
supplies fine detail only where its delay-Doppler cells are small enough
to place it, and broad features everywhere. Each season has its own
brightness-versus-incidence curve: the 2026 observation lit the centre of
the disk relatively more strongly.

The final products include individual-channel and combined maps, with and
without incidence normalization, plus coverage counts. Their small alignment
residuals demonstrate internal consistency; absolute placement remains
unverified. See [REPORT §5](REPORT.md#5-registration-and-stacking) for the
results and the [validation guide](validation/README.md) for checks to run
after changing the pipeline.
