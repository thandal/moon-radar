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
point (SRP)**: the shortest-path point on a smooth lunar ellipsoid.

## 2. Measure timing and frequency offsets

[registration_stability.py](registration_stability.py) calls
`compensated_tx` and `measure_offset` in [freq_offset_hunt.py](freq_offset_hunt.py).
Multiplying the RX samples by a compensated TX reference concentrates the
specular echo into a narrow tone. A ±40-sample search and fractional-sample
refinement find the timing offset; the tone's centroid estimates the remaining
frequency offset between the transmit and receive chains.

Timing corrections are usually tens of microseconds. The tone centroid is
less reliable when the echo fades, so a second frequency measurement follows
in step 4. Cross-polarized captures inherit their paired co-polarized capture's
corrections when the batch uses `--corrections-from`.

## 3. Form the delay–Doppler image

`compute_dd_image` in the imaging module resamples the TX reference onto the
RX timeline, interpolating unwrapped phase to preserve amplitude. It removes
the SRP Doppler, its linear drift, and the measured chain-frequency offset.
GPU correlation then measures echo strength across trial delays and Dopplers.

The resulting image has 3000 Doppler rows and roughly 2920 delay columns.
Its bright outline follows the lunar echo's two extreme-Doppler edges, or
**rims**. The grid samples the signal more finely than its physical resolution:
4 µs delay columns do not imply 4 µs resolution; the current chip rate gives
about 20 µs. Doppler resolution is roughly the inverse integration time.

## 4. Align the echo rims

`measure_rim_offset` compares the measured edges with the predicted rims.
Their mean offset refines the frequency calibration; their difference helps
identify distorted or smeared echoes.

The current `process_file` loop uses a 55 mHz search-window anchor and up to
12 refinement passes. If the initial search misses the echo, `rim_seed_search`
tries coarse offsets first. These July changes recovered looks whose faded
tone had put the image far from its predicted position. Cross-pol looks use
the co-pol rim correction because their own rims are usually too diffuse.

## 5. Sample the image onto the Moon

`lunar_projection` predicts the delay and window-averaged Doppler for each
HEALPix surface pixel and samples that cell of the DD image. Its Doppler model
uses the same SRP drift removed during imaging.

Surface points include LOLA terrain heights when the DEM is available; the
SRP and rim reference stay on the smooth ellipsoid. The current map spacing
is about 4.5 km. Sampling is nearest-neighbor, which leaves some visible rings
along constant-delay contours.

Several surface points can share one DD cell. The projection saves these
counts alongside the map so stacking can mask the most ambiguous region,
including the stripe through the reference point.

## 6. Select, align, and average the maps

The batch saves per-look maps, ambiguity counts, and calibration CSVs.
[stack_maps.py](stack_maps.py) rejects weak echoes, delay estimates at the
search limit, and captures with abnormal rim spread. The latest documented
run retains 107 of 111 looks per channel; one no-signal capture and three
captures with frequency drift are excluded from each.

The stacker averages maps in linear intensity, fits a brightness-versus-incidence
curve for each polarization, and solves session shifts from correlations
between larger surface features. Different viewing geometries rotate the
ambiguous stripe, allowing other looks to supply information there.

The final products include individual-channel and combined maps, with and
without incidence normalization, plus coverage counts. Their small alignment
residuals demonstrate internal consistency; absolute placement remains
unverified. See [REPORT §5](REPORT.md#5-registration-and-stacking) for the
results and the [validation guide](validation/README.md) for checks to run
after changing the pipeline.
