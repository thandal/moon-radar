#!/usr/bin/env bash
# Fetch LRO LOLA GDR cylindrical DEMs (PDS Geosciences node) into lola_dem/.
#
# The pipeline samples lunar topography via doppler_equator.load_lola_dem(),
# which picks the highest-resolution ldem_<N>.img present in lola_dem/.
# N is the grid resolution in pixels per degree:
#   ldem_4     2 MB   ~7.6 km/px
#   ldem_16   33 MB   ~1.9 km/px   (adequate for nside=400 maps, ~4.4 km/px)
#   ldem_64  506 MB   ~474 m/px    (oversampled for the maps; better for
#                                   SRP-local elevation extraction)
# Heights are meters relative to the 1737.4 km sphere in the DE421
# mean-Earth/polar frame -- the same sphere and frame (MOON_ME) as the SPICE
# geometry, so elevations add radially onto the ellipsoid surface points.
#
# Usage: ./fetch_lola_dem.sh [ppd ...]     # default: 16 64
set -euo pipefail
cd "$(dirname "$0")"

BASE=https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img
RES=("$@")
[ ${#RES[@]} -eq 0 ] && RES=(16 64)

mkdir -p lola_dem
for r in "${RES[@]}"; do
    for ext in lbl img; do
        # -C - resumes partial downloads (the 64 ppd file is ~506 MB).
        curl -sS -f -C - -o "lola_dem/ldem_${r}.${ext}" "$BASE/ldem_${r}.${ext}" \
            || { echo "ERROR fetching ldem_${r}.${ext}"; exit 1; }
    done
    echo "Fetched lola_dem/ldem_${r}.{img,lbl}"
done
