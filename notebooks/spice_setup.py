"""
Standard SPICE kernel set for the moon-radar pipeline.

Single source of truth for the kernel list that every analysis script was
previously hard-coding. `doppler_equator.py` deliberately does not furnish
kernels at import time; call `furnsh_kernels()` once at script startup
instead.

The set includes `observatory_radii.tpc`, which some older scripts omitted —
it is a tiny text kernel and harmless when unused, and required by the
geometry helpers that look up station radii.
"""

import os
import cspyce as csp

# Resolve relative to this file so scripts work regardless of cwd.
SPICE_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "spice_kernels")

SPICE_KERNELS = [
    "naif0012.tls",
    "de440s.bsp",
    "pck00011.tpc",
    "earth_latest_high_prec.bpc",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
    "observatories.bsp",
    "observatories.tf",
    "observatory_radii.tpc",
]


def furnsh_kernels(kernel_dir=None):
    """kclear() and furnish the standard kernel set."""
    kdir = kernel_dir if kernel_dir is not None else SPICE_KERNEL_DIR
    csp.kclear()
    for k in SPICE_KERNELS:
        csp.furnsh(os.path.join(kdir, k))
