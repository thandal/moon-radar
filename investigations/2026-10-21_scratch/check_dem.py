"""New DEM loader vs the old construction: identical grid?"""
import hashlib, os, sys
import numpy as np
sys.path.insert(0, "/home/than/code/moon-radar")
from spice_setup import furnsh_kernels; furnsh_kernels()
import doppler_equator as de

path = de.find_lola_dem()
de.load_lola_dem(path)
new = de._LOLA_INTERPOLATOR.values
h_new = hashlib.sha256(np.ascontiguousarray(new).tobytes()).hexdigest()
shape_new, dtype_new = new.shape, new.dtype
lat_new, lon_new = [g.copy() for g in de._LOLA_INTERPOLATOR.grid]
de._LOLA_INTERPOLATOR = None; de._LOLA_DEM_PATH = None; del new
import gc; gc.collect()

# old construction (copied from the previous loader)
lbl = de._parse_pds_label(os.path.splitext(path)[0] + ".lbl")
lines, samples = int(lbl["LINES"]), int(lbl["LINE_SAMPLES"])
scale_km = de._pds_number(lbl["SCALING_FACTOR"]) / 1000.0
dn = np.fromfile(path, dtype="<i2")
e = dn.reshape(lines, samples).astype(np.float32); del dn
e *= np.float32(scale_km)
e = e[::-1]
e = np.concatenate((e[:, -1:], e, e[:, :1]), axis=1)
h_old = hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest()
print("shape/dtype", shape_new, dtype_new, "vs", e.shape, e.dtype)
print("grid identical (sha256):", h_new == h_old)
