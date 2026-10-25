import sys; sys.path.insert(0, "/home/than/code/moon-radar")
from spice_setup import furnsh_kernels; furnsh_kernels()
import doppler_equator as de; de.load_lola_dem()
