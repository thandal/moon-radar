"""
Retired: re-measuring captures whose timing shift hit the search limit.

This script once re-processed three 2025-09-11 Stockert captures (06:39:21,
07:52:07, 08:05:44) whose chain delay railed the original +/-20-sample
shift search -- with +/-40 they resolved at +91 / +80 / +125 us, tone SNR
68-94 -- and patched their rows into an existing runs CSV in place. That
patching is unsafe and the code is gone (it is in the git history): it
rewrote the production CSV with an unquoted writer, dropped every column
added since, duplicated registration_stability.process_one by hand, and
pointed at a run directory that no longer exists.

The same job is now one ordinary batch run with a wider search, into a
fresh output directory, the opposite-sense channel first and the same-sense one from its twin:

    .conda/bin/python registration_stability.py --station <station> --chan chan1 \\
        --dates <dates> --per-date 0 --max-shift <N> --out-dir <new dir>
    .conda/bin/python registration_stability.py --station <station> --chan chan0 \\
        --dates <dates> --per-date 0 --max-shift <N> --out-dir <new dir> \\
        --corrections-from <new dir>/registration_runs_<station>_chan1.csv

stack_maps.py rejects any look whose shift still sits at the new limit.
"""

import sys


def main():
    sys.exit("recover_railed.py is retired: it patched runs CSVs in place. Re-run "
             "registration_stability.py with a larger --max-shift into a fresh "
             "--out-dir instead (see this file's docstring for the commands).")


if __name__ == "__main__":
    main()
