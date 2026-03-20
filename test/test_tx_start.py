import sigmf
import os
from astropy import time as at

DATA_PREFIX = "data.camras.nl/lunar-radar/2025-06-21/"
rx_filename = [f for f in os.listdir(DATA_PREFIX) if f.startswith('stockert') and f.endswith('meta')][0]
sigmf_file = sigmf.sigmffile.fromfile(DATA_PREFIX + rx_filename, skip_checksum=True)
rx_info = sigmf_file.get_global_info()
rx_captures = sigmf_file.get_captures()

tx_filename = rx_info['core:description'].split(';')[0]
tx_sigmf_filename = f"data.camras.nl/lunar-radar/tx_signals/{tx_filename}"
tx_sigmf_file = sigmf.sigmffile.fromfile(tx_sigmf_filename)
tx_captures = tx_sigmf_file.get_captures()

rx_start_astrotime = at.Time(rx_captures[0]['core:datetime'])
tx_start_astrotime = at.Time(tx_captures[0]['core:datetime'])

print("RX time:", rx_start_astrotime.utc.value)
print("TX time:", tx_start_astrotime.utc.value)

diff_s = (tx_start_astrotime - rx_start_astrotime).to_value('s')
print(f"True TX_START_OFFSET: {diff_s} s")

# Check 2025-09-16
DATA_PREFIX_09 = "data.camras.nl/lunar-radar/2025-09-16/"
rx_filename_09 = [f for f in os.listdir(DATA_PREFIX_09) if f.startswith('stockert') and f.endswith('meta')][0]
sigmf_file_09 = sigmf.sigmffile.fromfile(DATA_PREFIX_09 + rx_filename_09, skip_checksum=True)
rx_info_09 = sigmf_file_09.get_global_info()
tx_filename_09 = rx_info_09['core:description'].split(';')[0]
tx_sigmf_filename_09 = f"data.camras.nl/lunar-radar/tx_signals/{tx_filename_09}"
tx_sigmf_file_09 = sigmf.sigmffile.fromfile(tx_sigmf_filename_09)

rx_start_astrotime_09 = at.Time(sigmf_file_09.get_captures()[0]['core:datetime'])
tx_start_astrotime_09 = at.Time(tx_sigmf_file_09.get_captures()[0]['core:datetime'])
diff_s_09 = (tx_start_astrotime_09 - rx_start_astrotime_09).to_value('s')
print(f"2025-09-16 True TX_START_OFFSET: {diff_s_09} s")

