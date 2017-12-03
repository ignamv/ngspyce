"""
Plot the frequency response of an RC low-pass filter
"""

import ngspyce
from matplotlib import pyplot as plt
import numpy as np

# Read netlist
ngspyce.source('lowpass.net')

# Calculate small-signal transfer function between 1 kHz and 10 MHz, with 5
# points per decade
ngspyce.ac(mode='dec', npoints=7, fstart=1e3, fstop=10e6)

# Read results
freq = np.abs(ngspyce.vector('frequency'))
vout = ngspyce.vector('vout')

# And plot them
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Frequency response of an RC low-pass filter')

ax1.semilogx(freq, 20*np.log10(np.abs(vout)))
ax1.set_ylabel('Magnitude [dB]')
ax1.grid(True, which='both')

ax2.semilogx(freq, np.angle(vout, True))
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [degrees]')
ax2.grid(True, which='both')
ax2.margins(x=0)

plt.savefig('lowpass.png')
plt.show()
