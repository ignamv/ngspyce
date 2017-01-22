"""
Plot the frequency response of an RC low-pass filter
"""

import ngspyce
from matplotlib import pyplot as plt
import numpy as np

# Read netlist
ngspyce.cmd('source lowpass.net')

# Calculate small-signal transfer function between 1 kHz and 10 MHz, with 5
# points per decade
ngspyce.cmd('ac dec 5 1k 10meg')

# Read results
freq = np.abs(ngspyce.vector('frequency'))
vout = ngspyce.vector('vout')

# And plot them
fig = plt.figure()
fig.suptitle('Frequency response of an RC low-pass filter')

ax = fig.add_subplot(2, 1, 1)
ax.semilogx(freq, 20*np.log10(np.abs(vout)))
ax.set_ylabel('Magnitude [dB]')

ax = fig.add_subplot(2, 1, 2)
ax.semilogx(freq, np.angle(vout, True))
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Phase [degrees]')

plt.savefig('lowpass.png')
plt.show()
