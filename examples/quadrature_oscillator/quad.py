
# Plot output voltages of an operational amplifier quadrature oscillator
import ngspyce
import numpy as np
from matplotlib import pyplot as plt

# Load netlist
ngspyce.cmd(b'source quad.net')
# Simulate 10 ms
ngspyce.cmd(b'tran 12n 10m 1n')

# Read results
time, vsin, vcos = map(ngspyce.vector, ['time','Vsin','Vcos'])
# And plot them
plt.plot(time*1e3, vcos, label='Vcos')
plt.plot(time*1e3, vsin, label='Vsin')

plt.title('Quadrature op-amp oscillator output voltage')
plt.xlabel('Time [ms]')
plt.ylabel('Voltage [V]')
plt.savefig('quad.png')
plt.show()
