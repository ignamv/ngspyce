import ngspyce
import numpy as np
from matplotlib import pyplot as plt

ngspyce.cmd('source quad.net')
#trrandom(2 2m 0 10m 0)
ngspyce.cmd('tran 1m 20m')
print('\n'.join(ngspyce.vectorNames()))
vcos, vsin, time = ngspyce.vectors(['Vcos', 'Vsin', 'time']).values()
#np.savetxt('vcos.txt', vcos)
#np.savetxt('vsin.txt', vsin)
#np.savetxt('time.txt', time)
#exit()
plt.plot(time, vcos, label='Vcos')
plt.plot(time, vsin, label='Vsin')
plt.legend()
plt.show()
