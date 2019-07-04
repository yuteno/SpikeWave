import numpy as np
from neuron import Network
import matplotlib.pyplot as plt



net = Network(20, 20, 1.1)


#map_ = np.random.randint(0, 8, (20, 20))
map_ = np.ones((20, 20)) * 7

for i in range(20):
    map_[i, 10] = 1


plt.imshow(map_, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
plt.colorbar()
plt.show()
net.neurons[0][10].I += 1
for i in range(100):
    net.step(map_)
    plt.subplot(1,3,1)
    plt.imshow(net.spike_map, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.subplot(1,3,2)
    plt.imshow(net.D, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.subplot(1,3,3)
    plt.imshow(net.d, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.colorbar()
    plt.show()