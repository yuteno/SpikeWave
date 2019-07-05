import numpy as np
from neuron import Network
import matplotlib.pyplot as plt



net = Network(20, 20, 1.0, (0,3), (19,17))


map_ = np.ones((20, 20)) * 2

map_[5:13, 3:15] = 3

for _ in range(8):
    x = np.random.randint(0,17)
    y = np.random.randint(0,17)
    map_[x:x+1, y:y+1] = 9




#plt.imshow(map_, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
#plt.colorbar()
#plt.show()
#net.neurons[0][0].I += 1
for i in range(100):
    net.step(map_)
    plt.subplot(2,2,1)
    plt.imshow(net.spike_map, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.title("spike")
    plt.subplot(2,2,2)
    plt.imshow(net.D, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.title("D_{ij}")
    plt.subplot(2,2,3)
    plt.imshow(net.d, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.title("d_{ij}")
    plt.subplot(2,2,4)
    plt.imshow(map_, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.title("map")

    plt.colorbar()
    plt.savefig(f"fig/wo_road/{i}.png")
    plt.close()
    if net.terminated:
        break

