import numpy as np
from neuron import Network
import matplotlib.pyplot as plt



net = Network(3, 3, 1.0)


#map_ = np.random.randint(0, 8, (20, 20))
map_ = np.ones((3, 3)) * 6
map_[0][0] = 1
map_[0][1] = 2
map_[1][0] = 1
map_[2][0] = 2
map_[2][2] = 1





#plt.imshow(map_, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
#plt.colorbar()
#plt.show()
net.neurons[0][0].I += 1
for i in range(50):
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
    plt.savefig(f"fig/sample/{i}.png")
    plt.close()