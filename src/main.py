import numpy as np
from neuron import Network
import matplotlib.pyplot as plt



if __name__ == '__main__':
    net = Network(3, 3, 1.0, (0,0), (2,2))
    map_ = np.ones((3,3)) * 5
    map_[0, 0:2] = 1
    map_[0, 2] = 2
    map_[1,0] = 2
    map_[2,2] = 1
    directory = "sample"

    """
    start = (0, 3)
    goal = (19, 17)
    net = Network(20, 20, 1.0, start, goal)
    directory = "wo_road"

    map_ = np.ones((20, 20)) * 2

    map_[5:13, 3:15] = 8
    obstacle_num = 8
    for _ in range(obstacle_num):
        x = np.random.randint(0,19)
        y = np.random.randint(0,19)
        map_[x:x+1, y:y+1] = 10

    """


    for i in range(100):
        plt.subplots_adjust(wspace=0.4, hspace=0.6)
        plt.subplot(2,2,1)
        plt.imshow(net.spike_map, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
        plt.colorbar()
        plt.title("spike")
        plt.subplot(2,2,2)
        plt.imshow(net.D, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
        plt.colorbar()
        plt.title("D_{ij}")
        plt.subplot(2,2,3)
        plt.imshow(net.d, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
        plt.colorbar()
        plt.title("d_{ij}")
        plt.subplot(2,2,4)
        plt.imshow(map_, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
        plt.title("map")

        plt.colorbar()
        plt.savefig(f"fig/{directory}/{i}.png")
        plt.close()
        net.step(map_)
        if net.terminated:
            break

    result = net.readout()
    plt.figure()
    plt.imshow(result, interpolation='nearest', vmin=0, vmax = 10, cmap='jet')
    plt.colorbar()
    plt.savefig(f"fig/{directory}/path.png")
