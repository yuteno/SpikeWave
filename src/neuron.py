import numpy as np

class Neuron:
    def __init__(self):
        self.v = 0
        self.u = 0
        self.I = 0

    def update(self, d):
        self.v = self.u + self.I
        self.u = min(self.u + 1, 0)
        self.I = (d == 1).sum()


    def spike(self):
        is_spike = False
        if self.v > 0:
            self.v = 1
            self.u = -5
            is_spike = True

        return is_spike


class Network:
    def __init__(self, height, width, lr):
        self.lr = lr
        self.d = np.zeros((height * width, height * width))
        self.D = np.ones((height * width, height * width )) * 5
        self.in_indexes = np.zeros((height * width, height * width))
        self.spike_map = np.zeros((height, width))

        self.neurons = [[0 for j in range(width)] for i in range(height)]
        for i in range(height):
            for j in range(width):
                neuron = Neuron()
                self.neurons[i][j] = neuron
                index = j + i * width
                left = right = j
                up = down = i
                if j > 0:
                    left = j - 1
                if j < width - 1:
                    right = j + 1
                if i > 0:
                    up = i - 1
                if i < height - 1:
                    down = i + 1
                for _i in range(up, down + 1):
                    for _j in range(left, right + 1):
                        self.in_indexes[index][_j + _i * width] = 1
                self.in_indexes[index][index] = 0


    def step(self, map_in):
        self.spike_map = np.zeros_like(self.spike_map)
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                index = j + i * len(self.neurons)
                neuron = self.neurons[i][j]
                neuron.update(self.d[index] * self.in_indexes[index])
                self.d[index] = np.maximum(self.d[index] - 1, 0)
                if neuron.spike():
                    self.d[:, index] = self.D[:, index]
                    self.spike_map[i][j] = 9
                self.D[index,:] += self.lr * (map_in[i, j] - self.D[index, :])








