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
    def __init__(self, height, width, lr, start, goal):
        self.lr = lr
        self.d = np.zeros((height * width, height * width))
        self.D = np.ones((height * width, height * width )) * 5
        self.in_indexes = np.zeros((height * width, height * width))
        self.spike_map = np.zeros((height, width))
        self.AER = np.zeros((1, height, width))
        self.spike_map_noupdate = np.zeros((height, width))
        self._terminated = False
        self.start = start
        self.goal = goal
        self.SPIKED = 9

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

        self.neurons[start[0]][start[1]].I += 1


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
                    self.spike_map[i][j] = self.SPIKED
                    self.spike_map_noupdate[i][j] = self.SPIKED
                    if self.spike_map[self.goal] == self.SPIKED:
                        self._terminated = True
                self.D[index,:] += self.lr * (map_in[i, j] - self.D[index, :])
        self.AER = np.concatenate([self.AER, self.spike_map[np.newaxis, :, :]])
        #self.spike_map[self.start] = 10
        #self.spike_map[self.goal] = 10

    def readout(self):
        AER = self.AER
        path_map = np.zeros_like(AER[0])
        all_time_steps = AER.shape[0]
        width = AER.shape[1]
        height = AER.shape[2]

        current_point = np.array(self.goal)
        start_point = np.array(self.start)
        path_map[current_point[0],current_point[1]] = self.SPIKED
        for t in reversed(range(0, all_time_steps)):
            spike_map = AER[t - 1]
            distance = 10000000
            candidate = current_point
            for _x in range(current_point[1] - 1, current_point[1] + 1 + 1):
                if _x < 0 or _x >= width:
                    continue
                for _y in range(current_point[0] - 1, current_point[0] + 1 + 1):
                    if _y < 0 or _y >= height:
                        continue
                    if _y == current_point[0] and _x == current_point[1]:
                        continue

                    if spike_map[_y][_x] == self.SPIKED:
                        print(t, _y, _x)
                        temp_point = np.array([_y, _x])
                        temp_dist = np.linalg.norm(temp_point - start_point)
                        if temp_dist < distance:
                            distance = temp_dist
                            candidate = temp_point
            current_point = candidate
            path_map[current_point[0], current_point[1]] = self.SPIKED
            if (current_point == start_point).all():
                print(current_point)
                print(start_point)
                print("readout break")
                break
        return path_map




    @property
    def terminated(self):
        return self._terminated



