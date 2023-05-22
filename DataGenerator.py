import numpy as np
from tensorflow import keras
import os
import math
from collections import deque

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_directory, batch_size):
        self.directory = data_directory
        self.batch = batch_size
        self.sample_count = len(os.listdir(self.directory))//2
    
    def __len__(self):
        return math.ceil(self.sample_count / self.batch)
    
    def __getitem__(self, index):
        first = index * self.batch
        last = min((index + 1) * self.batch, self.sample_count)
        lx = []
        ly = []
        for i in range(first, last):
            (x, y) = self.__parseSample(i)
            lx.append(x)
            ly.append(y)
        # (lx, ly) = self.__parseSample(index)
        return (np.array(lx), np.array(ly))

    def __stringToMultiDimArray(self, st):
        helper_stack = deque()
        for ch in st:
            if ch == '[':
                helper_stack.append([])
            elif ch == ']':
                current = helper_stack.pop()
                if len(helper_stack) > 0:
                    helper_stack[-1].append(current)
                else:
                    helper_stack.append(current)
            elif ch not in [' ', ',']:
                helper_stack[-1].append(int(ch))
        return helper_stack[0]
    
    def __parseSample(self, id):
        sample_x_file_st = self.directory + str(id) + "_x.txt"
        sample_y_file_st = self.directory + str(id) + "_y.txt"
        sample_x_file = open(sample_x_file_st, 'r')
        sample_y_file = open(sample_y_file_st, 'r')
        sample_x_st = sample_x_file.read()
        sample_y_st = sample_y_file.read()
        sample_x = self.__stringToMultiDimArray(sample_x_st)
        sample_y = self.__stringToMultiDimArray(sample_y_st)
        return (sample_x, sample_y)