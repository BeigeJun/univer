import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

target = [[1.0, 0.0],[0.0 , 1.0],[0.0, 1.0],[0.0,1.0]]

input_data= [[0.1, 0.2],
            [0.2, 0.2],
            [0.8, 0.7],
            [0.9, 0.9]]
input_num = 2
out_num = 2

weight_in_to_out = np.random.uniform(low=0.01, high=0.2, size=(out_num, input_num))
bias_out = np.random.rand(out_num, 1)

bias = 1.0
error = 0.0
total_error = 0.0
lrate = 0.01
epochs = 1000

output=[0.0]*out_num

def sigmoid(x) :
    return (1.0 / (1.0 + np.exp(-x)))

def forward_pass(data, w, b, out):
    for i in range(len(w)):
        for number in range(len(w[i])):
            out[i] += w[i][number] * data[number]
        out[i] = sigmoid(out[i] + b[i])

    return out

def Forward_pass(data, w1, b):
    output = forward_pass(data, w1, b, [0.0] * out_num)
    return output

def first_delta_rule(data, weight, b, delta, target, output):
    for i in range(len(weight)):
        first_error = target[i] - output[i]
        delta[i] = first_error * output[i] * (1 - output[i])
        for j in range(len(weight[i])):
            weight[i][j] += lrate * delta[i] * data[j]
        b[i] += lrate * delta[i]
    return delta, weight

def Backward_pass(data, target, w1, b, out_put):
    delta_1 = np.zeros(len(w1))
    delta_1, w1 = first_delta_rule(data, w1, b, delta_1, target, out_put)
    return w1

def train(input_data, target_data, w1, b, lrate, epochs):
    # minimum_error = 10.0
    # comparison_error = 10.0
    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(input_data)):
            output = Forward_pass(input_data[i], w1, b)
            error = 0.0
            for j in range(len(target_data[i])):
                error += 0.5 * (target_data[i][j] - output[j]) ** 2
            total_error += error
            total_error = total_error / 2
            # comparison_error = total_error
            w1 = Backward_pass(input_data[i], target_data[i], w1, b, output)
        if epoch % 1 == 0:
            print("step : %4d    Error : %7.10f " % (epoch, total_error))
        # if minimum_error > comparison_error:
        #     minimum_error = comparison_error
        if total_error <= 0.03:
          break

train(input_data, target, weight_in_to_out, bias_out, lrate, epochs)
# 평가 데이터
input_data_2= [[0.1, 0.2],
            [0.2, 0.2],
            [0.8, 0.7],
            [0.9,0.9]]


for i in range(len(input_data_2)):
  a = Forward_pass(input_data_2[i], weight_in_to_out, bias_out)
  print(a)
  Max = a[0]
  Max_num = 0
  for j in range(len(a)-1):
      if(Max < a[j+1]):
          Max = a[j+1]
          Max_num = j+1
  if(Max_num == 0):
      print("A")
  elif(Max_num == 1):
      print("B")
