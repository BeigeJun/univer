import numpy as np

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))

# 타겟 벡터를 숫자로 매핑하는 딕셔너리
target_to_number = {
    (0.0, 1.0): 0,
    (1.0, 0.0): 1
}

target = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]

input_data = [[0.0, 0.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 1.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]]

input_num = 3
out1_num = 15
out2_num = 7
out_num = 2

bias = 1.0
error = 0.0
total_error = 0.0
lrate = 0.1
epochs = 10000

weight_hid2_to_out = np.random.uniform(low=-5.0, high=5.0, size=(out_num, out2_num))
weight_hid1_to_hid2 = np.random.uniform(low=-5.0, high=5.0, size=(out2_num, out1_num))
weight_in_to_hid1 = np.random.uniform(low=-5.0, high=5.0, size=(out1_num, input_num))
neurons_in_layers = [input_num, out1_num, out2_num, out_num]
biases = [2 * np.random.rand(neurons, 1) - 1 for neurons in neurons_in_layers[1:]]

out_1 = [0.0] * out1_num  # 은닉층 1번째 15개
out_2 = [0.0] * out2_num  # 은닉층 2번째 7개
output = [0.0] * out_num

def forward_pass(data, w, b, bias_num, out):
    for i in range(len(w)):
        for number in range(len(w[i])):
            out[i] += w[i][number] * data[number]
        out[i] = sigmoid(out[i] + b[bias_num][i])
    return out

def Forward_pass(data, w1, w2, w3, b):
    out_1 = forward_pass(data, w1, b, 0, [0.0] * out1_num)
    out_2 = forward_pass(out_1, w2, b, 1, [0.0] * out2_num)
    output = forward_pass(out_2, w3, b, 2, [0.0] * out_num)
    return out_1, out_2, output

def delta_rule(weight1, weight2, b, b_num, delta_1, delta_2, lrate, out_1, out_2):
    error = 0
    for i in range(len(weight1)):
        for j in range(len(weight2)):
            error += delta_1[j] * weight2[j][i]
        delta_2[i] = error * out_2[i] * (1 - out_2[i])
        for number in range(len(weight1[i])):
            weight1[i][number] += lrate * delta_2[i] * out_1[number]
        b[b_num][i] += lrate * delta_2[i]
    return delta_2, weight1

def first_delta_rule(weight, b, b_num, delta, target, output, out_2):
    for i in range(len(weight)):
        first_error = target[i] - output[i]
        delta[i] = first_error * output[i] * (1 - output[i])
        for j in range(len(weight[i])):
            weight[i][j] += lrate * delta[i] * out_2[j]
        b[b_num][i] += lrate * delta[i]
    return delta, weight

def Backward_pass(data, target, w1, w2, w3, b, out1, out2, out_put, lrate):
    delta_1 = np.zeros(len(w3))
    delta_2 = np.zeros(len(w2))
    delta_3 = np.zeros(len(w1))
    delta_1, w3 = first_delta_rule(w3, b, 2, delta_1, target, out_put, out2)
    delta_2, w2 = delta_rule(w2, w3, b, 1, delta_1, delta_2, lrate, out1, out2)
    delta_3, w1 = delta_rule(w1, w2, b, 0, delta_2, delta_3, lrate, data, out1)
    return w1, w2, w3

def train(input_data, target_data, w1, w2, w3, b, lrate, epochs):
    for epoch in range(epochs):
        total_error = 0.0
        for i in range(len(input_data)):
            out_1, out_2, output = Forward_pass(input_data[i], w1, w2, w3, b)
            error = 0.0
            for j in range(len(target_data[i])):
                error += 0.5 * (target_data[i][j] - output[j]) ** 2
            total_error += error
            total_error = total_error / 3

            w1, w2, w3 = Backward_pass(input_data[i], target_data[i], w1, w2, w3, b, out_1, out_2, output, lrate)

        if epoch % 100 == 0:
            print("step : %4d    Error : %7.10f " % (epoch, total_error))

train(input_data, target, weight_in_to_hid1, weight_hid1_to_hid2, weight_hid2_to_out, biases, lrate, epochs)

def find_closest_target(output, target_data):
    min_distance = float('inf')
    closest_target = None
    for target in target_data:
        distance = sum([(output[i] - target[i]) ** 2 for i in range(len(target))]) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_target = target
    return closest_target

input_data = [[0.0, 1.0, 1.0],
              [1.0, 1.0, 1.0]]

for i in range(2):
    out_1, out_2, output = Forward_pass(input_data[i], weight_in_to_hid1, weight_hid1_to_hid2, weight_hid2_to_out, biases)
    closest_target = find_closest_target(output, target)
    closest_target_tuple = tuple(closest_target)
    predicted_number = target_to_number[closest_target_tuple]
    print(f"Input: {input_data[i]}, Output: {output}, Closest Target: {closest_target}, Predicted Number: {predicted_number}")
