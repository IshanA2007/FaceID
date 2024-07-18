import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage
def display_image(x, y, code_output):
    # just for fun
    print("This should be:", y)
    print("Code classified it as:", code_output)
    image = x.reshape((32, 32))
    image = image * 255
    plt.imshow(image, cmap='gray')
    plt.show()
def load_train_set():
    with open("training.csv") as f:
        training_set = []
        for line in f:
            number, *image = line.strip().split(",")
            y = np.zeros((2, 1))
            y[int(number)] = 1
            x = np.zeros((1024, 1))
            for i in range(len(x)):
                x[i][0] = int(image[i])/255
            training_set.append((x, y))
    return training_set
def load_test_set():
    with open("training.csv") as f:
        testing_set = []
        for line in f:
            number, *image = line.strip().split(",")
            y = int(number)
            x = np.zeros((1024, 1))
            for i in range(len(x)):
                x[i][0] = int(image[i])/255
            testing_set.append((x, y))
    return testing_set
def test_network(weights, biases, testing_set):
    display = False
    a = input("display images of incorrect numbers?").lower()
    if len(a)>0 and a[0] == "y":
        display = True
    incorrect = 0
    total = len(testing_set)
    for x, y in testing_set:
        output = p_net(vectorizesigmoid, weights, biases, x)
        if output.argmax() == y:
            pass
        else:
            incorrect += 1
            if display:
                display_image(x, y, output.argmax())
    print("Out of", total, "total images in testing set,", incorrect, "were classified incorrectly.")
    print("Error:", str((incorrect/total)*100)+ "%")
def p_net(A_vec, weights, biases, ins):
    a = [ins]
    n = len(weights)
    for i in range(1, n):
        a.append(A_vec(weights[i] @ a[i-1] + biases[i]))
    return a[n-1]
def make_network(training_set, hidden_layers):
    weights = [None]
    biases = [None]
    lastlayersize = len(training_set[0][0])
    hidden_layers = hidden_layers + [len(training_set[0][1])]
    for layersize in hidden_layers:
        bias = np.random.uniform(-1, 1, size=(layersize, 1))
        biases.append(bias)
        weight = np.random.uniform(-1, 1, size=(layersize, lastlayersize))
        weights.append(weight)
        lastlayersize = layersize
    return weights, biases

def distort(x):
    thing_to_do = np.random.randint(1, 10)
    # if thing_to_do == 1, do nothing
    if thing_to_do == 2:
        # shift up 1 pixel
        x = np.roll(x, -32)
    elif thing_to_do == 3:
        # shift down 1 pixel
        x = np.roll(x, 32)
    elif thing_to_do == 4:
        # shift right 1 pixel
        x = x.reshape((32, 32))
        x = np.roll(x, 1)
        x = x.reshape((1024, 1))
    elif thing_to_do == 5:
        # shift left 1 pixel
        x = x.reshape((32, 32))
        x = np.roll(x, -1)
        x = x.reshape((1024, 1))
    elif thing_to_do == 6:
        # rotate 15 degrees left
        x = x.reshape((32, 32))
        x = ndimage.rotate(x, 15, reshape=False)
        x = x.reshape((1024, 1))
    elif thing_to_do == 7:
        # rotate 15 degrees left
        x = x.reshape((32, 32))
        x = ndimage.rotate(x, -15, reshape=False)
        x = x.reshape((1024, 1))
    elif thing_to_do == 8:
         # stretch horizontally
         x = x.reshape((32, 32))
         x = ndimage.zoom(x, (1, 1.1))
         # crop down to 28x28
         x = x[:, :32]
         x = x.reshape((1024, 1))
    elif thing_to_do == 9:
        # stretch vertically
        x = x.reshape((32, 32))
        x = ndimage.zoom(x, (1.1, 1))
        # crop down to 32x32
        x = x[:32, :]
        x = x.reshape((1024, 1))
    return x

def train_network(A_vec, A_deriv, weights, biases, epochs, learning_rate, training_set):
    layers = len(weights)
    for epochcount in range(epochs):
        print("epoch", str(epochcount))
        misclassified = 0
        total = 0
        for x, y in training_set:
            x = distort(x)
            a = [x]
            dots = [None]
            for i in range(1, layers):
                dot = weights[i] @ a[i-1] + biases[i]
                dots.append(dot)
                a.append(A_vec(dot))
            delta_n = np.multiply(A_deriv(dots[-1]), (y-a[-1]))
            total += 1
            if np.argmax(y) != np.argmax(a[-1]):
                misclassified += 1
            deltas = [0 for i in range(layers)]
            deltas[-1] = delta_n
            for i in range(layers-2, 0, -1):
                deltas[i] = np.multiply(A_deriv(dots[i]), np.transpose(weights[i+1]) @ deltas[i+1])
            for i in range(1, layers):
                biases[i] += learning_rate*deltas[i]
                weights[i] += learning_rate*deltas[i] @ np.transpose((a[i-1]))
        print("epoch complete. saving weights and biases...")
        print("Out of", total, "images in training set,", misclassified, "were incorrectly classified, for an error of", str(100*(misclassified/total)) + "%.")
        with open("mnistweightsbiases_distort.txt", "wb") as f:
            data = {"w": weights, "b": biases}
            pickle.dump(data, f)
    return weights, biases
def sigmoid(num):
    return 1 / (1 + np.e ** (num * -1))
def sigmoid_derivative(num):
    return (np.e**(num*-1)/((1+(np.e**(num * -1)))**2))
vectorizesigmoid = np.vectorize(sigmoid)
vectorizesigderiv = np.vectorize(sigmoid_derivative)
# TRAIN THE NETWORK
a = input("test network? 'y' to test, anything else to train.").lower()
if len(a) > 0 and a[0] == "y":
    print("loading weights and biases from file")
    with open("mnistweightsbiases_distort.txt", "rb") as f:
        data = pickle.load(f)
        w = data["w"]
        b = data["b"]
    print("data loaded!")
    print("loading test set...")
    testing_set = load_test_set()
    print("testing network")
    test_network(w, b, testing_set)
else:
    training_set = load_train_set()
    hidden_layers = [300, 100]
    a = input("load from file?").lower()
    if len(a) > 0 and a[0] == "y":
        print("loading weights and biases from file")
        with open("mnistweightsbiases_distort.txt", "rb") as f:
            data = pickle.load(f)
            w = data["w"]
            b = data["b"]
        print("data loaded!")
    else:
        print("generating random weights and biases")
        w, b = make_network(training_set, hidden_layers)
        print("generated weights and biases!")
    train_network(vectorizesigmoid, vectorizesigderiv, w, b, 100000000, 0.1, training_set)