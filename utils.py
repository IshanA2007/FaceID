import cv2
import numpy as np
import csv
def gen_training(uservid, othervid):
    training = []
    # process uservid, othervid into images (1024 numbers, 0-255), add to training list with ids
    gen_frame_data(uservid, training, 1)
    gen_frame_data(othervid, training, 0)
    return training

def gen_frame_data(filepath, training, id):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error opening video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (32, 32), interpolation=cv2.INTER_AREA)
        frame_numbers = resized_frame.flatten().tolist()
        training.append((id, frame_numbers))
    cap.release()
    return

def gen_training_csv(uservid, othervid):
    training = gen_training(uservid, othervid)
    # process training and add to csv
    with open("training.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        for item in training:
            row = [item[0]] + item[1]
            writer.writerow(row)


def gen_testing(uservid, othervid):
    testing = []
    # process uservid, othervid into images (1024 numbers, 0-255), add to testing list with ids
    gen_frame_data(uservid, testing, 1)
    gen_frame_data(othervid, testing, 0)
    return testing

def gen_testing_csv(uservid, othervid):
    testing = gen_testing(uservid, othervid)
    # process testing and add to csv
    with open("testing.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        for item in testing:
            row = [item[0]] + item[1]
            writer.writerow(row)

import matplotlib.pyplot as plt
def display_image(x):
    x = np.array(x)
    image = x.reshape((32, 32))
    image = image * 255
    plt.imshow(image, cmap='gray')
    plt.show()

gen_training_csv("uservid_diverse.MOV", "othervid_diverse.MOV")
#gen_testing_csv("uservid_testing.MOV", "othervid_testing.MOV")
