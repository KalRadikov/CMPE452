'''====================================================================================================================
 |   Assignment 1: Perceptron and Adaline Learning in Neural Networks
 |
 |   Author:  Kaloyan Radikov 10157529
 |
 |   Due Date:  Oct 8th 2016
 |
 *=================================================================================================================='''

# ----------------------------------------- Global vars and imports -------------------------------------------------- #

import random

bias = 1.0
learningRate = 0.1
maxIteration1 = 100
maxIteration2 = 1000
theta = 0.0

# ---------------------------------- Defining Basic Functions to be used in the code --------------------------------- #

def activationFire(data, element, w1, w2, w3, w4, w5, theta):
    output = (w1 * (data[element][0]) + w2 * (data[element][1]) + w3 * (data[element][2]) + w4 * (
        data[element][3]) + w5 * bias)
    if output >= theta:
        return 1
    else:
        return 0

# ---------------------------------- Importing Data for Perceptron Function ---------------------------------- #

f = open('train.txt', 'r')
g = open('test.txt', 'r')

count = 0
data = {}
data2 = {}
data3 = {}
data4 = {}

while True:
    y = f.readline()
    if y == '':
        break
    if(y.split(',')[4].strip() == 'Iris-setosa'):
        data[count] = [float((y.split(',')[0])),float(y.split(',')[1]),float(y.split(',')[2]),float(y.split(',')[3]), 0]
    else:
        data[count] = [float((y.split(',')[0])), float(y.split(',')[1]), float(y.split(',')[2]), float(y.split(',')[3]), 1]

    count = count + 1
f.close()

count=0

while True:
    z = g.readline()
    if z == '':
        break
    if(z.split(',')[4].strip() == 'Iris-setosa'):
        data2[count] = [float((z.split(',')[0])),float(z.split(',')[1]),float(z.split(',')[2]),float(z.split(',')[3]), 0]
    else:
        data2[count] = [float((z.split(',')[0])),float(z.split(',')[1]),float(z.split(',')[2]),float(z.split(',')[3]), 1]
    count = count + 1
g.close()

# ---------------------------------- Importing data for Adaline Learning Function ------------------------------------ #

f = open('train.txt', 'r')
g = open('test.txt', 'r')

count = 0

while True:
    y = f.readline()
    if y == '':
        break
    if(y.split(',')[4].strip() == 'Iris-versicolor'):
        data3[count] = [float((y.split(',')[0])),float(y.split(',')[1]),float(y.split(',')[2]),float(y.split(',')[3]),0]
    elif(y.split(',')[4].strip() == 'Iris-virginica'):
        data3[count] = [float((y.split(',')[0])),float(y.split(',')[1]),float(y.split(',')[2]),float(y.split(',')[3]),1]
    count +=1
f.close()

i=0

while True:
    y = g.readline()
    if y == '':
        break
    if (y.split(',')[4].strip() == 'Iris-versicolor'):
        i += 1
        data4[i] = [float((y.split(',')[0])), float(y.split(',')[1]), float(y.split(',')[2]), float(y.split(',')[3]), 0]
    elif (y.split(',')[4].strip() == 'Iris-virginica'):
        i += 1
        data4[i] = [float((y.split(',')[0])), float(y.split(',')[1]), float(y.split(',')[2]), float(y.split(',')[3]), 1]

g.close()

# --------------------------- Definitions of Functions that will be used for parts 1 to 6 ---------------------------- #


def flowerType_Q1 (petalLength, petalWidth):

    thetaFire = 2
    w1 = 0.5
    w2 = 2/1.5
    thetaTest = petalLength*w1 + petalWidth*w2

    if thetaTest < thetaFire:
        print("\nYes, this is a Iris-Setosa")
    else:
        print("\nNo, this is not an Iris-Setosa")


def simpleLearn_Q2(data):

    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    w3 = random.uniform(0, 1)
    w4 = random.uniform(0, 1)
    w5 = random.uniform(0, 1)
    iterate = 0

    # ------------------- Looping through to correcting weightings using simple learning ----------------------------- #

    while True:
        iterate += 1
        globalError = 0
        for element in data:
            y = activationFire(data, element, w1, w2, w3, w4, w5, theta)
            if y == 1 and data[element][4] == 0:
                w1 = w1 - (learningRate * data[element][0])
                w2 = w2 - (learningRate * data[element][1])
                w3 = w3 - (learningRate * data[element][2])
                w4 = w4 - (learningRate * data[element][3])
                w5 = w5 - (learningRate)
                globalError += 1
            elif (y== 0 and data[element][4] == 1):
                w1 = w1 + (learningRate *  data[element][0])
                w2 = w2 + (learningRate * data[element][1])
                w3 = w3 + (learningRate * data[element][2])
                w4 = w4 + (learningRate * data[element][3])
                w5 = w5 + (learningRate)
                globalError +=1
            else:
                globalError = globalError

        if ((globalError == 0) or iterate > maxIteration1):
            break

    print("\nTheta = ", "x1*", round(w1, 2), " + x2*", round(w2, 2), " + x3*", round(w3, 2), " + x4*", round(w4, 2),
              " + bias*", round(w5, 2), sep='')


def adalineLearn_Q4(data3):

    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    w3 = random.uniform(0, 1)
    w4 = random.uniform(0, 1)
    w5 = random.uniform(0, 1)
    iterate = 0

    # --------------------- Looping through to correcting weightings using adaline learning -------------------------- #

    while True:
        iterate += 1
        globalError = 0
        for element in data3:
            output = activationFire(data3, element, w1, w2, w3, w4, w5, theta)
            localError = ((data3[element][4]) - output)
            w1 = w1 + (learningRate * localError * data3[element][0])
            w2 = w2 + (learningRate * localError * data3[element][1])
            w3 = w3 + (learningRate * localError * data3[element][2])
            w4 = w4 + (learningRate * localError * data3[element][3])
            w5 = w5 + (learningRate * localError)
            globalError = globalError + (localError*localError)

        if ((globalError == 0) or iterate > maxIteration2):
            break

    testError = 0
    for vals in data4:
        val = activationFire(data4, vals, w1, w2, w3, w4, w5, theta)
        if (val != data4[vals][4]):
            testError = testError + 1

    # ----------------------------------- Printing Linear Separator Function ----------------------------------------- #

    print("\nTheta = ", "x1*", round(w1,2), " + x2*", round(w2,2)," + x3*",round(w3,2)," + x4*", round(w4,2), " + bias*", round(w5,2), sep = '')
    print("Number of iterations total:", iterate)
    print("Error Threshold :", globalError)
    print(testError, "out of 20 wrong\n")

    # ----------------------------------------Printing Output to text file ------------------------------------------- #

    text_file = open('output.txt', 'w')
    g = open('test.txt', 'r')
    count = 0
    test = {}

    print("This is the raw test Data:\n")
    text_file.write("This is the raw test Data\n")
    while True:
        y = g.readline()
        if y == '':
            break
        test[count] = [float((y.split(',')[0])), float(y.split(',')[1]), float(y.split(',')[2]),
                       float(y.split(',')[3]), str(y.split(',')[4])]
        count += 1
    g.close()
    for i in test:
        print(test[i])
        text_file.write(str(test[i]) + '\n')


    print("\nThese are the predicted flowers using adaline learning:\n")
    text_file.write("\nThese are the predicted flowers using adaline learning:\n")

    for element in data2:
        output = activationFire(data2, element, w1, w2, w3, w4, w5, theta)
        if element <= 10:
            data2[element][4] = 'Iris-setosa'
        elif (output == 0):
            data2[element][4]= 'Iris-versicolour'
        elif(output == 1):
            data2[element][4] = 'Iris-virginica'

        print(data2[element])
        text_file.write(str(data2[element]) + '\n')

    text_file.close()
    print("Output to file 'output.txt' has finished")


# ------------------------------------------- Calling all the functions ---------------------------------------------- #

flowerType_Q1(2, 5)
simpleLearn_Q2(data)
adalineLearn_Q4(data3)
