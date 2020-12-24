from svmutil import *
import numpy as np
import os
import random

TRAIN_PATH = f'/home/yangck/Desktop/libsvm/python/satimage.scale'
TEST_PATH = f'/home/yangck/Desktop/libsvm/python/satimage.scale.t'

def load_data(PATH,ans_class):
    with open(PATH) as f:
        dataset = []
        while True:
            input = f.readline()
            if not len(input):
                return dataset

            split_input = input[:-1].split(' ')
            groundtruth = 1 if int(split_input[0])==ans_class else -1
            feature = []
            id = 1
            for element in split_input[1:-1]:
                index,value = element.split(':')
                while(id<int(index)):
                    feature.append(float(0))
                    id += 1
                feature.append(float(value))
                id += 1

            dataset.append([np.array(feature,dtype='float64'),groundtruth])

def get_len_of_w(model):
    sv = model.get_SV()
    coef = model.get_sv_coef()
    w = np.array([0 for _ in range(36)],dtype='float64')
    for i in range(len(sv)):
        a_sv = []
        for j in range(1,37):
            try:
                a_sv.append(sv[i][j]*coef[i][0])
            except Exception as e:
                a_sv.append(0.0)

        w += np.array(a_sv,dtype='float64')

    l = 0.0
    for i in range(len(w)):
        l += pow(w[i],2)

    return pow(l,0.5)



# 15
# train_dataset = load_data(TRAIN_PATH,3)
# # print(len(train_dataset))
# y = [data[1] for data in train_dataset]
# x = [data[0] for data in train_dataset]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 0 -c 10 -q')
# model = svm_train(prob, param)
# print(get_len_of_w(model))


# 16
# train_dataset = load_data(TRAIN_PATH,5)
# y = [data[1] for data in train_dataset]
# x = [data[0] for data in train_dataset]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 1 -d 2 -c 10 -q -r 1 -g 1')
# model = svm_train(prob, param)
# p_labs, p_acc, p_vals = svm_predict(y, x, model)
# print(100-p_acc[0])


# 17
# train_dataset = load_data(TRAIN_PATH,5)
# y = [data[1] for data in train_dataset]
# x = [data[0] for data in train_dataset]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 1 -d 2 -c 10 -q -r 1 -g 1')
# model = svm_train(prob, param)
# print(len(model.get_SV()))


# 18/19
# train_dataset = load_data(TRAIN_PATH,6)
# y = [data[1] for data in train_dataset]
# x = [data[0] for data in train_dataset]
# test_dataset = load_data(TEST_PATH,6)
# test_y = [data[1] for data in test_dataset]
# test_x = [data[0] for data in test_dataset]
# prob = svm_problem(y, x)
# param = svm_parameter('-t 2 -g 1000 -c 0.1 -q')
# model = svm_train(prob, param)
# p_labs, p_acc, p_vals = svm_predict(test_y, test_x, model)
# print(100-p_acc[0])


# 20

dict = {'0.1':0,'1':0,'10':0,'100':0,'1000':0}

train_dataset = load_data(TRAIN_PATH,6)
for _ in range(1000):
    random.shuffle(train_dataset)
    eval_x = []
    eval_y = []
    for i in range(200):
        eval_y.append(train_dataset[i][1])
        eval_x.append(train_dataset[i][0])

    train_x = []
    train_y = []
    for i in range(200,len(train_dataset)):
        train_y.append(train_dataset[i][1])
        train_x.append(train_dataset[i][0])

    min_eout = 101
    best_lambda = None
    keys = list(dict.keys())
    keys.sort()
    for LAMBDA in keys:
        prob = svm_problem(train_y, train_x)
        param = svm_parameter(f'-t 2 -g {float(LAMBDA)} -c 0.1 -q')
        model = svm_train(prob, param)
        p_labs, p_acc, p_vals = svm_predict(eval_y, eval_x, model)

        if(100-p_acc[0] < min_eout):
            min_eout = 100-p_acc[0]
            best_lambda = LAMBDA

    dict[best_lambda] += 1

print(dict)


