import sys
import math
import arff
import random
#import time
import matplotlib.pyplot as plt
import numpy

def parse_file(filename):
    trainingData = arff.load(open(filename,'rb'))
    trainingInstances = trainingData['data']
    attributes = trainingData['attributes']
    return attributes,trainingInstances

def initialiseWeights(attributes):
    instanceSize = len(attributes) - 1
    weights = [0.1]*instanceSize
    return weights

def findInstanceOutput(weights,trainingInstance,bias):
    sum = 0
    for weight in weights:
        index = weights.index(weight)
        sum = sum+(weight*trainingInstance[index])
    sum = sum+bias
    denominator = 1+ math.exp(-sum)
    sigmoid = 1/denominator
    return sigmoid

def updateWeights(weights,trainingInstances,bias,eta):
    y = 0
    for instance in trainingInstances:
        output = findInstanceOutput(weights,instance,bias)
        if instance[-1]== 'Mine':
            y = 1
        else:
            y = 0
        gradient_descent = (y - output)*output*(1-output)
        bias = float(bias )+ eta*gradient_descent
        for weight in weights:
            index = weights.index(weight)
            x = instance[index]
            delta_weight = eta*gradient_descent*x
            weights[index] = weights[index]+delta_weight
    return bias,weights



def main():
    argumentList=(sys.argv)
    filename = argumentList[1]
    k = int(argumentList[2])
    eta = float(argumentList[3])
    epoch = int(argumentList[4])
    bias = 0.1
    attributes,trainingSet = parse_file(filename)
    weights = initialiseWeights(attributes)
    kfold_subsamples = create_k_subsmaples(k,trainingSet,attributes)
    bias_list, weight_list = create_learning_model(kfold_subsamples,weights,bias,eta,epoch,k)
    accuracy = get_accuracy(trainingSet,kfold_subsamples,weight_list,bias_list,attributes)
    #-------------plotting accuracies---------------------------
    # ten_fold_subsamples = create_k_subsmaples(10,trainingSet,attributes)
    # epoch_list = [1,10,100,1000]
    # y1values = []
    # y2values = []
    # for epoch in epoch_list:
    #     plot_bias_list,plot_weight_list = create_learning_model(ten_fold_subsamples,weights,bias,0.1,epoch,10)
    #     test_set_accuracy_list = []
    #     training_set_accuracy_list = []
    #     sum_test_set_accuracy = 0
    #     sum_training_set_accuracy = 0
    #     for subsets in ten_fold_subsamples:
    #         fold = ten_fold_subsamples.index(subsets)
    #         test_accuracy = find_accuracy(subsets,fold,plot_weight_list,plot_bias_list,attributes)
    #         test_set_accuracy_list.append(test_accuracy)
    #         sum_test_set_accuracy = sum_test_set_accuracy + test_accuracy
    #         training_set = []
    #         for remaining_subsets in ten_fold_subsamples:
    #             if remaining_subsets!=subsets:
    #                 training_set = training_set + remaining_subsets
    #         training_accuracy = find_accuracy(training_set,fold,plot_weight_list,plot_bias_list,attributes)
    #         training_set_accuracy_list.append(training_accuracy)
    #         sum_training_set_accuracy = sum_training_set_accuracy + training_accuracy
    #     average_test_accuracy = float(sum_test_set_accuracy)/10
    #     average_training_accuracy = float(sum_training_set_accuracy)/10
    #     y1values.append(average_test_accuracy)
    #     y2values.append(average_training_accuracy)
    #     print epoch,average_test_accuracy,average_training_accuracy
    # plt.plot(epoch_list,y1values,'r')
    # plt.plot(epoch_list,y2values,'b')
    # #plt.axis([1, 1000, 0, 1])
    # plt.xlabel('Epoch')
    # plt.ylabel('Average accuracies')
    # plt.show()

    #-------------ROC Curve---------------------------
    # ten_fold_subsamples = create_k_subsmaples(10,trainingSet,attributes)
    # plot_bias_list,plot_weight_list = create_learning_model(ten_fold_subsamples,weights,bias,0.1,100,10)
    # confidence_value_list = []
    # num_neg = 0
    # num_pos = 0
    # for test_sets in ten_fold_subsamples:
    #     fold = ten_fold_subsamples.index(test_sets)
    #     updated_weight = plot_weight_list[fold]
    #     new_bias = plot_bias_list[fold]
    #     for instance in test_sets:
    #         confidence_value,predicted_value = predict_class(updated_weight,instance,new_bias)
    #         actual_label = instance[-1]
    #         confidence_value_list.append((confidence_value,actual_label))
    #         if predicted_value==1:
    #             num_pos = num_pos+1
    #         else:
    #             num_neg = num_neg+1
    # sorted_confidence_list = sorted(confidence_value_list, key = lambda x: float(x[0]))
    # TP = 0
    # FP = 0
    # last_TP = 0
    # xvalues = []
    # yvalues = []
    # limit = len(sorted_confidence_list)
    # i = 1
    # while i < (limit):
    #     data = sorted_confidence_list[i]
    #     if data[0]!=sorted_confidence_list[i-1][0] and data[1]=='Rock' and TP > last_TP:
    #         FPR = float(FP)/float(num_neg)
    #         TPR = float(TP)/float(num_pos)
    #         xvalues.append(FPR)
    #         yvalues.append(TPR)
    #         last_TP = TP
    #     if data[1] == 'Mine':
    #         TP = TP + 1
    #     else:
    #         FP = FP + 1
    #     i = i+1
    # FPR = float(FP)/float(num_neg)
    # TPR = float(TP)/float(num_pos)
    # xvalues.append(FPR)
    # yvalues.append(TPR)
    # # print 'x coordinates', xvalues
    # # print len(xvalues)
    # # print 'y coordinate', yvalues
    # # print len(yvalues)
    # plt.plot(yvalues,xvalues,'r')
    # #plt.axis([1, 1000, 0, 1])
    # plt.xlabel('Specificity')
    # plt.ylabel('Sensitivity')
    # plt.show()

def find_accuracy(set,fold,weight_list,bias_list,attributes):
    positive_counts = 0
    updated_weights = weight_list[fold]
    new_bias = bias_list[fold]
    for instance in set:
        confidence_value,predicted_value = predict_class(updated_weights,instance,new_bias)
        actual_label = instance[-1]
        if predicted_value == 1:
            predicted_label = attributes[-1][1][1]
        else:
            predicted_label = attributes[-1][1][0]
        if predicted_label==actual_label:
            positive_counts = positive_counts+1
        accuracy = float(positive_counts)/float(len(set))
    return accuracy

def get_accuracy(trainingSet,kfold_subsamples,weight_list,bias_list,attributes):
    positive_counts = 0
    for instance in trainingSet:
        fold = find_data_fold(instance,kfold_subsamples)
        updated_weights = weight_list[fold]
        new_bias = bias_list[fold]
        confidence_value,predicted_value = predict_class(updated_weights,instance,new_bias)
        actual_label = instance[-1]
        if predicted_value == 1:
            predicted_label = attributes[-1][1][1]
        else:
            predicted_label = attributes[-1][1][0]
        if predicted_label==actual_label:
            positive_counts = positive_counts+1
        print 'fold:',fold+1,' predicted label:',predicted_label,' actual label:',actual_label,' confidence value:',confidence_value
    accuracy = float(positive_counts)/float(len(trainingSet))
    print 'correct:',positive_counts,'total:',len(trainingSet),'accuracy',accuracy
    return  accuracy

def predict_class(weights,data_instance,bias):
    output = findInstanceOutput(weights,data_instance,bias)
    return output, 1 if output > 0.5 else 0

def find_data_fold(data_instance,kfold_subsamples):
    for subsets in kfold_subsamples:
        if data_instance in subsets:
            return kfold_subsamples.index(subsets)

def create_learning_model(kfold_subsamples,weights,bias,eta,epoch,k):
    i = 0
    weight_list = []
    bias_list = []
    while i < k:
        #test_set = kfold_subsamples[i]
        j = 0
        new_trainingSet = []
        while j < k:
            if j!=i:
                new_trainingSet = new_trainingSet+kfold_subsamples[j]
            j = j + 1
        l = 0
        updated_weights = weights
        updated_bias  = bias
        while l < epoch:
            updated_bias,updated_weights = updateWeights(updated_weights,new_trainingSet,updated_bias,eta)
            l = l + 1
        bias_list.append(updated_bias)
        weight_list.append(updated_weights)
        i = i + 1
    return bias_list,weight_list

def create_k_subsmaples(k,trainingSet,attributes):
    class1 = attributes[-1][1][0] #Rock
    class2 = attributes[-1][1][1] #Mine
    total_set_size = len(trainingSet)
    class_1_set = []
    class_2_set = []
    for instance in trainingSet:
        if instance[-1] == class1:
            class_1_set.append(instance)
        else:
            class_2_set.append(instance)
    class_1_sample_size = int(len(class_1_set)/k)
    class_2_sample_size = int(len(class_2_set)/k)
    i = 0
    remaining_class_1_set = []
    remaining_class_2_set = []
    rand_smpl = [[]]*k
    while i < k:
        class1_rand_smpl,remaining_class_1_set = get_random_subset(class_1_set,class_1_sample_size)
        class2_rand_smpl, remaining_class_2_set = get_random_subset(class_2_set,class_2_sample_size)
        rand_smpl[i] = class1_rand_smpl+class2_rand_smpl
        i = i+1
    #print i
    class1_leftover_instances = remaining_class_1_set
    class2_leftover_instances = remaining_class_2_set
    random.shuffle(class1_leftover_instances)
    random.shuffle(class2_leftover_instances)
    j = 0
    n = 0
    while j < len(class1_leftover_instances) and n < k:
        rand_smpl[n].append(class1_leftover_instances[j])
        n = n+1
        j = j+1
    l = 0
    m = 0
    while l < len(class2_leftover_instances) and m < k:
        rand_smpl[m].append(class2_leftover_instances[l])
        m = m+1
        l = l+1
    return rand_smpl

def get_random_subset(set,set_size):
        rand_smpl = []
        rand_smpl_indices = random.sample(xrange(len(set)), int(set_size))
        for index in rand_smpl_indices:
            rand_smpl.append(set[index])
        for index in sorted(rand_smpl_indices,reverse=True):
            set.remove(set[index])
        return rand_smpl, set
