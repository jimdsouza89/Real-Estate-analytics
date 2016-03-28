'''
Created on Dec 3, 2014
@author: Jim D'Souza

Description : Experiments with Neural networks. Not completed yet.
'''

import xlrd
import csv
from openpyxl import load_workbook

import numpy

import pybrain
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import TanhLayer

import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO


import random
import datetime
import pydot

### Global variables ###
original_file_path = "D:\\Move\\EDWData.csv"
base_file_path_1 = "D:\\Move\\Input\\EDWData_flags_NN.csv"
base_file_path_2 = "D:\\Move\\Input\\EDWData_test.csv"
output_path = "D:\\Move\\Output\\decision_tree.dot"
output_file = "D:\\Move\\Output\\dt_op.csv"
property_output_file = "D:\\Move\\TextMining\\property_op.csv"
words1_path = "D:\\Move\\TextMining\\neg_txt_freq1.csv"
words2_path = "D:\\Move\\TextMining\\neg_txt_freq2.csv"
words3_path = "D:\\Move\\TextMining\\neg_txt_freq3.csv"

def property_flags ():   
    t0 = datetime.datetime.now()
    print "Starting time : ", t0
    ### Import DERIVED sheet from base data file - format csv ###
    in_file = csv.reader(open(original_file_path,"r"), delimiter = ",")
    property_data = []
    for row in in_file :
        property_data.append(row)
    #x=list(in_file)
    #base_data = numpy.array(x)
    
    print "File imported.", datetime.datetime.now() - t0
    
    ### Adding columns to the base file ###
    ### These columns are created from selecting strings from the Property Descriptions ###
    ### Flags of either 0 or 1 indicate whether the string exists in the Listing or not ###
    # 3 WORD STRINGS
    in_file = csv.reader(open(words3_path,"r"), delimiter = ",")
    words3 = []
    rownum = 0
    for row in in_file :
        if rownum > 0 :
            if int(row[1]) >= 100 :
                words3.append(row[0].strip())
        rownum += 1
    # 2 WORD STRINGS
    in_file = csv.reader(open(words2_path,"r"), delimiter = ",")
    words2 = []
    rownum = 0
    for row in in_file :
        if rownum > 0 :
            if int(row[1]) >= 500 :
                words2.append(row[0].strip())
        rownum += 1
    # 1 WORD STRINGS
    in_file = csv.reader(open(words1_path,"r"), delimiter = ",")
    words1 = []
    rownum = 0
    for row in in_file :
        if rownum > 0 :
            if int(row[1]) >= 1000 :
                words1.append(row[0].strip())
        rownum += 1
    print "Word freq imported.", datetime.datetime.now() - t0
    
    # Creating columns for each word
    for rownum in range(len(property_data)) :
        if rownum == 0 :
            for word in words1 :
                property_data[rownum].append("flag_"+word)
            for word in words2 :
                property_data[rownum].append("flag_"+word)
            for word in words3 :
                property_data[rownum].append("flag_"+word)
            print "Headers created.", datetime.datetime.now() - t0
        
        else :
            print rownum, " - ", datetime.datetime.now() - t0
            prop_desc = property_data[rownum][3].upper()
            for word in words1 :
                if prop_desc.find(word.upper()) >= 0:
                    property_data[rownum].append(1)
                else :
                    property_data[rownum].append(0)
            for word in words2 :
                if prop_desc.find(word.upper()) >= 0:
                    property_data[rownum].append(1)
                else :
                    property_data[rownum].append(0)
            for word in words3 :
                if prop_desc.find(word.upper()) >= 0:
                    property_data[rownum].append(1)
                else :
                    property_data[rownum].append(0)  
                    
        rownum += 1
    
    with open(property_output_file, 'wb') as op:
        writer = csv.writer(op, delimiter=',')
        writer.writerows(property_data)     
    print "Output file saved.", datetime.datetime.now() - t0

def main():
    
    t0 = datetime.datetime.now()
    
    #property_flags()
    
    print "Starting time : ", t0
    ### Import DERIVED sheet from base data file - format csv ###
    in_file = csv.reader(open(base_file_path_1,"r"), delimiter = ",")
    base_data = []
    x=list(in_file)
    base_data = numpy.array(x)
    
    header = base_data[0]
    base_data = base_data[1:]
    
    ### Import DERIVED sheet from base data file - format csv ###
    #in_file = csv.reader(open(base_file_path_2,"r"), delimiter = ",")
    #base_data = []
    #x=list(in_file)
    #test_data = numpy.array(x)
    
    #test_data = test_data[1:]
    
    ### Split the base data into training and testing data sets ###
    ### 17K Training and 5K Testing ###
    ### Use Random function to select 5K Testing data points ###
    ### NOTE : Afterwards, create a K-Fold training-testing split; will calculate error better ###
    random.shuffle(base_data)
    train_data = base_data[:35000]
    test_data  = base_data[35001:]
    
    print "File split into training and testing data sets."
    
    ### Separate train and test data sets into ID and FEATURES set ###
    ### Currently not keeping categorical variables number 3 4 and 5 ###
    ### Used ONE HOT ENCODING to include these  ###
    train_id      = train_data[:,0]
    train_feature = train_data[:,1:363].astype('float')
    train_result  = train_data[:,364].astype('float')

    test_id       = test_data[:,0]
    test_feature  = test_data[:,1:363].astype('float')
    test_result   = test_data[:,364].astype('float')

    
    header_id = header[0]
    header_feature = header[1:363]
    #header_feature = header[1:600]
    
    
    train_all = ClassificationDataSet(len(train_feature[0]), nb_classes=5)
    for x_i, y_i in zip(train_feature, train_result):
        train_all.appendLinked(x_i, y_i)
    train_all._convertToOneOfMany()
        
    
    test_all = ClassificationDataSet(len(test_feature[0]), nb_classes=5)
    for x_i, y_i in zip(test_feature, test_result):
        test_all.appendLinked(x_i, y_i)
    test_all._convertToOneOfMany()
    
    print "Training and testing data sets completed : ", datetime.datetime.now() - t0
    
    ### Create a Neural network classifier on the training data set ###
    ### Then fit the NN on the test data set ###
    fnn = buildNetwork(train_all.indim, 300, train_all.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(fnn, dataset=train_all, momentum=0.01, verbose=True, weightdecay=0.01, learningrate=0.01)
    
    print "Neural Network created : ", datetime.datetime.now() - t0
    
    trainer.train()    
    #trnresult = percentError( trainer.testOnClassData(),train_all['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=test_all ), test_all['target'] )

    print "epoch: %4d" % trainer.totalepochs, "  test error: %5.2f%%" % tstresult, " : ", datetime.datetime.now() - t0
        
    preds = fnn.activateOnDataset(test_all)
    # the highest output activation gives the class
    preds = preds.argmax(axis=1)
    #out = out.reshape(test_feature.shape)
        
    
    output = []
    #output.append(['ID','Preds','Results'])
    output.append(['ID','Preds','Results'])
    for i in range(len(preds)):
        output_row = []
        output_row.append(test_id[i])
        output_row.append(preds[i])
        output_row.append(test_result[i])            
        output.append(output_row)
    
    with open(output_file, 'wb') as op:
        writer = csv.writer(op, delimiter=',')
        writer.writerows(output)
    
    
    print "Output file saved."


if __name__ == '__main__':
    main()