'''
Created on Dec 2, 2014
@author: Jim D'Souza

Description : Logistic Regression classifier that predicts the time it takes to sell a property on the US real estate market.
Have about 50k data points. Split into 35k training, 5k testing and 10k validation data sets.
Accuracy of about 75%, which is below that of random forests and SGD.
Data sets not added to repository because of confidentiality.
'''

import xlrd
import csv
from openpyxl import load_workbook

import numpy
from numpy import linalg

import sklearn
from sklearn import tree
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
from sklearn.preprocessing import OneHotEncoder

import DecisionTree

import random
import datetime
import pydot

### Global variables ###
original_file_path = "D:\\Move\\EDWData.csv"
base_file_path_1 = "D:\\Move\\Input\\EDWData_flags.csv"
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
    
	# Keeping track of running time
    t0 = datetime.datetime.now()
    
	# This uses TF-IDF to identify strings that are most relevant to a Property Description, and then create dummy vars out of them
    #property_flags()
    
    print "Starting time : ", t0
	
    ### Import DERIVED sheet from base data file - format csv ###
    in_file = csv.reader(open(base_file_path_1,"r"), delimiter = ",")
    base_data = []
    x=list(in_file)
    base_data = numpy.array(x)
	
	# Separate data points and headers for simplicity    
    header = base_data[0]
    base_data = base_data[1:]
    
    ### Import DERIVED sheet from base data file - format csv ###
	### Testing phase - commented out ###
    #in_file = csv.reader(open(base_file_path_2,"r"), delimiter = ",")
    #base_data = []
    #x=list(in_file)
    #test_data = numpy.array(x)
    
    #test_data = test_data[1:]
    
    ### Split the base data into training and testing data sets ###
    ### 35K Training and 5K Testing ###
    ### Use Random function to select 5K Testing data points ###
    ### NOTE : Afterwards, create a K-Fold training-testing split; will calculate error better ###
    random.shuffle(base_data)
    train_data = base_data[:35000]
    test_data  = base_data[35001:]
    
    print "File split into training and testing data sets."
    
    ### Separate train and test data sets into ID and FEATURES set ###
    ### Currently not keeping categorical variables number 3 4 and 5 ###
    ### Use ONE HOT ENCODING later to include these in the decision tree ###
    train_id      = train_data[:,0]
    #train_feature = train_data[:,1:98].astype('float')
    #train_result  = train_data[:,99]
    train_feature = train_data[:,1:363].astype('float')
    train_result  = train_data[:,364]
    test_id       = test_data[:,0]
    #test_feature  = test_data[:,1:98].astype('float')
    #test_result   = test_data[:,99]
    test_feature  = test_data[:,1:363].astype('float')
    test_result   = test_data[:,364]
    
    header_id = header[0]
    #header_feature = header[1:98]
    header_feature = header[1:363]
    
    print "Training and testing data sets completed."
    
    ### ONE HOT ENCODER ###
    #enc = OneHotEncoder()
    #enc.fit(train_feature[:,3:6])
    #print enc.n_values_
    #print enc.feature_indices_
    #data = enc.transform(train_feature[:,3:6])
    
    #print data
    
    
    ### Create a Decision tree classifier on the training data set ###
    ### Then fit the Decision tree on the test data set ###
    #clf = RandomForestClassifier(n_estimators=20,max_depth=100,compute_importances=True,verbose=True)
    clf = LogisticRegression(penalty='l1',C=0.1,class_weight={'A':1.0,'B':1.3,'C':0.9,'D':3.0,'E':3.0},tol=0.0001)
    #clf = SGDClassifier(class_weight=None, learning_rate='optimal', 
    #                    loss='log', n_iter=100, penalty='l1', verbose=1)
    clf = clf.fit(train_feature,train_result)
    
    print "Fitting completed."
    
	### Predict the outcomes for the test data set, and save them into a file ###
    preds = clf.predict_proba(test_feature)
    coefs = clf.coef_
    #print coefs
    
    print "Prediction completed."
    
	### Multinomial logistic regression shows probabilities for each outcome class (5 in this case) ###
	### The result is the class with the highest probability ###
    output = []
    #output.append(['ID','Preds','Results'])
    output.append(['ID','Preds_A','Preds_B','Preds_C','Preds_D','Preds_E','Results','Pred'])
    for i in range(len(preds)):
        output_row = []
        output_row.append(test_id[i])
        
        output_row.append(preds[i][0])
        output_row.append(preds[i][1])
        output_row.append(preds[i][2])
        output_row.append(preds[i][3])
        output_row.append(preds[i][4])
        
        output_row.append(test_result[i])
        
        m = max(preds[i])
        if m == preds[i][0] :
            output_row.append('A')
        elif m == preds[i][1] :
            output_row.append('B')
        elif m == preds[i][2] :
            output_row.append('C')
        elif m == preds[i][3] :
            output_row.append('D')
        elif m == preds[i][4] :
            output_row.append('E')
        
        output.append(output_row)
    
    with open(output_file, 'wb') as op:
        writer = csv.writer(op, delimiter=',')
        writer.writerows(output)
    
    print "Output file saved."

if __name__ == '__main__':
	# Go to main function in line 120
    main()