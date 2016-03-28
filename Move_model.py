'''
Created on Dec 5, 2014
@author: Jim D'Souza

Description : Random Forest classifier that predicts the time it takes to sell a property on the US real estate market.
Have about 50k data points. Split into 35k training, 5k testing and 10k validation data sets.
Accuracy of about 80%, which beats logistic regression and SGD
Uses conditional probability to identify which variables can be added by the user to improve the selling time
'''

import csv
import json

import numpy
import pandas as pd
from pandas import *

import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.externals import joblib

import math
import random
import datetime

### Global variables ###
original_file_path = "D:\\Move\\EDWData.xlsx"
base_file_path_1 = "D:\\Move\\Input\\EDWData_flags.csv"

test_file_path_1 = "D:\\Move\\Input\\EDWData_test.xlsx"
test_file_path_2 = "D:\\Move\\Input\\EDWData_test.csv"
test_file_path_3 = "D:\\Move\\Input\\EDWData_test2.csv"

column_names_path = "D:\\Move\\Input\\column_names.csv"
listing_style_path = "D:\\Move\\Input\\EDWData_listingstyle.csv"
zip_path = "D:\\Move\\Input\\EDWData_zip.csv"

decision_tree_file = "D:\\Move\\Output\\dt_model.joblib.pkl"
output_path = "D:\\Move\\Output\\decision_tree.dot"
output_file = "D:\\Move\\Output\\dt_op.csv"
output_file_json = "D:\\Move\\Output\\dt_op.json"

property_output_file = "D:\\Move\\TextMining\\property_op.csv"
words1_path = "D:\\Move\\TextMining\\neg_txt_freq1.csv"
words2_path = "D:\\Move\\TextMining\\neg_txt_freq2.csv"
words3_path = "D:\\Move\\TextMining\\neg_txt_freq3.csv"

### Format the input data set, perform data cleaning, and convert to numpy ###
def format_to_numpy(in_file):
    
    in_file = in_file.fillna(0)
    
    # Step 1) Import Column names that will be needed in the final file
    column_file = csv.reader(open(column_names_path,"r"), delimiter = ",")
    rownum = 0
    for row in column_file :
        if rownum == 0 :
            column_headers = row
        elif rownum == 1 :
            word_list = row
        elif rownum == 2 :
            house_style_list = row
        rownum += 1
   
    # Step 2) Import ListingStyle and Zip data that will be merged with the original file to create new columns
    listing_styles = pd.read_csv(listing_style_path) 
    zip_data = pd.read_csv(zip_path)
    
    header_list_1 = list(in_file.columns.values)
    
    # Step 3) Reduce the number of columns to only those which are needed
    for header in header_list_1 :
        if header != "ListingStyle" and header != "ListingStartDate" and header != "PropertyDescription" and header != "ListingPostalCode" :
            if header not in column_headers :
                in_file = in_file.drop(header, 1)
    
    # Step 4) Add the date related columns
    in_file["ListingDayofMonth"] = in_file["ListingStartDate"].apply(lambda x:pd.to_datetime(x).day)
    in_file = in_file.drop("ListingStartDate", 1)
    
    
    # Step 5) Add the price per sq ft
    in_file["ListingPricePerSqFt"] = in_file["ListingOriginalPrice"]/in_file["ListingSquareFeet"]
    in_file["ListingPricePerSqFt"] = in_file["ListingPricePerSqFt"].apply(lambda x: 0 if x == float('Inf') else x)
    
    
    # Step 6) Add the zip level data columns
    in_file["Zip"] = in_file["ListingPostalCode"].apply(lambda x:str(x)[:3])
    in_file = in_file.drop("ListingPostalCode", 1)
    
    in_file = pandas.merge(in_file,zip_data,on="Zip",how='left')
    in_file = in_file.drop("Zip", 1)

    in_file["flag_ListingPriceperSqFt"] = in_file["ListingPricePerSqFt"] - in_file["Zip_PricePerSqFt"]
    in_file["flag_ListingPriceperSqFt"] = in_file["flag_ListingPriceperSqFt"].apply(lambda x: 1 if x >= 0 else 0)
    
    
    
    # Step 7) Add the new Listing Styles
    in_file = pandas.merge(in_file,listing_styles,on="ListingStyle",how='left')
    
    in_file["ListingStyle_New"] = in_file["ListingStyle_New"].apply(lambda x: "Other" if isnull(x) else x)
    in_file = in_file.drop("ListingStyle", 1)
    in_file["ListingStyle"] = in_file["ListingStyle_New"]
    in_file = in_file.drop("ListingStyle_New", 1)
    
    
    # Step 8) Add the flags for Listing Styles
    for ls in house_style_list :
        if ls != "" :
            in_file["flag_"+ls] = in_file["ListingStyle"].apply(lambda x:1 if x==ls else 0)
    
    in_file = in_file.drop("ListingStyle", 1)
        
        
    # Step 9) Add the PropertyDescription Related Columns
    in_file["PropertyDescriptionLength"] = in_file["PropertyDescription"].apply(lambda x:len(str(x)))
    
    
    # Step 10) Add the flags for WORDS in Property Description
    for word in word_list :
        if word != "" :
            in_file["flag_"+word] = in_file["PropertyDescription"].str.contains("(?i)"+word)
            in_file["flag_"+word] = in_file["flag_"+word].apply(lambda x: 1 if x==True else 0)
    
    in_file = in_file.drop("PropertyDescription", 1)        
    
    # Dummy columns - not needed for analysis
    in_file["ListingLength"] = 0
    in_file["ListingLengthBuckets"] = 0
        
    
    in_file = in_file[column_headers]
    #in_file.to_csv("D:\\Move\\Input\\check.csv", sep=',', encoding='utf-8')  
    
    base_data = in_file.values

    return base_data, list(in_file.columns.values)  
    
### Used TF-IDF ot identify strings that are relevant to property descriptions. Then created dummy vars out of these strings ###
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
    
### The following functions were used to calculate the conditional probability ###
### of the effect that including a particular variable would have on the selling time of the property ###
def numerator(bayes,result):
    prob_count  = bayes["Result"][bayes["Result"]==result].count()
    total_count = bayes["Result"].count()
    
    return float(prob_count)/float(total_count)

def denominator_1(bayes,index,result):
    
    total_count = bayes["Result"][bayes["Result"]==result].count()
    
    index_2 = bayes[bayes["Result"]==result].index.tolist()
    index_new = list(set(index).intersection(index_2))
        
    prob_count = len(index_new)
    
    return float(prob_count)/float(total_count)

def denominator_2(bayes,index,result):
    
    total_count = bayes["Result"][bayes["Result"]!=result].count()
    
    index_2 = bayes[bayes["Result"]!=result].index.tolist()
    index_new = list(set(index).intersection(index_2))
        
    prob_count = len(index_new)
    
    return float(prob_count)/float(total_count)

def probability_of_class(bayes,result,index,rownum):
    
    probs = {}
    
    # 1) P(A=output is of a certain class) = count of output in that class/total count
    probs["A"] = numerator(bayes,result)
        
    # 2) P(B=certain features select | A=output is of a certain class)
    probs["B|A"] = denominator_1(bayes, index, result)
              
    # 3) P(B=certain features select | !A=output is not of a certain class)
    probs["B|!A"] = denominator_2(bayes,index, result)
        
    # 4) Overall Probability of falling in particular class, given selected features
    if probs["B|A"] == 0 and probs["B|!A"] == 0 :
        prob = 0.0
    else :
        prob = (probs["A"]*probs["B|A"])/((probs["A"]*probs["B|A"]) + ((1.0-probs["A"])*probs["B|!A"]))   
    return prob

### This function displays comments regarding the properties selling time, and what can be done to improve it ###
def add_comments(var) :
    
    if var == "PropertyDescriptionLength" :
        comments = "Consider adding a property description, to reduce the DoM."
    elif var == "ListingOriginalPrice" :
        comments = "Consider adding a property price, to reduce the DoM."
    elif var == "ListingNumberOfBedrooms" :
        comments = "Consider adding the number of bedrooms, to reduce the DoM."
    elif var == "ListingNumberOfStories" :
        comments = "Consider adding the number of stories, to reduce the DoM."
    elif var == "ListingSquareFeet" :
        comments = "Consider adding the square foot area of the property, to reduce the DoM."
    elif var == "ListingPricePerSqFt" :
        comments = "Consider adding the square foot area of the property and the property price, to reduce the DoM."
    elif var == "ListingPhotoCount" :
        comments = "Consider adding a photo of the property, to reduce the DoM."
    elif var == "IsForeclosure" :
        comments = "Consider adding details about the status of the property."
    elif var == "IsPriceDropped" :
        comments = "Consider adding details about the price status of the property."
    elif var == "IsPriceIncreased" :
        comments = "Consider adding details about the price status of the property."
    elif var == "IsRental" :
        comments = "Consider adding details about the rental status of the property."
    elif var == "IsDisplayed" :
        comments = "Consider changing the isDiplayed value."
    elif var == "IsVideo" :
        comments = "Consider adding a video of the property, to reduce the DoM."
    elif var == "IsVirtualTour" :
        comments = "Consider adding a virtual tour of the property, to reduce the DoM."
    elif var == "IsOpenHouse" :
        comments = "Consider adding an Open House for the property, to reduce the DoM."
    elif var == "HasCoBroke" :
        comments = "Consider adding a Co-Broker for the property, to reduce the DoM."
    elif var == "flag_ListingPriceperSqFt" :
        comments = "Consider reducing the Price/Sq.Ft. to average levels of your ZIP."
    elif var == "None" :
        comments = "This is the expected DoM of your property in the listed ZIP."

    return comments

### Main function ###
def main(input_mode):
    
	### Keeping track of running time ###
    t0 = datetime.datetime.now()
    
    #property_flags()
    
    print "Starting time : ", t0
    ### Import DERIVED sheet from base data file - format csv ###
    
	### Input mode 1 denotes testing phase ###
    if input_mode == 1 :
        testing = 1
        if testing == 0 :
            in_file = pandas.io.excel.read_excel(original_file_path,sheetname="Original")
            base_data, header = format_to_numpy(in_file)
        else :
            in_file = csv.reader(open(base_file_path_1,"r"), delimiter = ",")
            base_data = []
            x=list(in_file)
            base_data = numpy.array(x)
    
            header = base_data[0]
            base_data = base_data[1:]
    
	### Input mode 0 denotes running phase ###
    if input_mode == 0 :
        
        # Part 1 - reading in the original data
        in_file = csv.reader(open(base_file_path_1,"r"), delimiter = ",")
        base_data = []
        x=list(in_file)
        base_data = numpy.array(x)
    
        header = base_data[0]
        base_data = base_data[1:]

        # Part 2 - reading in the test file
        testing = 0
        if testing == 0 :
            #in_file = pandas.io.excel.read_excel(test_file_path_1,sheetname="Original")
            in_file = pandas.read_csv(test_file_path_3)
            test_data, header = format_to_numpy(in_file)
        else :
            in_file = csv.reader(open(test_file_path_2,"r"), delimiter = ",")
            test_data = []
            x=list(in_file)
            test_data = numpy.array(x)
            test_data = test_data[1:]

    
    ### Split the base data into training and testing data sets ###
    ### 17K Training and 5K Testing ###
    ### Use Random function to select 5K Testing data points ###
    ### NOTE : Afterwards, create a K-Fold training-testing split; will calculate error better ###
    if input_mode == 1 :
        random.shuffle(base_data)
        train_data = base_data[:35000]
        test_data  = base_data[35001:]
    else :
        train_data = base_data
    
    print "File split into training and testing data sets : ", datetime.datetime.now() - t0
    
    ### Separate train and test data sets into ID and FEATURES set ###
    train_id      = train_data[:,0]
    train_feature = train_data[:,1:359].astype('float')
    train_result  = train_data[:,360]

    test_id       = test_data[:,0]
    test_feature  = test_data[:,1:359].astype('float')
    test_result = test_data[:,360]
    
    header_id = header[0]
    header_feature = header[1:359]
    
    print "Training and testing data sets completed : ", datetime.datetime.now() - t0
    
    
    
    ### Create a Decision tree classifier on the training data set ###
    ### Then fit the Decision tree on the test data set ###
    #clf = RandomForestClassifier(n_estimators=20,max_depth=100,compute_importances=True,verbose=True)
    if input_mode == 1 :
        clf = tree.DecisionTreeClassifier(max_depth=200,compute_importances=True)
        clf = clf.fit(train_feature,train_result)
    
        # Store the Decision tree at an external location
        # This can later be used to compute predictions
        print ("Writing Decision Tree Model to: ", decision_tree_file)
        _ = joblib.dump(clf, decision_tree_file, compress = 9)
    else :
        print ("Importing model file...")
        clf = joblib.load(decision_tree_file)
    
    preds = clf.predict(test_feature)
    
    print "Decision tree created : ", datetime.datetime.now() - t0
    
    # Calculate the feature ranking
    if input_mode == 1 :
        print "Calculating feature importance..."
    
        important_features = []
        for x,i in enumerate(clf.feature_importances_):
            if i>numpy.average(clf.feature_importances_):
                important_features.append(str(x))
        
        importances = clf.feature_importances_
        indices = numpy.argsort(importances)[::-1]
    
        print("Feature ranking:")
        important_names = []
        for f in range(len(header_feature)):
            important_names.append(header_feature[indices[f]])
            print("%d. feature %s (%f)" % (f + 1, header_feature[indices[f]], importances[indices[f]])), " : ", indices[f]
    
    print "Feature ranking completed  : ", datetime.datetime.now() - t0
       
    
    # Export decision tree
    #if input_mode == 1:
    #    tree.export_graphviz(clf, out_file=output_path, feature_names=header_feature)
    #    print "Decision tree exported  : ", datetime.datetime.now() - t0
       
    
    print "Creating the Bayesian model  : ", datetime.datetime.now() - t0
    
    # Probability of a data point belonging in a particular class can be expressed as a function of the probabilities...
    # of events in certain features
    # i.e. given the presence of certain features, would the addition of an extra feature contribute to the probability...
    # of the data point having a shorter DoM?
    
    # P(A|B) =           P(A) x P(B|A) / [ P(A) x P(B|A)   +   P(!A) x P(B|!A) ]
    # Here, event A is the classification into a higher class of DoM
    # event B is the selection of a certain set of features
    
    bayes_feature = train_feature[:,0:19]
    bayes_header = header_feature[0:19]
    
    bayes = pd.DataFrame(bayes_feature,columns=bayes_header)
    bayes["Result"] = train_result
    
    
    bayes_feature_2 = test_feature[:,0:19]
    bayes_header_2 = header_feature[0:19]
    
    bayes_2 = pd.DataFrame(bayes_feature_2,columns=bayes_header_2)
    bayes_2["Result"] = test_result
    
    # Switch flag for listing price per sq ft - 1 means less than avg, 0 means greater
    bayes["flag_ListingPriceperSqFt"] = bayes["flag_ListingPriceperSqFt"].apply(lambda x: 1 if x==0 else 0)
    bayes_2["flag_ListingPriceperSqFt"] = bayes_2["flag_ListingPriceperSqFt"].apply(lambda x: 1 if x==0 else 0)
    rownum = 0
    
    variable_list = []
    if input_mode == 1 :
        loop_length = len(bayes_2)
    else :
        loop_length = 1
    
    while rownum < loop_length :
        result = bayes_2["Result"][rownum]
        prediction = preds[rownum]
        
        if result != "A" :
            print "\n", rownum ," \n Actual prediction = ", prediction
            
            # This list is used to store variables that should be changed in order to move the prediction to a higher DoM class
            variable_row = []
            
            ### First, calculate probability of falling in a current class, based on the given features
            # This dictionary stores the variable and the value 
            # 1 - if the variable is part of the data points variable set
            # 0 - if the variable is not a part of it
            variable_select = {}
            
            for i in bayes_2 :
                if bayes_2[i][rownum] > 0 :
                    variable_select[i] = 1
                else :
                    variable_select[i] = 0
            
            # The index selects only those data points from the training data set, where the variables from...
            # variable_select are set to 1
            index = bayes.index.tolist()
            for i in bayes :
                if variable_select[i] > 0 and i != "Result":
                    index_new = bayes[bayes[i] > 0].index.tolist()
                    index = list(set(index).intersection(index_new))
            
            # Result_new is the DoM class that is immediately above the current DoM class prediction
            if input_mode == 1 :
                if result == "B" :
                    result_new = "A"
                elif result == "C" :
                    result_new = "B"
                elif result == "D" :
                    result_new = "C"
                elif result == "E" :
                    result_new = "D"
            
            else :
                if prediction == "B" :
                    result_new = "A"
                elif prediction == "C" :
                    result_new = "B"
                elif prediction == "D" :
                    result_new = "C"
                elif prediction == "E" :
                    result_new = "D"
            
            # First we calculate the probability of a data point falling in a higher DoM class, given the current variable set
            prob = probability_of_class(bayes,result_new,index,rownum)
            print " ", prob*100.0, "% probabiity of data point : ", rownum, " falling in output class : ", result_new
            
            ### Now, find the probability of falling into a higher class, by tweaking certain variables
            orig_prob = prob
            max_prob = prob
            max_var = "None"
            for key in variable_select :
                if variable_select[key] == 0 :
                    variable_select[key] = 1
                    
                    index = bayes.index.tolist()
                    for i in bayes :
                        if variable_select[i] > 0 and i != "Result":
                            index_new = bayes[bayes[i] > 0].index.tolist()
                            index = list(set(index).intersection(index_new))
            
                    prob = probability_of_class(bayes,result_new,index,rownum)
                    if prob > max_prob :
                        max_prob = prob
                        max_var  = key
                    
                    if prob > orig_prob :
                        print " ", prob*100.0, "% of ", result_new ," after changing variable ", key
                        variable_row.append(key)
                        #variable_row.append(prob*100.0)
                        variable_row.append(add_comments(key))

                    variable_select[key] = 0
                    
            variable_list.append(variable_row)       
            print " ", max_prob*100.0, "% after changing variable ", max_var

        rownum = rownum + 1
   
    print "Bayesian probabilities calculated  : ", datetime.datetime.now() - t0
   
    ### Export the output to a file
    output = []
    if input_mode == 1 :
        output.append(['ID','Preds','Results'])      
    else :
        output.append(['ID','Preds'])    
    for i in range(len(preds)):
        output_row = []
        output_row.append(test_id[i])
        if preds[i] == "A" :
            output_row.append("00-07")
        elif preds[i] == "B" :
            output_row.append("07-15")
        elif preds[i] == "C" :
            output_row.append("15-30")
        elif preds[i] == "D" :
            output_row.append("30-90")
        elif preds[i] == "E" :
            output_row.append("90 + ")
        output_row.append(test_result[i])
        output.append(output_row)
        if input_mode == 0 :
            output.append(['Variables to change'])
            output_row_2 = []
            for vl in range(len(variable_list[i])):
                if vl%2 == 0 :
                    output_row_2.append(variable_list[i][vl])
                else :
                    output_row_2.append(variable_list[i][vl]) 
                    output.append(output_row_2)
                    output_row_2 = []
   
    ### Save the final output file to directory
    with open(output_file, 'wb') as op:
        writer = csv.writer(op, delimiter=',')
        writer.writerows(output)
    
    jsonfile = open(output_file_json, 'w')
    for row in output:
        json.dump(row, jsonfile)
        jsonfile.write('\n')
    
    print "Output file saved : ", datetime.datetime.now() - t0


if __name__ == '__main__':
    # 1 for testing i.e. split base data into training and testing
    # 0 for single line input i.e. train on entire base data, and classify the data point individually
    input_mode = 0
    
    main(input_mode)