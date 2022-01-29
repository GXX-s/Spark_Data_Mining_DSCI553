## Improvement 1: Add "star rating" from the original case. 
from pyspark import SparkContext, StorageLevel, SparkConf

import itertools
import sys
import json
import csv
import time
import math
import random
import numpy as np
import pandas as pd
import pickle
from operator import add

import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def calculate_rmse(label, predictions):
    rmse_sum = 0
    for i in range(0, len(label)):
        rmse_sum += (label[i]-predictions[i])**2
    rmse_sum = rmse_sum/len(label)
    return math.sqrt(rmse_sum)


def calculate_absolute_differences(label, predictions):
    result = {'0-1': 0, '1-2':0, '2-3':0, '3-4':0, '>4':0}
    for i in range(0, len(label)):
        abs_diff = abs(label[i]-predictions[i])
        if abs_diff <= 1:
            result['0-1'] += 1
        elif abs_diff <= 2:
            result['1-2'] += 1
        elif abs_diff <= 3:
            result['2-3'] += 1
        elif abs_diff <= 4:
            result['3-4'] += 1
        else:
            result['>4'] += 1
        
    return result

class additional_features:
    def __init__(self, business, checkin, photo, review, tip, user):
        self.business_json = business
        self.checkin_json = checkin
        self.photo_json = photo
        self.review_json = review
        self.tip_json = tip
        self.user_json = user
        
    def process_user(self):
        # get fans info 
        f = open(self.user_json, 'r')
        content = f.readlines()
        f.close()
        output = {}
        for review in content:
            json_obj = json.loads(review)
            uid = json_obj['user_id']
            avg_star = json_obj['average_stars']
            user_review_count = json_obj['review_count']
            useful = json_obj['useful']
            funny = json_obj['funny']
            cool = json_obj['cool']
            fans = json_obj['fans']
            
            output[uid] = (avg_star, user_review_count, useful, funny, cool, fans)
            
        return output
        
        
        
    def process_business_star_review(self):
        # get star ratings of businesses
        f = open(self.business_json, 'r')
        content = f.readlines()
        f.close()
        output = {}
        for bus_line in content:
            json_obj = json.loads(bus_line)
            bid = json_obj['business_id']
            star = json_obj['stars']
            review_count = json_obj['review_count']
            longitude = json_obj['longitude']
            latitude = json_obj['latitude']
            output_item = (star, review_count, longitude, latitude)
            output[bid] = output_item
            
        return output
    
    def process_tips(self):
        # return two dictionaries: {uid: tip}, {bid: tip}
        uid_tip = {}
        bid_tip = {}
        f = open(self.tip_json, 'r')
        content = f.readlines()
        f.close()
        
        for line in content:
            json_obj = json.loads(line)
            bid = json_obj['business_id']
            uid = json_obj['user_id']
            if uid not in uid_tip:
                output[uid] = 1
            else:
                output[uid] += 1
            
            if bid not in bid_tip:
                output[bid] = 1
            else:
                output[bid] += 1
        return uid_tip, bid_tip
        
        
    
    
    def process_user_evaluations(self):
        # aggragate the evaluations (useful, etc.) of a user's reviews 
        f = open(self.review_json, 'r')
        content = f.readlines()
        f.close()
        output = {}
        for review in content:
            json_obj = json.loads(review)
            bid = json_obj['business_id']
            uid = json_obj['user_id']
            useful = json_obj['useful']
            funny = json_obj['funny']
            cool = json_obj['cool']
            if uid not in output:
                output[uid] = [useful, funny, cool]
            else:
                output[uid] = list(map(add, output[uid], [useful, funny, cool]) )
            
        return output

  
    

def process(line):
    splited_line= line[0].split(',')
    if len(splited_line)<3:
        splited_line.append(0.0)
    return (splited_line[0], splited_line[1], splited_line[2])

def process_prediction(line):
    splited_line= line[0].split(',')
    return ((splited_line[0], splited_line[1]), (float(splited_line[2]), float(splited_line[3])))



def get_user_bus_info(pair, star_review_info=None, user_profile=None):
    """
    pair: (uid, bid, score)
    star_review_info: {bid: (star, review_count)}
    """
    uid = pair[0]
    bid = pair[1]
    score = pair[2]
    '''
    if uid in average_rating_per_user:
        user_total = average_rating_per_user.get(uid)[1]
        avg_user_score = average_rating_per_user.get(uid)[0]/user_total
    else:
        avg_user_score = USER_AVERAGE
        user_total = 0
    
    if bid in average_rating_per_business:
        bus_total = average_rating_per_business[bid][1]
        avg_bus_score = average_rating_per_business[bid][0]/bus_total
    else:
        avg_bus_score = BUSINESS_AVERAGE
        bus_total = 0
    '''
    # star_review_info
    if star_review_info == None:
        star = 0.0
        review_count = 0.0
        longitude =0.0
        latitude = 0.0
    else:
        star = star_review_info.get(bid, [0,0,0,0])[0]
        review_count = star_review_info.get(bid, [0,0])[1]
        longitude = star_review_info.get(bid, [0,0])[2]
        latitude = star_review_info.get(bid, [0,0])[3]
        
    # user review profile
    if user_profile == None:
        user_avg_star = 0
        user_review_count = 0
        useful = 0
        funny = 0
        cool = 0
        fans = 0
    else:
        user_avg_star = user_profile.get(uid, [0,0,0,0,0,0])[0]
        user_review_count = user_profile.get(uid, [0,0,0,0,0,0])[1]
        useful = user_profile.get(uid, [0,0,0,0,0])[2]
        funny = user_profile.get(uid, [0,0,0,0,0])[3]
        cool = user_profile.get(uid, [0,0,0,0,0])[4]
        fans = user_profile.get(uid, [0,0,0,0,0])[5]
    
    feature_list = (uid, bid, star, review_count, longitude, latitude, user_avg_star, useful, funny, cool, fans, score)
    
    
    return feature_list

if __name__ == '__main__':
    
    ## settings
    
    #input_file_path_train = sys.argv[1]
    #input_file_path_validation = sys.argv[2]

    #output_file_path = sys.argv[4]
    
    #train_file = './yelp_train.csv'
    #test_file = './yelp_test.csv'
    #output_file = './task22_output.csv'
    
    
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    train_file = folder_path + '/yelp_train.csv'
    
    #aditional documents:
    business_json = folder_path + "/business.json"
    checkin_json = folder_path + "/checkin.json"
    photo_json = folder_path + "/photo.json"
    review_json = folder_path + "/review_train.json"
    tip_json = folder_path + "/tip.json"
    user_json = folder_path + "/user.json"
    

    conf = SparkConf().setMaster("local") \
            .setAppName("task1") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)
    
    start = time.time()
    
    ## 0. read the data and some pre-processing
     
    # load training data:
    uid_bid_rdd = sc.textFile(train_file).map(lambda line: line.split('\n')).map(lambda line: process(line))
    headers = uid_bid_rdd.first()
    finalRdd_train = uid_bid_rdd.filter(lambda line: line != headers)
    
    # load testing data:
    uid_bid_rdd_val = sc.textFile(test_file).map(lambda line: line.split('\n')).map(lambda line: process(line))
    headers = uid_bid_rdd_val.first()
    finalRdd_val = uid_bid_rdd_val.filter(lambda line: line != headers)
    
    # print(len(finalRdd_train))
    
    
    ## 1. Prepare the training data
    # here we set two types of features. Generate from origianl dataset: user_average, business_average, user_review_count, business_review_count
    all_info_train_map = sc.broadcast(
        finalRdd_train.map(lambda line: ((line[0], line[1]), float(line[2]))).collectAsMap()).value

    users_business_train_map_rdd = finalRdd_train.map(lambda line: (line[0], line[1])).groupByKey().mapValues(lambda line: set(line)) \
        .sortByKey().collectAsMap()
    users_business_train_map = sc.broadcast(users_business_train_map_rdd).value

    business_user_train_map_rdd = finalRdd_train.map(lambda line: (line[1], line[0])).groupByKey().mapValues(lambda line: set(line)) \
        .sortByKey().collectAsMap()

    business_user_train_map = sc.broadcast(business_user_train_map_rdd).value
    # print(business_user_train_map)

    '''
    average_rating_per_business_rdd = finalRdd_train.map(lambda line: (line[1], float(line[2]))).mapValues(lambda line: (line, 1)) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).sortByKey().collectAsMap()
    
    # the output is in the format {bus: (total_score, total_count), ...}
    average_rating_per_business = sc.broadcast(average_rating_per_business_rdd).value 
    value_list = list(average_rating_per_business.values())
    BUSINESS_AVERAGE = sum([pair[0] for pair in value_list])/sum([pair[1] for pair in value_list])
    # print(BUSINESS_AVERAGE)
    
    average_rating_per_user_rdd = finalRdd_train.map(lambda line: (line[0], float(line[2]))).mapValues(lambda line: (line, 1)) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).sortByKey().collectAsMap()
    
    # Similarly, the output is in the format {user: ({)total_score, total_count), ...}
    average_rating_per_user = sc.broadcast(average_rating_per_user_rdd).value
    value_list = list(average_rating_per_user.values())
    USER_AVERAGE = sum([pair[0] for pair in value_list])/sum([pair[1] for pair in value_list])
    # print(BUSINESS_AVERAGE)
    '''
    
    # others are generated from other dataset. Can do different time count since this block is in modules.

    additional_data_class = additional_features(business=business_json, checkin=checkin_json, photo=photo_json, review=review_json, tip=tip_json, user=user_json)
    bus_columns = additional_data_class.process_business_star_review()
    user_columns = additional_data_class.process_user()
    # tip_columns = additional_data_class.process_tips()
    
    
    ## 2. Aggregate the data to generate train/test
    ## seems like xgboost only support numpy/pandas. so unfortunately I need to use it. 
    training_data_raw = finalRdd_train.map(lambda line: (line[0], line[1], float(line[2]))).collect() # in a format of user - bus - score
    testing_data_raw = finalRdd_val.map(lambda line: (line[0], line[1], float(line[2]))).collect() # in a format of user - bus - score

    
    xgboost_training_x = list()
    for pair in training_data_raw:
        tuple_train_line = get_user_bus_info(pair, star_review_info = bus_columns, user_profile = user_columns)
        xgboost_training_x.append(tuple_train_line)
    
    xgboost_testing_x = list()
    for pair in testing_data_raw:
        tuple_train_line = get_user_bus_info(pair, star_review_info = bus_columns, user_profile = user_columns)
        xgboost_testing_x.append(tuple_train_line)

    
    # print(len(xgboost_training_x))
    features_name = ["uid","bid", "star", "review_count", "longitude", "latitude", "user_avg_star", "useful", "funny", "cool", "fans", "score"]
    train_data_dataframe = pd.DataFrame(xgboost_training_x, columns=features_name)
    test_data_dataframe = pd.DataFrame(xgboost_testing_x, columns=features_name)
    
    ## 3. Model-based method
    train_X = train_data_dataframe.drop(columns = ['uid','bid','score'])
    train_y = train_data_dataframe['score']
    test_X = test_data_dataframe.drop(columns = ['uid','bid','score'])
    test_y = test_data_dataframe['score']
    
    scaler_X = StandardScaler().fit(train_X)
    train_X = scaler_X.transform(train_X)
    test_X = scaler_X.transform(test_X)
    
    
    model = xgb.XGBRegressor()

    model.fit(X=train_X, y=train_y)
    # pickle.dump(model, open('2.2_model_output', 'wb'))

    ## 4. Testing on validation set
    xgbprediction = model.predict(test_X)
    
    ## 5. calculate the rmse
    test_y = test_data_dataframe['score'].tolist()
    test_y = [float(i) for i in test_y]
    

    ## write the result. 
    with open(output_file, "w") as f:
        f.write("user_id, business_id, prediction\n")
        for i in range(0, len(xgbprediction)):
            original = testing_data_raw[i]
            prediction = xgbprediction[i]
            line = str(original[0]) + ',' +  str(original[1]) + ',' + str(prediction) + '\n'
            f.write(line)
   
    # do some calculation
    
    predictions_list = xgbprediction.tolist()
    rmse = calculate_rmse(test_y, predictions_list)
    abs_diff_list = calculate_absolute_differences(test_y, predictions_list)
    print("RMSE:", rmse)
    print(str(abs_diff_list))

    sc.stop()
    

    print("Duration: ", time.time()-start)
