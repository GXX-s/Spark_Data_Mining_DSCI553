## Task 2.3 based on 2.2 and 2.1

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

import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



def calculate_rmse(mixed_data):
    rmse = math.sqrt(mixed_data.map(lambda line: (line[1][1] - line[1][0]) ** 2).mean())
    return rmse

def process_prediction(line):
    splited_line= line[0].split(',')
    return ((splited_line[0], splited_line[1]), (float(splited_line[2]), float(splited_line[3])))


def calculate_pearson_similarity(bid1, bid2):
    # get the commonly rated users for these two businesses. 
    bid1_users = business_user_train_map.get(bid1, [])
    bid2_users = business_user_train_map.get(bid2, [])
    intersections = list(set(bid1_users) & set(bid2_users))
    if len(intersections)>0:
        # for bid1 and bid2, calculate average co-rated users' score
        bid1_scores = []
        bid2_scores = []
        for co_rated_user in intersections:
            bid1_scores.append(float(all_info_train_map.get((co_rated_user, bid1))))
            bid2_scores.append(float(all_info_train_map.get((co_rated_user, bid2))))
        bid1_avg = sum(bid1_scores)/len(bid1_scores)
        bid2_avg = sum(bid2_scores)/len(bid2_scores)
        # numerator
        num = 0
        for i in range(0, len(bid1_scores)):
            num += (bid1_scores[i]-bid1_avg)*(bid2_scores[i]-bid2_avg)

        # denominator
        den_1 = 0
        den_2 = 0
        for i in range(0, len(bid1_scores)):
            den_1 += (bid1_scores[i]-bid1_avg)**2
            den_2 += (bid2_scores[i]-bid2_avg)**2

        den = math.sqrt(den_1)*math.sqrt(den_2)
        if den == 0:
            return 0, len(intersections)
        # correlation between two business
        else:
            return num/den, len(intersections)
    else:
        return 0, 0
    

def make_prediction_item_based(uid, bid, num_of_neighbors = 20):
    # use pearson correlation
    # for each uid-bid pair, calculate other bid co-rated by the uid. 
    # if such uid does not exist, use the average rating. 
    
    ## 1. find the bids also rated by the user, return an empty list if false.
    rated_bids = users_business_train_map.get(uid, [])
    
    ## 2. for those bids, calculate the similarities between them and the selected bid. 
    similarity_list = []
    for candidate_bid in rated_bids:
        similarity, len_intersections = calculate_pearson_similarity(bid, candidate_bid)
        similarity_list.append((candidate_bid, similarity))
    
    ## 3. sort and find the tops (num of neightbors):
    similarity_list.sort(key=lambda x:x[1])
    max_n = min(num_of_neighbors, len(similarity_list))
    top_n_neighbours = similarity_list[-1*max_n:]
    
    ## 4. get the final score and return
    final_score = 0
    
    for pair in top_n_neighbours:
        final_score+= float(all_info_train_map.get((uid, pair[0])))
    if len(top_n_neighbours) == 0:
        avg_rating = average_rating_per_business.get(bid)[0]/average_rating_per_business.get(bid)[1]
        score = avg_rating
    else:
        score = final_score/len(top_n_neighbours)
    
    return ((uid, bid), score, len_intersections)
    


def item_based_CF(finalRdd_train, finalRdd_val):
    # prepare the rdd in a format of ((uid, bid), score)
    val_with_ratings = finalRdd_val.map(lambda line: ((line[0], line[1]), float(line[2])))

    users_business_val_set = finalRdd_val.map(lambda line: (line[0], line[1])).sortByKey()

    predicted_ratings = users_business_val_set.map(lambda line: make_prediction_item_based(line[0], line[1])).persist()

    max_rating = predicted_ratings.map(lambda line: line[1]).max()
    min_rating = predicted_ratings.map(lambda line: line[1]).min()
    max_min_diff = max_rating - min_rating


    normalized_predicted_ratings = predicted_ratings\
        .mapValues(lambda line: ((line - min_rating) / max_min_diff) * 4 + 1)

    original_and_predicted = val_with_ratings.join(normalized_predicted_ratings)
    rmse = calculate_rmse(original_and_predicted)
    abs_diff = calculate_absolute_differences(original_and_predicted)
    
    
    output_predictions = original_and_predicted \
        .map(lambda line: (line[0][0], line[0][1], line[1][0], line[1][1])) \
        .collect()

    # print("Completed Predicting")
    with open(output_file, "w+", encoding="utf-8") as fp:
        fp.write("user_id, business_id, original, prediction\n")
        fp.write('\n'.join('{},{},{},{}'.format(x[0], x[1], x[2], x[3]) for x in output_predictions))

    print("RMSE: ", rmse)
    print("ABS Diff: ", str(abs_diff))



def calculate_absolute_differences(mixed_data):
    abs_diff = {'0-1': 0, '1-2':0, '2-3':0, '3-4':0, '>4':0}
    for line in mixed_data.collect():
        abs_diff = abs(line[1][1] - line[1][0])
        if abs_diff <= 1:
            abs_diff['0-1'] += 1
        elif abs_diff <= 2:
            abs_diff['1-2'] += 1
        elif abs_diff <= 3:
            abs_diff['2-3'] += 1
        elif abs_diff <= 4:
            abs_diff['3-4'] += 1
        else:
            abs_diff['>4'] += 1
        
    return abs_diff

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
    #print(model.get_xgb_params())

    ## 4. Testing on validation set
    xgbprediction = model.predict(test_X)
    
    
    ## 5. Item-based method
    
    val_with_ratings = finalRdd_val.map(lambda line: ((line[0], line[1]), float(line[2])))
    users_business_val_set = finalRdd_val.map(lambda line: (line[0], line[1])).sortByKey()
    predicted_ratings = users_business_val_set.map(lambda line: make_prediction_item_based(line[0], line[1])).persist()
    
    # the key idea is to use the number of neighboring items as the weight of item-based method. more neighbors, higher weight.
    predicted_ratings_score_and_intersection_len = predicted_ratings.map(lambda line: (line[1], line[2])).collect()
    
    ## 6. Combination
    
    print(len(xgbprediction), len(predicted_ratings_score_and_intersection_len))
    final_score = []
    for i in range(0, len(xgbprediction)):
        xgb_result = xgbprediction[i]
        item_result = predicted_ratings_score_and_intersection_len[i]
        # weight for item-based
        alpha = min(item_result[1]/50.0/2.0, 0.5)
        final_score.append(item_result[0]*alpha + xgb_result*(1-alpha))
    
    
    ## write the result. 
    with open(output_file, "w") as f:
        #f.write("user_id,business_id,stars,prediction\n")
        f.write("user_id,business_id,prediction\n")
        for i in range(0, len(xgbprediction)):
            original = testing_data_raw[i]
            prediction = final_score[i]
            # line = str(original[0]) + ',' +  str(original[1]) + ',' + str(original[2]) + ',' + str(prediction) + '\n'
            line =  str(original[0]) + ',' +  str(original[1]) + ',' + str(prediction) + '\n'
            f.write(line)
   
    # do some calculation
    test_y = test_data_dataframe['score'].tolist()
    test_y = [float(i) for i in test_y]
    
    result = calculate_rmse(test_y , final_score)
    print("rmse:", result)

    sc.stop()

    print("Duration: ", time.time()-start)
