from pyspark import SparkContext, StorageLevel, SparkConf

import itertools
import sys
import json
import csv
import time
import math
import random



def process(line):
    splited_line= line[0].split(',')
    if len(line)<3:
        splited_line.append(0)
    return (splited_line[0], splited_line[1], splited_line[2])

def convertValuesToTuple(lineSet):
    newlineSet = []
    for line in lineSet:
        newlineSet += [(line, 1)]
    return newlineSet

def normalize_rating(rating, min_rating, max_min_diff, average):
    if rating == 0.0:
        return ((average - min_rating) / max_min_diff) * 4 + 1
    else:
        return((rating - min_rating) / max_min_diff) * 4 + 1

def calculate_rmse(mixed_data):
    rmse = math.sqrt(mixed_data.map(lambda line: (line[1][1] - line[1][0]) ** 2).mean())
    return rmse

def calculate_absolute_differences(mixed_data):
    abs_diff = {'0-1': 0, '1-2':0, '2-3':0, '3-4':0, '>4':0}
    for line in mixed_data.collect():
        if abs(line[1][1] - line[1][0]) <= 1:
            abs_diff['0-1'] += 1
        elif abs(line[1][1] - line[1][0]) <= 2:
            abs_diff['1-2'] += 1
        elif abs(line[1][1] - line[1][0]) <= 3:
            abs_diff['2-3'] += 1
        elif abs(line[1][1] - line[1][0]) <= 4:
            abs_diff['3-4'] += 1
        else:
            abs_diff['>4'] += 1
        
    return abs_diff

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
            return 0
        # correlation between two business
        else:
            return num/den
    else:
        return 0
    

def make_prediction_item_based(uid, bid, num_of_neighbors = 50):
    # use pearson correlation
    # for each uid-bid pair, calculate other bid co-rated by the uid. 
    # if such uid does not exist, use the average rating. 
    
    ## 1. find the bids also rated by the user, return an empty list if false.
    rated_bids = users_business_train_map.get(uid, [])
    
    ## 2. for those bids, calculate the similarities between them and the selected bid. 
    similarity_list = []
    for candidate_bid in rated_bids:
        similarity = calculate_pearson_similarity(bid, candidate_bid)
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
    
    return ((uid, bid), score)
    


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
    with open(output_file, "w+", encoding="utf-8") as f:
        f.write("user_id, business_id, prediction\n")
        f.write('\n'.join('{},{},{}'.format(x[0], x[1], x[3]) for x in output_predictions))

    print("RMSE: ", rmse)
    print("ABS Diff: ", str(abs_diff))


def hash_gen(users, hash_num, user_num):
    user_list = list(users)
    system_max_value = sys.maxsize
    hashed_users = [system_max_value for i in range(0, hash_num)]
    
    for user in user_list:
        for i in range(1, hash_num+1):
            # a custom hash implementation
            hash_code = (i*(user+73)) % num_users
            
            if hash_code < hashed_users[i-1]:
                hashed_users[i-1] = hash_code
    return hashed_users



if __name__ == '__main__':
    
    ## settings
    
    #input_file_path_train = sys.argv[1]
    #input_file_path_validation = sys.argv[2]

    #output_file_path = sys.argv[4]
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    

    conf = SparkConf().setMaster("local") \
            .setAppName("task1") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)
    

    # input_file_path_validation = "./yelp_val.csv"
    # input_file_path_train = "./yelp_train.csv"
    # output_file_path = "./task2__result_4.csv"
    # case = 4
    
    
    start = time.time()

    # load training data:
    uid_bid_rdd = sc.textFile(train_file).map(lambda line: line.split('\n')).map(lambda line: process(line))
    headers = uid_bid_rdd.first()
    finalRdd_train = uid_bid_rdd.filter(lambda line: line != headers)
    
    # load testing data:
    uid_bid_rdd_val = sc.textFile(test_file).map(lambda line: line.split('\n')).map(lambda line: process(line))
    headers = uid_bid_rdd_val.first()
    finalRdd_val = uid_bid_rdd_val.filter(lambda line: line != headers)
    
    all_info_train_map = sc.broadcast(finalRdd_train.map(lambda line: ((line[0], line[1]), float(line[2]))).collectAsMap()).value

    
    users_business_train_map_rdd = finalRdd_train.map(lambda line: (line[0], line[1])).groupByKey().mapValues(lambda line: set(line)) \
        .sortByKey().collectAsMap()
    users_business_train_map = sc.broadcast(users_business_train_map_rdd).value

    business_user_train_map_rdd = finalRdd_train.map(lambda line: (line[1], line[0])).groupByKey().mapValues(lambda line: set(line)) \
        .sortByKey().collectAsMap()

    business_user_train_map = sc.broadcast(business_user_train_map_rdd).value

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

    
    item_based_CF(finalRdd_train, finalRdd_val)

    
    # als_model.save(sc, "model.txt")
    print("Duration: ", time.time()-start)
    
    # sc.stop()
