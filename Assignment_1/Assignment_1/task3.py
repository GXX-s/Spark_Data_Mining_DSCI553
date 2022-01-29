

import sys
import json
from pyspark import SparkContext

from operator import add
from datetime import datetime
import time



def sort_tuple(tup):
    num = len(tup)
    for i in range(0, num):
        for j in  range(0, num-i-1):
            if (tup[j][1] < tup[j+1][1]):
                temp = tup[j]
                tup[j] = tup[j+1]
                tup[j+1] = temp
    return tup
def sort_tuple_2(tup):
    num = len(tup)
    for i in range(0, num):
        for j in  range(0, num-i-1):
            if (tup[j][0] >tup[j+1][0]):
                temp = tup[j]
                tup[j] = tup[j+1]
                tup[j+1] = temp
    return tup




if __name__ == '__main__':


    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]
    
    sc = SparkContext.getOrCreate()
    
    test_review = sc.textFile(input_file_1).map(lambda r: json.loads(r))
    business = sc.textFile(input_file_2).map(lambda r: json.loads(r))
    
    result_dict = dict()
    
    ## Average stars for each city
    
    review_star = test_review.map(lambda df: (df['business_id'], df['stars'])).persist()
    business_city = business.map(lambda df: (df['business_id'], df['city'])).persist()
    
    score_sum_len = review_star.groupByKey()\
    .mapValues(lambda values:[float(value) for value in values ])\
    .map(lambda df: (df[0],(sum(df[1]),len(df[1]))))
    
    merge_rdd = business_city.leftOuterJoin(score_sum_len).persist()

    sorted_rdd = merge_rdd.map(lambda kvv: kvv[1]).filter(lambda kv: kv[1] is not None)\
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
    .mapValues(lambda value: float(value[0] / value[1])) \
    .sortBy(lambda df: (-df[1], df[0])).collect()
    
    
    lines =sorted_rdd
    final_lines = ['city,', 'stars\n']
    for item in lines:
        final_lines.append(str(item[0])+ ',' + str(item[1]) + '\n')
    
    
    with open(output_filepath_question_a,'w') as output_file_1:
        output_file_1.writelines(final_lines)
    output_file_1.close()
    
    
    
    # Compare the execution time of using two methods to print top 10 cities with highest stars.
    ## Method1: Collect all the data, sort in python, and then print the first 10 cities
    
    score_sum_len = review_star.groupByKey()\
    .mapValues(lambda values:[float(value) for value in values ])\
    .map(lambda df: (df[0],(sum(df[1]),len(df[1]))))
    merge_rdd = business_city.leftOuterJoin(score_sum_len).persist()


    sorted_rdd = merge_rdd.map(lambda kvv: kvv[1]).filter(lambda kv: kv[1] is not None)\
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
    .mapValues(lambda value: float(value[0] / value[1])).collect()
    print(sorted_rdd)


    time_start = time.time()
    new = sort_tuple(sorted_rdd)
    final = sort_tuple_2(new)
    print(final[:10])
    time_end = time.time()
    exe_time_m = time_end - time_start
    
    
    ## Method2: Sort in Spark, take the first 10 cities, and then print these 10 cities

    score_sum_len = review_star.groupByKey()\
    .mapValues(lambda values:[float(value) for value in values ])\
    .map(lambda df: (df[0],(sum(df[1]),len(df[1]))))
    merge_rdd = business_city.leftOuterJoin(score_sum_len).persist()


    reduced_rdd = merge_rdd.map(lambda kvv: kvv[1]).filter(lambda kv: kv[1] is not None)\
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
    .mapValues(lambda value: float(value[0] / value[1])).persist()
    time_start = time.time()
    sorted_rdd = reduced_rdd.sortBy(lambda kv: (-kv[1], kv[0])).take(10)
    print(sorted_rdd)
    time_end = time.time()
    exe_time_n = time_end - time_start
    
    
    
    ## Output
    result_dict_b = {}
    result_dict_b['m1'] = exe_time_m
    result_dict_b['m2'] = exe_time_n
    result_dict_b['reason'] = 'Python does the work faster because: 1. It can load all the data at one time, while Spark has to shuffle and aggregate the data from all the distributed chunk (the map tasks) to generate the final sorted list. 2. Also because transformation lazinesss, the previous transformations are evaluated only after calling on an action, so it takes extra time. Whereas, sorting via Python is called on the data collected from RDD (action finished), so it does not spend time on evaluating previous transformation tasks.'


    with open(output_filepath_question_b,'w+') as output_file_2:
        json.dump(result_dict_b,output_file_2)
    output_file_2.close()
