from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from graphframes import GraphFrame

import itertools
import os
import sys
import time

# execution command: spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py 7 ub_sample_data.csv output.txt

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")




def save_to_txt(result, path):
    with open(path, 'w+') as f:
        for id in result:
            f.writelines(str(id)[1:-1] + "\n")


if __name__ == '__main__':
    start = time.time()
    # define input variables
    # filter_threshold = "7"
    # input_path = "ub_sample_data.csv"
    # output_path = "task1.txt"

    filter_threshold = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    conf = SparkConf().setMaster("local") \
        .setAppName("task1") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
     
    sc = SparkContext(conf=conf)
    sparkSession = SparkSession(sc)
    sc.setLogLevel("WARN")

    # 1. read data
    raw_rdd = sc.textFile(input_path)
    header = raw_rdd.first()
    uid_bid = raw_rdd.filter(lambda line: line != header).map(lambda line: (line.split(',')[0], line.split(',')[1]))
    uid_bid_dict = uid_bid.groupByKey().mapValues(lambda item: sorted(list(item))).collectAsMap()
    
    ## create uid pairs
    uid_pairs = list(itertools.combinations(list(uid_bid_dict.keys()), 2))

    # 2. create edge list
    edge_list = list()
    storage_set = set()
    for item in uid_pairs:
        intersect = set(uid_bid_dict[item[0]]).intersection(set(uid_bid_dict[item[1]]))
        if len(intersect) >= int(filter_threshold):
            edge_list.append(tuple(item))
            edge_list.append(tuple((item[1], item[0])))
            storage_set.add(item[0])
            storage_set.add(item[1])

    graph_df = sc.parallelize(list(storage_set)).map(lambda uid: (uid,)).toDF(['id'])
    edge_df = sc.parallelize(edge_list).toDF(["src", "dst"])

    graph_frame = GraphFrame(graph_df, edge_df)

    community = graph_frame.labelPropagation(maxIter=5)

    com_rdd = community.rdd.coalesce(1).map(lambda line: (line[1], line[0]))
    com_rdd = com_rdd.groupByKey().map(lambda label: sorted(list(label[1]))).sortBy(lambda line: (len(line), line))

    # 3. save to target file
    save_to_txt(com_rdd.collect(), output_path)

    ## get execution time
    print("Duration: ", str(time.time() - start))
