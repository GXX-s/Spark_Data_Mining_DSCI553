
import sys
import json

from pyspark import SparkContext
from operator import add

from datetime import datetime
import time

if __name__ == '__main__':
	# review_json_path = "./resource/asnlib/publicdata/review.json"
    # output_file_path = "output_task3.json"
    # n_partition = '21'

	input_file_1 = sys.argv[1]
	output_file_path = sys.argv[2]
	n_partition = int(sys.argv[3])

	sc = SparkContext.getOrCreate()

	result_dict = dict()
	result_default = dict()
	result_customized = dict()

	#test_review = sc.textFile(input_file_1).map(lambda r: json.loads(r))

	## default: get the number of partitions
	time_start=time.time()
	test_review_rdd = sc.textFile(input_file_1).map(lambda r: json.loads(r)).persist()

	business_review_rdd = test_review_rdd.map(lambda df: (df['business_id'],df['review_id'])).persist()
    
	resultF_default = business_review_rdd.distinct().map(lambda df: (df[0],1)).reduceByKey(add).takeOrdered(10, key=lambda df: (-df[1], df[0]))
	time_end=time.time()

	result_default['n_partition'] = business_review_rdd.getNumPartitions()
	result_default['n_items'] = business_review_rdd.glom().map(len).collect()
	result_default['exe_time'] = time_end-time_start
	

	# customized: get the number of partitions
	time_start=time.time()
	test_review_rdd = sc.textFile(input_file_1).map(lambda r: json.loads(r)).persist()
	### find a subset first, map first and then partitionby
	business_review_rdd = test_review_rdd.map(lambda df: (df['business_id'],df['review_id'])).partitionBy(n_partition,lambda key:ord(key[:1])).persist()

	result_F = business_review_rdd.distinct().map(lambda df: (df[0],1)).reduceByKey(add).takeOrdered(10, key=lambda df: (-df[1], df[0]))
	time_end=time.time()
	result_customized['n_partition'] = business_review_rdd.getNumPartitions()
	result_customized['n_items']  = business_review_rdd.glom().map(len).collect()
	#mapPartitions(): Return a new RDD by applying a function to each partition of this RDD.

	#result['n_items'] =  business_id.mapPartitions(lambda a: len(list(a.values))).collect()
	
	result_customized['exe_time'] = time_end-time_start

	result_dict = {'default':result_default,'customized':result_customized}

	with open(output_file_path,'w+') as output_file:
		json.dump(result_dict,output_file)
	output_file.close()


