

import sys
import json

from pyspark import SparkContext
from datetime import datetime
from operator import add
sc = SparkContext.getOrCreate()

if __name__ == '__main__':

	# input_json_path = "./resource/asnlib/publicdata/review_sample.json"
	# output_file_path = "output1.json"
	# year = '2018'


	input_json_path = sys.argv[1]
	output_file_path = sys.argv[2]
		

	result = dict()
	test_review_rdd = sc.textFile(input_json_path).map(lambda r: json.loads(r))

	## A. The total number of reviews (0.5 point)
	result['n_review']= test_review_rdd.map(lambda l: ("count",1)).reduceByKey(lambda a,b: a + b).collect()[0][1]
	
	## B. The number of reviews in 2018 (0.5 point)
	date_rdd = test_review_rdd.map(lambda df: (df['review_id'], df['date']))
	review_in_given_year = date_rdd.filter(lambda df: datetime.strptime(
	df[1], '%Y-%m-%d %H:%M:%S').year == 2018).map(lambda df: ('Year',1)).persist()
	result['n_review_2018'] = review_in_given_year.reduceByKey(lambda a,b: a + b).collect()[0][1]

	## C. The number of distinct users who wrote reviews (0.5 point)
	user_rdd = test_review_rdd.map(lambda df: (df['user_id']))
	users = user_rdd.distinct().map(lambda df: ('User',1)).persist()
	result['n_user']= users.reduceByKey(add).collect()[0][1]
 	
 	## D.The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote (0.5 point)
	user_review_rdd = test_review_rdd.map(lambda df: (df['user_id'],df['review_id']))
	result['top10_user'] = user_review_rdd.distinct().map(lambda df: (df[0], 1)).reduceByKey(add).takeOrdered(10, key=lambda df: (-df[1], df[0]))
	
	## E. The number of distinct businesses that have been reviewed (0.5 point)
	business_rdd = test_review_rdd.map(lambda df: (df['business_id']))
	business = business_rdd.distinct().map(lambda df: ('Business',1)).persist()
	result['n_business'] = business.reduceByKey(add).collect()[0][1]

    ## F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had (0.5 point)
	business_review_rdd = test_review_rdd.map(lambda df: (df['business_id'],df['review_id']))
	business_review_rdd.persist()
	result['top10_business'] = business_review_rdd.distinct().map(lambda df: (df[0],1)).reduceByKey(add).takeOrdered(10, key=lambda df: (-df[1], df[0]))

	print(result)

	with open(output_file_path, 'w+') as output_file:
	        json.dump(result, output_file)
	output_file.close()


