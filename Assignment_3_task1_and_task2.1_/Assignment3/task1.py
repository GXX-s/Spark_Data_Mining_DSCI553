from pyspark import SparkContext, SparkConf
import sys
import itertools
import time 



def customLSH(bus_id, signatures, n_rows, n_bands):
    signature_pairs = []
    for band in range(0,n_bands):
        sig = signatures[band*n_rows:(band*n_rows)+n_rows]
        sig.insert(0, band)
        signature_pair = (tuple(sig), bus_id)
        signature_pairs.append(signature_pair)

    return signature_pairs



def get_similar_bus(bus):

    similar_bus = list(itertools.combinations(sorted(bus),2))
    
    return sorted(similar_bus)


def jaccard_similarity(pairs, bus_matrix):

    p1 = set(bus_matrix.get(pairs[0]))
    p2 = set(bus_matrix.get(pairs[1]))

    intersection = len(p1.intersection(p2))
    union = len(p1.union(p2))

    similarity = float(intersection)/float(union)

    return (pairs, similarity)


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


def process(line):
    splited_line= line[0].split(',')
    return (splited_line[0], splited_line[1], splited_line[2])


if __name__ == "__main__":
    output = []
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]


    conf = SparkConf().setMaster("local") \
            .setAppName("task1") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)

    
    jaccard_support = 0.5
    hash_num = 100
    n_bands = 40
    n_rows = 2

    start = time.time()
    
    uid_bid_rdd = sc.textFile(input_file_path).map(lambda line: line.split('\n')).map(lambda line: process(line))
    headers = uid_bid_rdd.first()
    final_rdd = uid_bid_rdd.filter(lambda line: line != headers)
    

    users = final_rdd.map(lambda line: line[0]).distinct()
    businesses = final_rdd.map(lambda line: line[1]).distinct()
    num_users = users.count()
    num_businesses = businesses.count()
    
    # print('Num of users/businesses:', num_users, num_businesses)
    
    

    user_index_dict = final_rdd.map(lambda line: line[0]).zipWithIndex().collectAsMap()

    # tokenize and generate the business-user map 
    bus_user_dict = final_rdd.map(lambda line: (line[1], user_index_dict.get(line[0])))
    bus_user_dict = bus_user_dict.groupByKey().sortByKey().mapValues(lambda line: set(line)).persist()
    bus_user_dict_collected = bus_user_dict.collect()
    bus_matrix = {}

    for bus in bus_user_dict_collected:
        bus_matrix.update({bus[0]: bus[1]})

    # do hash
    pairs = bus_user_dict.mapValues(lambda users: hash_gen(users, hash_num, num_users))
    pairs = pairs.flatMap(lambda line: customLSH(line[0], list(line[1]), n_rows, n_bands))
    pairs = pairs.groupByKey().filter(lambda line: len(list(line[1])) > 1)
    pairs = pairs.flatMap(lambda line: get_similar_bus(sorted(list(line[1])))).distinct()

    results = pairs.map(lambda cd: jaccard_similarity(cd, bus_matrix)).filter(lambda cd: cd[1] >= jaccard_support).sortByKey()

    output = results.collect()


    with open(output_file_path, "w") as f:
        f.write("business_id_1, business_id_2, similarity\n")
        f.write('\n'.join('{},{},{}'.format(item[0][0], item[0][1], item[1]) for item in output))
    
    
    # for debug
    print("output count: ", len(output))
    print("total time: " + str(time.time()-start))
