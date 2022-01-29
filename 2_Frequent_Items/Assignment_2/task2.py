from operator import add
from functools import reduce
import math
import collections
import copy
from itertools import combinations

from pyspark import SparkContext, SparkConf, StorageLevel
import sys
import time
import csv


random_bucket_number = 1000


    

def candidate_list_to_dic(candidate_list):
    return [tuple(candidate.split(",")) for candidate in candidate_list]

def reduce_basket_size(basket, candidate_list):
    # return the intersection of two list for frequent items
    intersection_part = list(set(basket).intersection(set(candidate_list)))
    return sorted(intersection_part)

def gen_permutation(comb_list):
    if len(comb_list) > 0:
        size = len(comb_list[0])
        perm_list = list()
        for i, pairs_f in enumerate(comb_list[:-1]):
            for pairs_l in comb_list[i + 1:]:
                if pairs_f[:-1] == pairs_l[:-1]:
                    union_result = set(pairs_f).union(set(pairs_l))
                    comb = tuple(sorted(list(union_result)))
                    # create a list to store temp pairs
                    temp_pair = []
                    for pair in combinations(comb, size):
                        temp_pair.append(pair)
                    if set(temp_pair).issubset(set(comb_list)): # cannot use in here
                        perm_list.append(comb)
                else:
                    break

        return perm_list
    

def hash_function(combination):
    # do a customized hash function
    combination_list = list(combination)
    result = 0
    for word in combination_list:
        result += sum(ord(c) for c in str(word))
    return result % random_bucket_number

def generate_candidate_list(baskets, support):
    # generate the candidate list from the partitions
    bitmap = [0 for _ in range(random_bucket_number)]
    bitmap = list(map(lambda val: True if val >= support else False, bitmap))
    
    
    temp_storage = collections.defaultdict(list)
    for basket in baskets:
        for item in basket:
            temp_storage[item].append(1)
        comb_baskets = combinations(basket, 2)
        for combs in comb_baskets:
            hash_result = hash_function(combs)
            bitmap[hash_result] = (bitmap[hash_result] + 1)
    # then do a filtering based on support
    filtered_result = dict(filter(lambda item: len(item[1]) >= support, temp_storage.items()))
    candidates = list(filtered_result.keys())
    candidate_sorted = sorted(candidates)

    return candidate_sorted, bitmap

    
def find_candidate_itemset(data_baskets, original_support, total_size):
    list_baskets = list(data_baskets)
    data_baskets = copy.deepcopy(list_baskets)
    support = math.ceil(original_support * len(list_baskets) / total_size)

    # print("Baskets in list: ", list_baskets)
    all_candidate_dict = collections.defaultdict(list) # use this instead of {} to avoid key error
    
    # Try to generate a candidate list
    candidate_list_original, candidate_bitmap = generate_candidate_list(list_baskets, support)
    candidate_list = candidate_list_original
    
    index = 1
    all_candidate_dict[str(index)] = candidate_list_to_dic(candidate_list_original)
    # Repeat to remove candidates until list empty
    while None is not candidate_list and len(candidate_list) > 0:
        index += 1
        check = collections.defaultdict(list)
        for basket in list_baskets:
            basket = reduce_basket_size(basket, candidate_list_original)
            # print("check the shrinked basket result here: " basket)
            if len(basket) >= index:
                if index == 2:
                    # do pairs case, and append to current check result
                    for pair in combinations(basket, index):
                        if candidate_bitmap[hash_function(pair)]:
                            check[pair].append(1)

                if index >= 3:
                    # do the rest of the cases
                    for candidate_item in candidate_list:
                        if set(candidate_item).issubset(set(basket)):
                            check[candidate_item].append(1)


        # filter the check result based on support, and generate new candidates accordingly.
        filtered_dict = dict(filter(lambda item: len(item[1]) >= support, check.items()))
        candidate_list = gen_permutation(sorted(list(filtered_dict.keys())))
        if len(filtered_dict) == 0:
            break
        all_candidate_dict[str(index)] = list(filtered_dict.keys())

    # combine pair items to 1d.
    yield reduce(lambda a1, a2: a1 + a2, all_candidate_dict.values())




def count_frequent(data_baskets, candidate_pairs):
    temp_counter = collections.defaultdict(list)
    for pairs in candidate_pairs:
        if set(pairs).issubset(set(data_baskets)):
            temp_counter[pairs].append(1)

    yield [tuple((key, sum(value))) for key, value in temp_counter.items()]



def reformat(itemset_data):
    # do some reformat to fit in the homework requirements.
    count_len = 1
    output = ""
    for pair in itemset_data:
        if len(pair) == 1:
            output += str("(" + str(pair)[1:-2] + "),")

        elif len(pair) != count_len:
            output = output[:-1] + "\n\n"
            count_len = len(pair)
            output += (str(pair) + ",")
        else:
            output += (str(pair) + ",")

    return output[:-1]


def write_output(candidate, frequent, path):
    with open(path, 'w+') as f:
        str_result = 'Candidates:\n' + reformat(candidate) + '\n\n' + 'Frequent Itemsets:\n' + reformat(frequent)
        f.write(str_result)
        
def pre_processing(csv_path, output_path):
    output_list = [['DATE-CUSTOMER_ID', 'PRODUCT_ID']]
    with open(csv_path, "r") as f:
        csvreader = csv.reader(f)
        header = []
        header = next(csvreader)
        
        rows = []
        for row in csvreader:
            output_list.append([str(row[0]) + str(row[1]), int(str(row[5]))])
    
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(output_list)
        


if __name__ == '__main__':


#    filter_threshold = "20"
#    support = "50"
#    input_csv_path = "../Assignment2/resource/asnlib/publicdata/ta_feng_all_months_merged.csv"
#    output_file_path = "../Assignment2/test_out/output_task_2_test.txt"


    # Use the following arguments in the final submission
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_csv_path = sys.argv[3]
    output_file_path = sys.argv[4]
    
    
    # do the proper configurations first before start counting the time; this section should be commented out in the final code.
#    configuration = SparkConf()
#    configuration.set("spark.driver.memory", "4g")
#    configuration.set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate()
    
    # the main process.
    start = time.time()
    
    # define a partition number, use 2 here
    partition_number = 2
    
    # replace the path below to "./ta_feng_processed.csv" or anything required by the submission, and see if it works.
    # do note that do not use the relative directory for testing environment.
    proc_csv_path = "./ta_feng_processed.csv"
    
    


    # pre-process
    pre_processing(input_csv_path,proc_csv_path)
    
    
    filter_threshold = int(filter_threshold)
    support = int(support)
    
    rdd_initial = sc.textFile(proc_csv_path, partition_number)
    header = rdd_initial.first()
    data_rdd = rdd_initial.filter(lambda line: line != header)

    
    # case 1
    rdd = data_rdd.map(lambda line: (line.split(',')[0], line.split(',')[1]))
    rdd = rdd.groupByKey().map(lambda uid_bids: (uid_bids[0], tuple(set(uid_bids[1]))))
    # add filter
    rdd = rdd.filter(lambda uid_bids: len(uid_bids[1]) > filter_threshold).map(lambda uid_bids: uid_bids[1])

    # disk only
    rdd.persist(StorageLevel.DISK_ONLY)
    
    
    # SON algorithm
    total_size = rdd.count()
    
    # print("The total data size is: ", total_size)
    candidate_itemset = rdd.mapPartitions(lambda partition: find_candidate_itemset(data_baskets=partition, original_support=support, total_size=total_size))
    # get distinct and then sort
    candidate_itemset = candidate_itemset.flatMap(lambda pairs: pairs).distinct().sortBy(lambda pairs: (len(pairs), pairs)).collect()
    # print("candidate_itemset", candidate_itemset)

    frequent_itemset = rdd.flatMap(lambda basket: count_frequent(data_baskets=basket,candidate_pairs=candidate_itemset)) .flatMap(lambda pairs: pairs).reduceByKey(add) \
    # filter through support
    frequent_itemset = frequent_itemset.filter(lambda count: count[1] >= int(support))
    # do sort
    frequent_itemset = frequent_itemset.map(lambda count: count[0]).sortBy(lambda pairs: (len(pairs), pairs))
    frequent_itemset = frequent_itemset.collect()
    
    
    # finally need to export the result to the file.
    write_output(candidate=candidate_itemset,
                  frequent=frequent_itemset,
                  path=output_file_path)

    print("Duration: %d s." % (time.time() - start))
