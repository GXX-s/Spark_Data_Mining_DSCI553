import json
import csv
import binascii
import random
import time
import sys

from pyspark import SparkContext, SparkConf


class BlackBox:

    def ask(self, file, num):
        lines = open(file,'r').readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users
    
def myhashs(input_string):
    '''
    input_string is a string of the given user id
    '''
    # Some custom hash function settings
    
    num_of_hash_functions = 10
    filter_length = 69997
    
    
    function_list = []
    result = []
    
    def customHashFunction(a, b, m):
        def hash_function(x):
            return (a * x + b) % 12289 % 769 # some random prime
        return hash_function
    
    a_list = random.sample(range(2, sys.maxsize - 1), num_of_hash_functions)
    b_list = random.sample(range(2, sys.maxsize - 1), num_of_hash_functions)
    
    for a, b in zip(a_list, b_list):
        function_list.append(customHashFunction(a, b, filter_length))
        
    input_string_int = stringConvert(input_string)
    
    for func in function_list:
        result.append(func(input_string_int))
        
    return result
    
        

def stringConvert(input_string):
    return int(binascii.hexlify(input_string.encode('utf8')),16)


def getLongestZero(hash_list):
    result = 0 # initialize
    for hash_int in hash_list:
        binary_string = bin(hash_int)[2:]
        count = len(binary_string) - len(binary_string.rstrip("0"))
        if count > result:
            result = count
    return result
    


if __name__ == '__main__':
    start = time.time()
    
    # input_file_path = 'users.txt'
    # stream_size = 300
    # num_of_tasks = 30
    # output_file_path = 'task2_output.txt'
    
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_tasks = int(sys.argv[3])
    output_file_path = sys.argv[4]


    # spark settings
    conf = SparkConf().setMaster("local[*]").setAppName("task1").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    

    bx = BlackBox()
    
    # store the output
    output = [] 
    final_list = [0.0, 0.0]
    
    ## Start of the main loop.
    for _ in range(0, num_of_tasks):
        stream_users = bx.ask(input_file_path, stream_size)
        # store ground truth
        uid_list = []
        stream_hash_list = []
        estimation = 0.0
        # append hash to hash_list
        for uid in stream_users:
            if uid not in uid_list:
                uid_list.append(uid)
            stream_hash_list.append(myhashs(uid))
            
        
        # then do the estimation
        estimation = 0.0
        for i in range(0, len(stream_hash_list[0])):  # num of hash functions
            hash_result = [item[i] for item in stream_hash_list]
            max_num_of_zeros = getLongestZero(hash_result)
            estimation += 2**max_num_of_zeros
        final_estimation = int(estimation / len(stream_hash_list[0]))
        print("ground", len(uid_list), "estimation", final_estimation)
        final_list[0] += len(uid_list)
        final_list[1] += final_estimation
        output.append(str(_)+','+str(len(uid_list))+','+str(final_estimation)+'\n')
        
    sc.stop()
    print(final_list[1]/final_list[0])
    with open(output_file_path, 'w') as f:
        f.write('Time,Ground Truth,Estimation\n')
        f.writelines(output)
    
    print("Duration: ", time.time() - start)
        
        
        
        
