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


def getFPR(predictions, user_set):
    pass

def myhashs(input_string):
    '''
    input_string is a string of the given user id
    '''
    # Some custom hash function settings
    
    num_of_hash_functions = 5
    filter_length = 69997
    
    
    function_list = []
    result = []
    
    def customHashFunction(a, b, m):
        def hash_function(x):
            return ((a * x + b) % 24593) % m # some random prime
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





if __name__ == '__main__':
    start = time.time()
    
    #input_file_path = 'users.txt'
    #stream_size = 100
    #num_of_tasks = 30
    #output_file_path = 'task1_output.txt'
    
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_tasks = int(sys.argv[3])
    output_file_path = sys.argv[4]
    
    
    stream_hash_list = [] 
    # also record groud truth
    uid_list = []
    
    

    # spark settings
    conf = SparkConf().setMaster("local[*]").setAppName("task1").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    

    bx = BlackBox()
    
    # store the output
    fp_count_list = []
    output = [] 
    
    ## Start of the main loop.
    for _ in range(0, num_of_tasks):
        stream_users = bx.ask(input_file_path, stream_size)
        fp_count = 0.0
        # append hash to hash_list
        for uid in stream_users:
            hash_result = myhashs(uid)
            if hash_result in stream_hash_list:
                # print('yes')
                if uid not in uid_list:
                    fp_count += 1
            
            uid_list.append(uid)
            stream_hash_list.append(myhashs(uid))
            
        print(_, fp_count)
        output.append(str(_)+','+str(float(fp_count/stream_size))+'\n')
        
    sc.stop()
    
    with open(output_file_path, 'w') as f:
        f.write('Time,FPR\n')
        f.writelines(output)
    
    print("Duration: ", time.time() - start)
        
