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
    # stream_size = 100
    # num_of_tasks = 30
    # output_file_path = 'task3_output.txt'
    
    input_file_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_tasks = int(sys.argv[3])
    output_file_path = sys.argv[4]


    # spark settings
    conf = SparkConf().setMaster("local[*]").setAppName("task1").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    random.seed(553)

    bx = BlackBox()
    
    # store the output
    user_list = [] 
    output_list = []
    total_user_count = 0
    output_list.append("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    print("seqnum,0_id,20_id,40_id,60_id,80_id")
    ## Start of the main loop.
    for _ in range(0, num_of_tasks):
        stream_users = bx.ask(input_file_path, stream_size)
        if total_user_count == 0:
            user_list = stream_users
            total_user_count += 100
        else:
            for i in range(0, len(stream_users)):
                total_user_count += 1
                float_prob = random.random()
                if float_prob < float(stream_size)/total_user_count:
                    # accept the sample
                    selected_index = random.randint(0,99)  # this will generate from 0 to 99, include 99
                    user_list[selected_index] = stream_users[i]
                
                
                

        # then do the estimation
        print(str(_),user_list[0],user_list[20],user_list[40],user_list[60],user_list[80])
        output_line = [str(_),user_list[0],user_list[20],user_list[40],user_list[60],user_list[80]]
        output_line_processed = ",".join(output_line) + '\n'
        output_list.append(output_line_processed)
        
    sc.stop()

    with open(output_file_path, 'w') as f:
        f.writelines(output_list)

    print("Duration: ", time.time() - start)
        
        
        
        
