import json
import csv
import binascii
import random
import time
import sys

from pyspark import SparkContext, SparkConf
from sklearn.cluster import KMeans
import numpy as np


def ClusterMerge(cluster_list):
    new_list = cluster_list.copy()
    for i in range(0, len(cluster_list)):
        for j in range(0, len(cluster_list)):
            if i < j:
                distance = cluster_list[i].MahalanobisDistanceCluster(cluster_list[j])
                if distance < (cluster_list[i].d**0.5)*2: # then need to merge the two CS.
                    # print("Merge two clusters.")
                    cluster_list[i].mergeCluster(cluster_list[j])
                    del new_list[j]
                    return True, new_list
    
    return False, new_list

def ClusterMergeCSDS(DS_list, CS_list):
    # only merge CS to DS. DS are not changed at all. 
    CS_list_copy = CS_list.copy()
    
    for cs_index in range(0, len(CS_list)):
        cs_cluster = CS_list[cs_index]
        distance_to_DS_list = []
        for i in range(0, len(DS_list)):
            ds_cluster = DS_list[i]
            distance = cs_cluster.MahalanobisDistanceCluster(ds_cluster)
            if distance < (DS_list[i].d**0.5)*2:
                distance_to_DS_list.append(distance)
        if len(distance_to_DS_list) != 0:
            min_distance_index = distance_to_DS_list.index(min(distance_to_DS_list))
            # then merge that CS to the corresponding DS.
            DS_list[min_distance_index].mergeCluster(cs_cluster)
            CS_list_copy.remove(cs_cluster)
    
    # return the remaining CS.
    return CS_list_copy
        
            
        



class Cluster:
    def __init__(self):
        self.NoP = 0
        self.SUM = None
        self.SUMSQ = None
        self.cluster_label = [] # a list to store all the index - cluster details.
        self.d = 0 # dimension
        
    def getSTD(self):
        '''
        Get the standard deviation of the cluster on each dimension. The output is a list. 
        '''
        std_list = []
        for i in range(0, self.d):
            std_square = (self.SUMSQ[i]/self.NoP - (self.SUM[i]/self.NoP)**2)
            std = std_square**0.5
            std_list.append(std)
            
        return std_list
        
        
    def MahalanobisDistance(self, datapoint):
        std_list = self.getSTD()
        yi_square_sum = 0.0
        for i in range(0, self.d):
            yi_square_sum += ((datapoint[1][i] - self.SUM[i]/self.NoP)/std_list[i])**2
        result = yi_square_sum**0.5
        return result
    
    def MahalanobisDistanceCluster(self, another_cluster):
        std_list_a = self.getSTD()
        std_list_b = another_cluster.getSTD()
        
        yi_square_sum = 0.0
        yi_square_sum_2 = 0.0
        for i in range(0, self.d):
            yi_square_sum += ((another_cluster.SUM[i]/another_cluster.NoP - self.SUM[i]/self.NoP)/(std_list_a[i]*std_list_b[i]))**2
            yi_square_sum_2 += ((another_cluster.SUM[i]/another_cluster.NoP - self.SUM[i]/self.NoP)/((std_list_b[i]+std_list_a[i])/2))**2

        result = yi_square_sum**0.5
        result_2 = yi_square_sum_2**0.5
        # print(result, result_2, self.d**0.5 * 2)
        return result
        
        
        
        
    def addPoints(self, data_list: list):
        '''
        The datalist contains many datapoints that are in the same format as sample data (original data from the rdd)
        Its format is: ((index, label), [value, value, ...])
        '''
        for datapoint in data_list:
            # first check if all items in datapoint are float
            for item in datapoint[1]:
                if type(item) != float:
                    print(datapoint)
            
            self.NoP += 1
            if self.SUM == None:
                self.SUM = datapoint[1]
            else:
                self.SUM = [sum(x) for x in zip(self.SUM, datapoint[1])]
                
            if self.SUMSQ == None:
                self.SUMSQ = [x**2 for x in datapoint[1]]
            else:
                square_addition = [x**2 for x in datapoint[1]]
                self.SUMSQ = [sum(x) for x in zip(self.SUMSQ, square_addition)]
                
            if self.d == 0:
                self.d = len(datapoint[1])
                
            self.cluster_label.append(datapoint[0])
                
    def mergeCluster(self, another_cluster):
        self.NoP += another_cluster.NoP
        self.SUM += another_cluster.SUM
        self.SUMSQ += another_cluster.SUMSQ
        # append the label_list to the current list too.
        self.cluster_label += another_cluster.cluster_label
        
                
        
        
def clusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]       
        
        
    
    
    


if __name__ == '__main__':
    start = time.time()
    
    input_file_path = sys.argv[1] # 'hw6_clustering.txt'
    n_cluster = int(sys.argv[2]) #10
    output_file_path = sys.argv[3] #'output.txt'
    
    
    conf = SparkConf().setMaster("local[*]") \
        .setAppName("task") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    start_time = time.time()
    
    output_txt_list = []
    
    ## step 0. Initialization
    DS = []
    CS = []
    RS = []  # note that RS contains a series of points instead of a series of clusters.
    
    ## step 1. Load 20% of the input data randomly
    row_data_rdd = sc.textFile(input_file_path).map(lambda line: line.split(",")).map(lambda line: ((line[0],line[1]), [float(x) for x in line[2:]]))
    
    # [index, group, values, values, ...]
    input_length = row_data_rdd.count()
    # sample_data = row_data_rdd.takeSample(False, int(input_length/5), 553) # 553 is the seed
    weights = [0.2,0.2,0.2,0.2,0.2]
    d1, d2, d3, d4, d5 = row_data_rdd.randomSplit(weights, 553) # 553 is the seed
    rdd_list = [d1, d2, d3, d4, d5]
    sample_data = rdd_list[0].collect()
    
    # print(sample_data[0])
    # print(input_length, len(sample_data))
    
    ## Step 2. Run kmeans with a large K.
    # some pre-processing
    sample_data_index = [item[0] for item in sample_data]
    sample_data_X = [item[1] for item in sample_data]
    
    kmeans = KMeans(n_clusters=n_cluster*5, random_state=0).fit(sample_data_X)
    # print(kmeans.labels_)
    # get those items that only occurred once. 
    label_list = kmeans.labels_.tolist()
    index_to_delete = []
    for label in set(label_list):
        if label_list.count(label) == 1:
            index_to_delete.append(label_list.index(label))
    
    # Add them to RS, also delete them from sample data.
    for i in sorted(index_to_delete, reverse=True):
        RS.append(sample_data[i])
        del sample_data[i]
    
    # print(RS, len(sample_data))
    
    ## Step 4. Do kmeans again
    sample_data_index = [item[0] for item in sample_data]
    sample_data_X = [item[1] for item in sample_data]
    
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(sample_data_X)
    
    output_dic = {}
    for label in set(kmeans.labels_):
        output_dic[label] = (kmeans.labels_ == label).sum()
    print(output_dic)

    
    
    ## Step 5. Generate DS.
    # to do this, put datapoints in groups first based on their label. 
    sample_data_nparray = np.array(sample_data)
    for label in set(kmeans.labels_):
        data_indices_list = clusterIndicesNumpy(label, kmeans.labels_)
        datapoints = list(sample_data_nparray[data_indices_list])
        DS_cluster = Cluster()
        DS_cluster.addPoints(datapoints)
        DS.append(DS_cluster)
    
    
    ## Step 6. Kmeans to the points in RS. 
    sample_data = RS.copy()
    kmeans_input_index = [item[0] for item in sample_data]
    kmeans_input_X = [item[1] for item in sample_data]
    
    kmeans = KMeans(n_clusters=min(n_cluster*5, len(sample_data)), random_state=0).fit(kmeans_input_X)
    
    label_list = kmeans.labels_.tolist()
    index_to_delete = []
    for label in set(label_list):
        if label_list.count(label) == 1:
            index_to_delete.append(label_list.index(label))
    # clear RS
    RS = []
    # Add them to RS, also delete them from sample data.
    for i in sorted(index_to_delete, reverse=True):
        RS.append(sample_data[i])
        del sample_data[i]
    
    # For the rest of the data, generate CS. 
    if len(sample_data)!=0:
        sample_data_nparray = np.array(sample_data)
        for label in set(kmeans.labels_):
            data_indices_list = clusterIndicesNumpy(label, kmeans.labels_)
            datapoints = list(sample_data_nparray[data_indices_list])
            CS_Cluster = Cluster()
            CS_cluster.addPoints(datapoints)
            CS.append(CS_Cluster)
            
    
    
    print('The intermediate results')
    output_txt_list.append('The intermediate results')
    print('Round 1: '+  str(sum(x.NoP for x in DS)) +',' + str(len(CS)) +',' + str(sum(x.NoP for x in CS)) +',' +str(len(RS)))
    output_txt_list.append('Round 1: '+  str(sum(x.NoP for x in DS)) +',' + str(len(CS)) +',' + str(sum(x.NoP for x in CS)) +',' +str(len(RS)))
    
    ############## Recursive section #####################
            
    for data_group_index in range(1,5):   
        # print('Round ', data_group_index+1)
        ## Step 7. Load another 20% of data randomly
        sample_data = rdd_list[data_group_index].collect()

        ## Step 8. Assign to DS
        for point in sample_data:
            # compare distance with each DS. 
            distance_list = []
            ds_index_list = []
            for ds_index in range(0, len(DS)):
                cluster = DS[ds_index]
                distance = cluster.MahalanobisDistance(point)
                if distance < (len(point[1])**0.5)*2:
                    distance_list.append(distance)
                    ds_index_list.append(ds_index)
            if len(distance_list) != 0:
                min_index = distance_list.index(min(distance_list))
                # assign it to that DS
                DS[ds_index_list[min_index]].addPoints([point])
            ## Step 9. otherwise, assign the points to the nearest CS.
            else:
                distance_list = []
                cs_index_list = []
                for cs_index in range(0, len(CS)):
                    cluster = CS[cs_index]
                    distance = cluster.MahalanobisDistance(point)
                    if distance < (len(point[1])**0.5)*2:
                        distance_list.append(distance)
                        cs_index_list.append(cs_index)
                if len(distance_list) != 0:
                    min_index = distance_list.index(min(distance_list))
                    # assign it to that DS
                    CS[cs_index_list[min_index]].addPoints([point])

                ## Step 10. The rest of points are in RS now.
                else:
                    RS.append(point)
        ## Step 11. Kmeans on RS again.
        # print("Before step11, the length of RS is: ", str(len(RS)))
        sample_data = RS.copy()
        kmeans_input_index = [item[0] for item in sample_data]
        kmeans_input_X = [item[1] for item in sample_data]

        kmeans = KMeans(n_clusters=min(n_cluster*5, len(sample_data)), random_state=0).fit(kmeans_input_X)

        label_list = kmeans.labels_.tolist()
        index_to_delete = []
        CS_labels = []
        for label in set(label_list):
            if label_list.count(label) == 1:
                index_to_delete.append(label_list.index(label))
            elif label_list.count(label) > 1:
                CS_labels.append(label)
        # clear RS
        RS = []
        # Add them to RS, also delete them from sample data.
        for i in sorted(index_to_delete, reverse=True):
            RS.append(sample_data[i])

        # print('RS, CS candidate label size', len(RS), len(CS_labels))
        # For the rest of the data, generate CS. 
        if len(CS_labels)!=0:
            #print("Still have . Continue go generate CS")
            sample_data_nparray = np.array(sample_data)
            #print(sample_data, sample_data_nparray)
            for label in set(CS_labels):
                data_indices_list = clusterIndicesNumpy(label, kmeans.labels_)
                datapoints = list(sample_data_nparray[data_indices_list])
                CS_Cluster = Cluster()
                CS_Cluster.addPoints(datapoints)
                CS.append(CS_Cluster)

        ## Step 12. Merge CS clusters.
        # somehow I need to refert to:https://datascience.stackexchange.com/questions/11364/mahalanobis-distance-between-two-clusters
        # calculate the distance between two clusters.
        continue_merge = True
        while continue_merge:
            continue_merge, CS = ClusterMerge(CS)
        
        #print('Checking NoP in each cluster.')
        #print([x.NoP for x in DS])
        print('Round %s: '%str(data_group_index+1) + str(sum(x.NoP for x in DS)) +',' + str(len(CS)) +',' + str(sum(x.NoP for x in CS)) +',' +str(len(RS)))
        output_txt_list.append('Round %s: '%str(data_group_index+1) + str(sum(x.NoP for x in DS)) +',' + str(len(CS)) +',' + str(sum(x.NoP for x in CS)) +',' +str(len(RS))) 
    # In the end, merge DS with CS again. 
    # Note that in this step, only merge CS to DS but not DS together. 
    remaining_CS = ClusterMergeCSDS(DS, CS)
    
    
    #print('Final result: ', str(sum(x.NoP for x in DS)) +',' + str(len(CS)) +',' + str(sum(x.NoP for x in CS)) +',' +str(len(RS)))
    sc.stop()
    
    
    ##### Lastly, write the output file.
    output_list = []
    
    
    for i in range(0, len(DS)):
        cluster = DS[i]
        for item in cluster.cluster_label:
            output_list.append((int(item[0]), item[1], i))  # index, original label, clustered label.
        
    final_CS_index = []
    if len(CS)!= 0:
        for i in range(0, len(CS)):
            cluster = CS[i]
            for item in cluster.cluster_label:
                if int(item[0]) not in final_CS_index:
                    output_list.append((int(item[0]), item[1], -1))  # index, original label, clustered label.
                    final_CS_index.append(int(item[0]))
    for point in RS:
        if int(point[0][0]) not in final_CS_index:
            output_list.append((int(point[0][0]), point[0][1], -1))
        
    ## now the output list can be finally printed out. After that, re-order and check accuracy.
    
    output_list = list(set(output_list))
    output_list.sort()
    
    output_txt_list.append('\nThe clustering results:')
    for item in output_list:
        output_txt_list.append(str(item[0])+','+str(item[1]))
    
    
    
    print("Duration: ", str(time.time() - start_time))
    
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(output_txt_list))
    
    ## write the final result
    
    
    
            
    
