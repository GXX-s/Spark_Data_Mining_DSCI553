import itertools
import sys
import time
import random

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession



class CustomGraphFrame(object):
    # A custom graphframe implementation

    def __init__(self, edges, vertexes):
        
        # Initialize the edges and vertexes, then do some copy
        self.edges = edges
        self.vertexes = vertexes
        self.original_vertexes = vertexes
        self.original_edges = edges
        self.m = self._count_edges(edges)
        
        # create a dictionary to store the vertex weights
        self.vertex_weight_dict = dict()
        # initialize
        for vertex in self.vertexes:
            self.vertex_weight_dict.setdefault(vertex, 1)
        
        
        
        
        # build adjacent matrix for original edges
        edge_set = set()
        for start_node, end_nodes in edges.items():
            for end_node in end_nodes:
                edge_set.add(re_order(start_node, end_node))
        self.matrix = edge_set

        # variable using for get betweenness
        self.betweenness_result_dict = dict()
        self.sorted_betwenness_list = None

        # variable using for get modularity
        self.best_communities = None


    def _count_edges(self, edges):
        """
        Count the number of edges
        """
        visited_node = set()
        output_count = 0
        for start, end in edges.items():
            for end_node in end:
                key = re_order(start, end_node)
                if key not in visited_node:
                    visited_node.add(key)
                    output_count += 1
        return output_count

    def _construct_tree(self, root):
        # A custom implementation build tree method for BFS
        tree = dict()
        tree[root] = (0, list())

        visited = set()

        to_visit = list()
        to_visit.append(root)
        
        # go through the nodes to form the tree
        while len(to_visit) > 0:
            parent = to_visit.pop(0)
            visited.add(parent)
            for child in self.edges[parent]:
                if child not in visited:
                    visited.add(child)
                    tree[child] = (tree[parent][0] + 1, [parent])
                    to_visit.append(child)
                elif tree[parent][0] + 1 == tree[child][0]:
                    tree[child][1].append(parent)

        return {key: value for key, value in sorted(tree.items(), key=lambda pair: -pair[1][0])}

    def _traverse_tree(self, tree_dict):
        """
        Traverse the tree, then get the weights for each edge
        """
        # first find number of paths given the tree_dict
        height_dict = dict()
        path_dict = dict()
        for child_node, level_parents in tree_dict.items():
            height_dict.setdefault(level_parents[0], []).append((child_node, level_parents[1]))

        for level in range(0, len(height_dict.keys())):
            for (child_node, parent_list) in height_dict[level]:
                if len(parent_list) > 0:
                    path_dict[child_node] = sum([path_dict[parent]
                                                          for parent in parent_list])
                else:
                    path_dict[child_node] = 1
        
        copy_weight_dict = self.vertex_weight_dict.copy()
        
        output_dict = dict()
        for key, value in tree_dict.items():
            # check remaining tree depth
            if len(value[1]) > 0:
                den = sum([path_dict[parent_node] for parent_node in value[1]])
                for parent_node in value[1]:
                    temp = re_order(key, parent_node)
                    contribution = float(float(copy_weight_dict[key]) * int(path_dict[parent_node]) / den)
                    output_dict[temp] = contribution
                    # finally update the weight
                    copy_weight_dict = update_dict_with_inc(copy_weight_dict, parent_node, contribution)

        return output_dict



    def getCommunities(self):

        # initialize a min-max value first
        max_mod = float("-inf")
        # print(self.sorted_betwenness_list)
        if len(self.sorted_betwenness_list) > 0:
            self._cut_edges(self.sorted_betwenness_list)
            self.best_communities, max_mod = self._getModularity()
            self.sorted_betwenness_list = self.getBetweenness()

        # then perform edge cutting and calculate modularity.
        while True:
            self._cut_edges(self.sorted_betwenness_list)
            communities, current_modularity = self._getModularity()
            self.sorted_betwenness_list = self.getBetweenness()
            if current_modularity < max_mod:
                break
            else:
                self.best_communities = communities
                max_mod = current_modularity
                
        output = sorted(self.best_communities, key=lambda com: (len(com), com[0], com[1]))

        return output

    def _cut_edges(self, edge_tuple_list):
        """
        Remove the edge with the highest betweeness value
        """
        
        edge_pair = edge_tuple_list[0][0]
        
        # In this implementation, all the edges with the highest betweenness are cut at once
        
        if self.edges[edge_pair[0]] is not None:
            try:
                self.edges[edge_pair[0]].remove(edge_pair[1])
            except: # handle the key value errors 
                pass

        if self.edges[edge_pair[1]] is not None:
            try:
                self.edges[edge_pair[1]].remove(edge_pair[0])
            except: # handle the key value errors 
                pass
    

    def _getModularity(self):
        """
        get modularity based on the previous betwenness result
        """
        # first find the communities
        communities = list()  # store the community result
        to_visit = list()  
        temp_node_set = set()  
        visited = set() 
        
        # add some randomness to pick a root
        selected_root = self.vertexes[random.randint(0, len(self.vertexes) - 1)]
        temp_node_set.add(selected_root)
        to_visit.append(selected_root)
        # do the cutting loop.
        while len(self.vertexes) != len(visited) :
            while len(to_visit) > 0:
                parent = to_visit.pop(0)
                temp_node_set.add(parent)
                visited.add(parent)
                for child in self.edges[parent]:
                    if child not in visited:
                        temp_node_set.add(child)
                        to_visit.append(child)
                        visited.add(child)
            sorted_temp_set =  sorted(temp_node_set)           
            communities.append(sorted_temp_set)
            temp_node_set = set()
            if len(visited) < len(self.vertexes):
                # pick one from rest of unvisited nodes
                to_visit.append(set(self.vertexes).difference(visited).pop())

        # 2. get modularity in two steps. First, count original edge number. Then, create a adjacent matrix.
        temp_sum_result = 0
        for com in communities:
            for node_pair in itertools.combinations(list(com), 2):
                temp_k = re_order(node_pair[0], node_pair[1])
                k1 = len(self.edges[node_pair[0]])
                k2 = len(self.edges[node_pair[1]])
                if temp_k in self.matrix:
                    adj = 1 
                else:
                    adj = 0
                add_result = float(adj - (k1 * k2 / (2 * self.m)))
                temp_sum_result += add_result
        return communities, float(temp_sum_result / (2 * self.m))
    
    
    

    
    def getBetweenness(self):
        self.betweenness_result_dict = dict()
        for node in self.vertexes:
            tree = self._construct_tree(root=node)
            temp_dict = self._traverse_tree(tree)
            self.betweenness_result_dict = extend_dict(self.betweenness_result_dict, temp_dict)
        self.betweenness_result_dict = dict(map(lambda k_v: (k_v[0], float(k_v[1] / 2)),self.betweenness_result_dict.items()))
        self.sorted_betwenness_list = sorted(self.betweenness_result_dict.items(), key=lambda k_v: (-k_v[1], k_v[0][0]))
        return self.sorted_betwenness_list
    
    
    


    
def re_order(i, j):
    if i < j:
        return (i, j) 
    else:
        return (j, i)


def save_to_txt(result, path):
    with open(path, 'w+') as f:
        for id in result:
            f.writelines(str(id)[1:-1] + "\n")



def update_dict_with_inc(obj, key, inc):
    old = obj[key]
    obj[key] = float(old + inc)
    return obj


def extend_dict(obj, inc_dic):
    for k, v in inc_dic.items():
        if k in obj.keys():
            obj = update_dict_with_inc(obj, k, v)
        else:
            obj[k] = v
    return obj



if __name__ == '__main__':
    start = time.time()
    # define input variables
    #filter_threshold = "7"
    #input_path = "ub_sample_data.csv"
    #betweenness_file_path = "task2_bet.txt"
    #community_file_path = "task2_com.txt"

    filter_threshold = sys.argv[1]
    input_path = sys.argv[2]
    betweenness_file_path = sys.argv[3]
    community_file_path = sys.argv[4]

    conf = SparkConf().setMaster("local") \
        .setAppName("task2") \
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

            
            
    ############# 

    
    vertexes = sc.parallelize(sorted(list(storage_set))).collect()
    #print(vertexes)
    # ['0FMte0z-repSVWSJ_BaQTg', '0FVcoJko1kfZCrJRfssfIA', '0KhRPd66BZGHCtsb9mGh_g', ....] 
    
    edges = sc.parallelize(edge_list).groupByKey().mapValues(lambda uid: sorted(list(set(uid)))).collectAsMap()
    # print(edges)
    # {'39FT2Ui8KUXwmUt6hnwy-g': ['0FVcoJko1kfZCrJRfssfIA', '1KQi8Ymatd4ySAd4fhSfaw', '79yaBDbLASfIdB-C2c8DzA', 'B0ENvYKQdNNr1Izd2r-BAA', 'ChshgCKJTdIDg17JKtFuJw', 'DKolrsBSwMTpTJL22dqJRQ', 'JM0GL6Dx4EuZ1mprLk5Gyg', 'KLB3wIYUwKDPMbijIE92vg', 'OoyQYSeYNyRVOmdO3tsxYA', 'PE8s8ACYABRNANI-T_WmzA', 'R4l3ONHzGBakKKNo4TN9iQ', 'Uo5dPwoDpYBzOnmUnjxJ6A', '_Pn-EmWO-pFPFg81ZIEiDw', '_VTEyUzzH92X3w-IpGaXVA', 'ay4M5J28kBUf0odOQct0BA', 'bHufZ2OTlC-OUxBDRXxViw', 'bSUS0YcvS7UelmHvCzNWBA', 'dTeSvET2SR5LDF_J07wJAQ', 'dzJDCQ5vubQBJTfYTEmcbg', 'mu4XvWvJOb3XpG1C_CHCWA', 'qtOCfMTrozmUSHWIcohc6Q', 'sdLns7062kz3Ur_b8wgeYw', 'zBi_JWB5uUdVuz3JLoAxGQ'],
    
    custom_graphframe = CustomGraphFrame(edges, vertexes)
    betweenness = custom_graphframe.getBetweenness()
    communities = custom_graphframe.getCommunities()
    
    save_to_txt(betweenness, betweenness_file_path)
    save_to_txt(communities, community_file_path)
    
    ## get execution time
    print("Duration: ", str(time.time() - start)) 
    sc.stop()
    
