


from pyspark import SparkContext
import os

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
sc = SparkContext('local[*]','wordCount')



input_file_path= './test.txt'
textRDD = sc.textFile(input_file_path)

counts = textRDD.flatMap(lambda line:line.split(' ')).map(lambda word: (word,1)).reduceByKey(lambda a,b:a+b).collect()
for each_word in counts:
  print(each_word)




