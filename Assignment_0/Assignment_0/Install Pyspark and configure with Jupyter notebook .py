#!/usr/bin/env python
# coding: utf-8

# # PySpark in Jupyter
# There are two ways to get PySpark available in a Jupyter Notebook:
# 
#    Configure PySpark driver to use Jupyter Notebook: running pyspark will automatically open a Jupyter Notebook
# 
# Load a regular Jupyter Notebook and load PySpark using findSpark package
# 
# First option is quicker but specific to Jupyter Notebook, second option is a broader approach to get PySpark available in your favorite IDE.
# 
# ## Method 1 — Configure PySpark driver
# 
# Update PySpark driver environment variables: add these lines to your ~/.bashrc (or ~/.zshrc) file.
# 
# export PYSPARK_DRIVER_PYTHON=jupyter
# 
# export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
# 
# Restart your terminal and launch PySpark again:
# 
# $ pyspark
# 
# Now, this command should start a Jupyter Notebook in your web browser. Create a new notebook by clicking on ‘New’ > ‘Notebooks Python [default]’.
# 
# Copy and paste our Pi calculation script and run it by pressing Shift + Enter.
# 
# 
# 

# ## The second method, use findspark package

# In[1]:


import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName="Pi")





pyspark.__version__




sc.version


sc.pythonVer


sc.master ## url of the cluster of 'local' string to run in local mode of the sparkContext

## loading data with pyspark
rdd = sc.parallelize([1,2,3,4,5])

rdd2 = sc.textFile('test.txt')


rdd2



