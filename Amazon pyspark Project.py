# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ##**Project Name : Amazon E-commerce Data Analysis**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #**Project Objective**
# MAGIC
# MAGIC The objective of this project is to analyze an Amazon dataset to understand its structure, preprocess the data, perform exploratory data analysis, create visualizations, and develop a recommendation system. This project will provide insights into Amazon's e-commerce operations and demonstrate various data analysis techniques.

# COMMAND ----------

# MAGIC %md
# MAGIC ###**Introduction to data**
# MAGIC **Importing Libraries and Dataframe**

# COMMAND ----------

#importing required Libaries
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg,count,desc,when,col,length

# COMMAND ----------

#Creating df1 By my Amazon.csv
df1 = spark.read.format("csv")\
.option("header", "true")\
.option('inferschema',"true")\
.option("mode", "PERIMISSIVE")\
.load("dbfs:/FileStore/shared_uploads/nikhilarjune2@gmail.com/amazon.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Data Inspection

# COMMAND ----------

#Display data
df1.display(5)

# COMMAND ----------

#schema 
df1.printSchema()

# COMMAND ----------

# Handle missing values if any
df1 = df1.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ##**Exploratory Data Analysis (EDA)**

# COMMAND ----------

# Calculate summary statistics
df1.describe().display()

# COMMAND ----------

# Count the number of products in each category
df1.groupBy("category").count().display()

# COMMAND ----------

#product name 
df1.select('product_name').display()

# COMMAND ----------



# COMMAND ----------

#What is the Top rated product\ Top prodcut by rating  
top_rated_product = df1.groupBy('product_id',"product_name").agg(avg('rating').alias('avg_rating')).orderBy(desc('avg_rating')).limit(10)
top_rated_product.display()

# COMMAND ----------

#most reviewed product 
most_reviewed_product = df1.groupBy('product_id','product_name').count().orderBy(desc('count')).limit(10)
most_reviewed_product.display()

# COMMAND ----------

#Discount Analysis
discount_analysis = df1.groupBy('category').agg(avg('discount_percentage').alias('avg_discount'))
discount_analysis.display()

# COMMAND ----------

#User Engagment
user_engagement = df1.groupBy('product_id').agg(avg('rating').alias('avg_rating'),count('rating').alias('rating_count'))
user_engagement.display()

# COMMAND ----------

#creating temp table from df1
df1.createOrReplaceTempView('amazon_Sales_table')

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from amazon_Sales_table

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from amazon_Sales_table order by product_id desc limit 10
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###**Data Visualization**

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df1.toPandas()

# COMMAND ----------

# Plotting example: Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['rating'], bins=10, kde=True)
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##**Results**
# MAGIC Data Preprocessing: Successfully cleaned the dataset, ensuring no missing values in critical columns (user_id, product_id, rating) and correctly casting data types.
# MAGIC
# MAGIC ###**Exploratory Data Analysis:**
# MAGIC
# MAGIC Generated summary statistics for the dataset.
# MAGIC
# MAGIC Identified patterns in product ratings and categories.
# MAGIC
# MAGIC Visualized the distribution of product ratings.
# MAGIC
