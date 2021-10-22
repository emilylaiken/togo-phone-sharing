'''
Utility functions for anonymizing, processing, and querying aggregated CDR data

author: Emily Aiken
date: April 2020

Credits:
assign_home_locations comes from WB implementation https://github.com/worldbank/covid-mobile-data

'''

# Imports
import sys
import os
import datetime
import shutil
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, Window

def get_spark_session():
    '''
    Gets or creates spark session, with context and logging preferences set
    '''
    # Build spark session
    spark = SparkSession \
        .builder \
        .appName("mm") \
        .config("spark.sql.files.maxPartitionBytes", 64 * 1024 * 1024) \
        .config("spark.driver.memory", '50g') \
        .config("spark.driver.maxResultSize", "2g")\
        .getOrCreate()
    # Add dependencies used in UDF
    # spark.sparkContext.addPyFile("/home/em/covid/dependencies/hashids.py")
    # Change logging to just error messages
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def save_df(df, outfname, sep=','):
    ''' 
    Saves spark dataframe to csv file, using work-around to deal with spark's automatic partitioning and naming
    '''
    outfolder = outfname[:-4]
    df.repartition(1).write.csv(path=outfolder, mode="overwrite", header="true", sep=sep)
    # Work around to deal with spark automatic naming
    old_fname = [fname for fname in os.listdir(outfolder) if fname[-4:] == '.csv'][0]
    os.rename(outfolder + '/' + old_fname, outfname)
    shutil.rmtree(outfolder)

def save_parquet(df, outfname):
    '''
    Save spark dataframe to parquet file
    '''
    df.write.mode('overwrite').parquet(outfname)

def delete_dir(dir):
    '''
    Deletes a directory if it already exists
    '''
    if os.path.isdir(dir):
        print('Overwriting directory ' + dir)
        shutil.rmtree(dir)

def str_to_date(indate):
    '''
    Converts string of form YYYY-MM-DD to date type (rather than relying on built-in ones which use datetime)
    '''
    return datetime.date(int(indate.split('-')[0]), int(indate.split('-')[1]), int(indate.split('-')[2]))

def assign_home_locations(df, frequency, geo):
    '''
    Assign home locations to those who place calls in CDR
    Arguments: 
        df - Spark dataframe with appropriate schema
        frequency - string for frequency of calculation of home location from {'day', 'week', 'month', 'year'}
        geo - String for geographic granularity of assignment (for Togo, from {'prefecture', 'canton', 'site_id'})
    '''
    df = df.withColumnRenamed(geo, 'r')
    # Get calls for each user each day, ordered in time
    user_day = Window\
        .orderBy(desc_nulls_last('datetime'))\
        .partitionBy('caller_msisdn', 'day')
    # Get calls for each user for each time period (day/week/month/year), ordered in time
    user_frequency = Window\
        .orderBy(desc_nulls_last('last_region_count'))\
        .partitionBy('caller_msisdn', frequency)
    result = df.na.fill({'r' : 99999})\
        .withColumn('last_timestamp', first('datetime').over(user_day))\
        .withColumn('last_region', when(col('datetime') == col('last_timestamp'), 1).otherwise(0))\
        .orderBy('datetime')\
        .groupby('caller_msisdn', frequency, 'r')\
        .agg(sum('last_region').alias('last_region_count'))\
        .withColumn('modal_region', when(first('last_region_count').over(user_frequency) == col('last_region_count'),1).otherwise(0))\
        .where(col('modal_region') == 1)\
        .groupby('caller_msisdn', frequency)\
        .agg(last('r').alias('home_region'), last('last_region_count').alias('confidence'))
    return result