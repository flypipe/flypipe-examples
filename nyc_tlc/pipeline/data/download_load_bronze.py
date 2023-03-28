import os
from flypipe import node
from urllib import request
import pyspark.sql.functions as F

@node(
    type='pyspark',
    description='Download data from NYC TLC data (https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet)',
    tags=['pipeline', 'bronze'],
    spark_context=True,
    group="pipeline.data"
)
def download_load_bronze(spark, year=2023, month=1, write=False):
    
    if not write:
        return spark.sql("select 1")
    
    # Download data
    os.makedirs('/dbfs/tmp/nyc_taxi/data', exist_ok=True)
    
    month_str = "0" + str(month) if month <= 9 else str(month)
    parquet_file = f"tmp/nyc_taxi/data/yellow_tripdata_{year}_{month_str}.parquet"

    if not os.path.exists(f"/dbfs/{parquet_file}"):
        request.urlretrieve(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month_str}.parquet", f"/dbfs/{parquet_file}")    
    data = spark.read.parquet(f"dbfs:/{parquet_file}")
    data = data.withColumn("metadata_year", F.lit(year))
    data = data.withColumn("metadata_month", F.lit(month))
    
    # Save to Table
    spark.sql("CREATE DATABASE IF NOT EXISTS nyc_tlc")
    
    if spark.catalog.tableExists("nyc_tlc.bronze_yellow_trip"):
        spark.sql(f"DELETE FROM nyc_tlc.bronze_yellow_trip WHERE metadata_year = {year} and metadata_month = {month}")            
    data.write.mode("append").saveAsTable("nyc_tlc.bronze_yellow_trip")
        
    return data
