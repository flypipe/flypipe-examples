# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Clean previous checkpoints & data

# COMMAND ----------

if spark.catalog.tableExists("nyc_tlc.gold_yellow_trip_tip_prediction"):
    spark.sql("drop table nyc_tlc.gold_yellow_trip_tip_prediction")

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/user/hive/warehouse/nyc_tlc.db/gold_yellow_trip_tip_prediction

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /dbfs/tmp/streaming/checkpoints/nyc_tlc

# COMMAND ----------

from nyc_tlc.pipeline.data.download_load_bronze import download_load_bronze

download_load_bronze.run(
    spark, 
    parameters={
        download_load_bronze: {'year': 2022, 'month': 11, 'write': True},
    }
)

# COMMAND ----------

display(spark.sql("select metadata_year, metadata_month, count(1) from nyc_tlc.bronze_yellow_trip group by 1,2"))

# COMMAND ----------

from nyc_tlc.pipeline.data.download_load_bronze import download_load_bronze
from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip
from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler

def transform(df_trips, batch_id):
    

    df = batch_inverse_scaler.run(
        spark,
        parameters={
            silver_yellow_trip: {'write': True}
        },
        inputs={
            download_load_bronze: df_trips,            
        }
    )

    output_table = "nyc_tlc.gold_yellow_trip_tip_prediction"
    if spark.catalog.tableExists(output_table):
        df.createOrReplaceTempView("updates")
       
        df._jdf.sparkSession().sql(f"""
            MERGE INTO {output_table} t
            USING updates s
            ON s.trip_id = t.trip_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
    else:

        (
            df
            .write
            .format("delta")
            .mode("append")
            .saveAsTable(output_table)
            
        )
        
from time import time
app_id = "nyc_tlc_prediction_tip_amount"
version = int(time() * 1000)

(
    spark
    .readStream
    .option("ignoreDeletes", "true")
    .format("delta")
    .table("nyc_tlc.bronze_yellow_trip")    
    .writeStream
    .option("txnVersion", version)
    .option("txnAppId", app_id)
    .trigger(availableNow=True) #Comment for continuous processing
    .option("checkpointLocation", "dbfs:/tmp/streaming/checkpoints/nyc_tlc")
    .foreachBatch(transform)
    .start()
)

# COMMAND ----------


from time import sleep
output_table = "nyc_tlc.gold_yellow_trip_tip_prediction"

initial_count = 0
if spark.catalog.tableExists(output_table):
    initial_count = spark.sql(f"select count(*) as quantity from {output_table}").collect()[0].quantity    

while True:
    try:
        final_count = spark.sql(f"select count(*) as quantity from {output_table}").collect()[0].quantity
        if final_count != initial_count:
            print(f"Inserted {final_count - initial_count} records (total {final_count})")
            break
        else:
            print(f"Number of records {initial_count} not changed")
            sleep(10)
    except Exception as e:
        print(f"Number of records {initial_count} not changed")
        sleep(10)

