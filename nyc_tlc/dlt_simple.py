# Databricks notebook source
# MAGIC %pip install flypipe==3.0.2 /dbfs/FileStore/tables/nyc_tlc-1.0.0-py3-none-any.whl --force-reinstall

# COMMAND ----------

import sys
import os
import dlt

@dlt.table(
    comment="Loading data from nyc_tlc.bronze_yellow_trip"
)
def dlt_bronze_yellow_trip():
    return (
        spark.read.format("delta")
        .load("dbfs:/user/hive/warehouse/nyc_tlc.db/bronze_yellow_trip")
    )

@dlt.table(
    comment="Prediction"
)
def predictions():
    from nyc_tlc.pipeline.data.download_load_bronze import download_load_bronze
    from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler
    
    return (
        batch_inverse_scaler
        .run(
            spark,
            inputs={
                download_load_bronze: dlt.read("dlt_bronze_yellow_trip")
            }
        )
    )

