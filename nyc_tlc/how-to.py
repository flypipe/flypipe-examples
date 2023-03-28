# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # ML Model Pipeline using [New York City Taxi & Limousine Comission Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
# MAGIC 
# MAGIC This pipeline contains 5 subgraphs:
# MAGIC 
# MAGIC 1. **pipeline.data**: downloads and transforms Yellow Taxi Trips into table `nyc_tlc.silver_yellow_trip`
# MAGIC 2. **pipeline.feature**: feature tranformations
# MAGIC 3. **model.train**: downsample, train/test split, applies standard scaler, train RandomForestRegressor to predict passengers `tip_amount` and calculates r2 square metrics
# MAGIC 4. **prediction.batch**: calculates predictions and inverse scale the predictions (used for big data volume)
# MAGIC 5. **prediction.online**: calculates predictions and inverse scale the predictions (used for small amount of data, ideal for low latency requests)

# COMMAND ----------

# MAGIC %pip install flypipe==3.0.2

# COMMAND ----------

# DBTITLE 1,Fix: Known error `cannot import scape from Jinja2`
# MAGIC %pip install Jinja2==3.0.3

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Pipeline Graph

# COMMAND ----------

from nyc_tlc.pipeline.model.graph import graph
displayHTML(graph.html())

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Downloading and transforming Yellow Taxi trips

# COMMAND ----------

from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip
from nyc_tlc.pipeline.data.download_load_bronze import download_load_bronze

year = 2022
month = 11

silver_yellow_trip.run(
    spark, 
    parameters={
        download_load_bronze: {'year': year, 'month': month, 'write': True},
        silver_yellow_trip: {'year': year, 'month': month, 'write': True}
    }
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select metadata_year, metadata_month, count(1) from nyc_tlc.silver_yellow_trip group by 1,2;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Training the model

# COMMAND ----------

import os
import mlflow
from nyc_tlc.pipeline.model.train.metrics import metrics

RUN_ID = None
MODEL_DIR = "nyc_tlc_tip_amount"
EXPERIMENT_DIR = f"/Shared/{MODEL_DIR}"
ARTIFACT_DIR = f"dbfs:/ml/models/{MODEL_DIR}"

try:
    mlflow.end_run()
except Exception as e:
    pass

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_DIR)
    experiment_id = experiment.experiment_id
except AttributeError as e:
    experiment_id = mlflow.create_experiment(EXPERIMENT_DIR, artifact_location=ARTIFACT_DIR)

print(f"Experiment '{EXPERIMENT_DIR}': {experiment_id}")
mlflow.start_run(experiment_id=experiment_id)

try:
    os.makedirs(os.path.join(ARTIFACT_DIR, mlflow.active_run().info.run_id).replace("dbfs:", "/dbfs"), exist_ok=True)
    print(f"Active run_id: {mlflow.active_run().info.run_id}")
    RUN_ID = mlflow.active_run().info.run_id
    df = metrics.run(spark)
finally:
    mlflow.end_run()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Batch Predictions

# COMMAND ----------


from nyc_tlc.pipeline.model.data import data
from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler

displayHTML(online_inverse_scaler.html())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Default RUN_ID

# COMMAND ----------

from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler

df = batch_inverse_scaler.run(spark)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Custom RUN_ID

# COMMAND ----------

from nyc_tlc.pipeline.model.predict.batch_predict import batch_predict
from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler

df = batch_inverse_scaler.run(
    spark,
    parameters = {
        batch_inverse_scaler: {'run_id': RUN_ID},
        batch_predict: {'run_id': RUN_ID}
    })
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Online Predictions

# COMMAND ----------


from nyc_tlc.pipeline.model.data import data
from nyc_tlc.pipeline.model.predict.online_inverse_scaler import online_inverse_scaler

displayHTML(online_inverse_scaler.html())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Manual Input (Simulating API request)

# COMMAND ----------

import pandas as pd
from decimal import Decimal
from pandas import Timestamp

df_input = pd.DataFrame([
    {
        'trip_id': 2022110, 
        'metadata_year': 2022, 
        'metadata_month': 11, 
        'provider_id': 'Creative Mobile Technologies', 
        'pickup_taxi_zone_id': 50, 
        'pickup_datetime': Timestamp('2022-11-01 00:51:22'), 
        'dropoff_taxi_zone_id': 50, 
        'dropoff_datetime': Timestamp('2022-11-01 00:56:24'), 
        'number_passenger': 1, 
        'trip_distance_miles': 0.6000000238418579, 
        'rate_type': 'standard rate', 
        'payment_type': 'credit card', 
        'fare_amount': Decimal('4.50'), 'extra': Decimal('0.50'), 'mta_tax': Decimal('0.50'), 'improvement_surcharge': Decimal('0.30'), 
        'tip_amount': Decimal('0.00'), 'tolls_amount': Decimal('0.00'), 'total_amount': Decimal('5.80'), 
        'congestion_surcharge': Decimal('0.00'), 'airport_fee': Decimal('0.00')}])

display(df_input)

# COMMAND ----------


from nyc_tlc.pipeline.model.data import data
from nyc_tlc.pipeline.model.predict.online_predict import online_predict
from nyc_tlc.pipeline.model.predict.online_inverse_scaler import online_inverse_scaler
from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip

df_ = online_inverse_scaler.run(
    pandas_on_spark_use_pandas=True,
    inputs={
        silver_yellow_trip: df_input,
    },
    parameters = {
        online_predict: {'run_id': RUN_ID},
        online_predict: {'run_id': RUN_ID}
    })

display(df_)
