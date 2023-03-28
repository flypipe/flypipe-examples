from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import String
from nyc_tlc.pipeline.model.train.split import split
import numpy as np
import pyspark.pandas as ps
from sklearn.preprocessing import StandardScaler
import mlflow
import os
import pickle

@node(
    type="pandas",
    description="Scale trip_amount",
    tags=["model", "train"],
    group="model.train",
    dependencies=[
        split.select("trip_id", "data_type", "pickup_hour", "pickup_taxi_zone_id", "payment_cash", 
        "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown", "tip_amount").alias("data")
    ],
    output=split.output
)
def scale(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data[['tip_amount']])
    
    if mlflow.active_run():
        artifact_path = mlflow.active_run().info.artifact_uri
        artifact_path = artifact_path.replace("dbfs:", "/dbfs")
        artifact_path = os.path.join(artifact_path, 'random_forest', 'artifacts')
        os.makedirs(artifact_path, exist_ok=True)
        pickle.dump(scaler, open(os.path.join(artifact_path, 'scaler.pkl'), 'wb'))
    
    data[['tip_amount']] = scaler.transform(data[['tip_amount']])
    return data
