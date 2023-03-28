from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import String
from nyc_tlc.pipeline.model.train.downsample import downsample
import numpy as np
import pyspark.pandas as ps

@node(
    type="pandas_on_spark",
    description="Tag data into train, test and validation data",
    dependencies=[
        downsample.select("trip_id", "pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown", "tip_amount").alias("data")
    ],
    tags=["model", "train"],
    group="model.train",
    output=Schema(
        downsample.output.get("trip_id"), 
        Column("data_type", String(), "training data type (train, test, validation)"),
        downsample.output.get("pickup_hour"), 
        downsample.output.get("pickup_taxi_zone_id"),
        downsample.output.get("payment_cash"), 
        downsample.output.get("payment_no_charge"), 
        downsample.output.get("payment_credit_card"), 
        downsample.output.get("payment_dispute"), 
        downsample.output.get("payment_unknown"),
        downsample.output.get("tip_amount"),
    )
)
def split(data):
    train, test = np.split(data, [int(.7*len(data))])
    train['data_type'] = "train"
    test['data_type'] = "test"

    return ps.concat([train, test])