import pickle
import mlflow
import pandas as pd
from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import Decimal
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from nyc_tlc.pipeline.model.data import data
from nyc_tlc.pipeline.model.predict.online_predict import online_predict

MODEL_DIR = "nyc_tlc_tip_amount"

@node(
    type="pandas",
    description="Inverse Scale predictions",
    group="prediction.online",
    dependencies=[
        online_predict.select("trip_id", "scaled_predicted_tip_amount").alias('data')
    ],
    output=Schema(
        online_predict.output.get("trip_id"),
        Column('predicted_tip_amount', Decimal(13,2), "Predicted Tip Amount")
    )
)
def online_inverse_scaler(data, run_id="29bc319133ee4d1f8926489c1939644a"):    

    scaler = None
    with open(f"/dbfs/ml/models/{MODEL_DIR}/{run_id}/artifacts/random_forest/artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)    

    data['predicted_tip_amount'] = scaler.inverse_transform(data[['scaled_predicted_tip_amount']])
    return data
