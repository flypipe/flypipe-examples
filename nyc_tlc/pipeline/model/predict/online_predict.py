import pickle
import mlflow
from pyspark.sql.functions import struct, col

from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import Double
from nyc_tlc.pipeline.model.data import data
import pandas as pd
import numpy as np

MODEL_DIR = "nyc_tlc_tip_amount"

@node(
    type="pandas",
    description="Online predictions using Pandas",
    group="prediction.online",
    dependencies=[
        data.select("trip_id", "pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown")
    ],
    output=Schema(
        data.output.get("trip_id"),
        Column('scaled_predicted_tip_amount', Double(), "Predicted tip amount (scaled)"),
    )
)
def online_predict(data, run_id="11d4b6c8e1014fccad07090489707c77"):
    
    
    logged_model = f'runs:/{run_id}/random_forest'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    x_cols = ["pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown"]
    data['pickup_hour'] = data['pickup_hour'].astype(np.int32)
    data['pickup_taxi_zone_id'] = data['pickup_taxi_zone_id'].astype(np.int32)
    
    data['scaled_predicted_tip_amount'] = loaded_model.predict(data[x_cols])

    return data
