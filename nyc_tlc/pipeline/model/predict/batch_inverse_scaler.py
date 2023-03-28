import pickle
import mlflow
import pandas as pd
from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import Decimal
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, DecimalType

from nyc_tlc.pipeline.model.data import data
from nyc_tlc.pipeline.model.predict.batch_predict import batch_predict

MODEL_DIR = "nyc_tlc_tip_amount"

@node(
    type="pyspark",
    description="Scale data",
    group="prediction.batch",
    dependencies=[
        batch_predict.select("trip_id", "scaled_predicted_tip_amount").alias('data')
    ],
    output=Schema(
        batch_predict.output.get("trip_id"),
        Column('predicted_tip_amount', Decimal(13,2), "Predicted Tip Amount")
    )
)
def batch_inverse_scaler(data, run_id="29bc319133ee4d1f8926489c1939644a"):    

    scaler = None
    with open(f"/dbfs/ml/models/{MODEL_DIR}/{run_id}/artifacts/random_forest/artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)    

    def inverse_scale(x):
        return float(scaler.inverse_transform([x])[0])
    
    # Converting function to UDF 
    convertUDF = F.udf(lambda z: inverse_scale(z), DoubleType())
    data = data.withColumn('predicted_tip_amount', convertUDF(F.col('scaled_predicted_tip_amount')).cast(DecimalType(13,2)))
    
    return data