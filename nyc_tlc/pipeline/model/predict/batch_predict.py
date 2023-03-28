import pickle
import mlflow
from pyspark.sql.functions import struct, col

from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import Double
from nyc_tlc.pipeline.model.data import data

@node(
    type="pyspark",
    description="Prediction for big volume of data (batch)",
    group="prediction.batch",
    spark_context=True,
    dependencies=[
        data.select("trip_id", "pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown")
    ],
    output=Schema(
        data.output.get("trip_id"),
        Column('scaled_predicted_tip_amount', Double(), "Predicted tip amount (scaled)"),
    )
)
def batch_predict(spark, data, run_id="11d4b6c8e1014fccad07090489707c77"):
    
    logged_model = f'runs:/{run_id}/random_forest'
    
    # # Load model as a Spark UDF. Override result_type if the model does not return double values.
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')
    
    # Load model as a Spark UDF. Override result_type if the model does not return double values.
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='double')

    # Predict on a Spark DataFrame.
    x_cols = ["pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown"]
    data = data.withColumn('scaled_predicted_tip_amount', loaded_model(struct(*map(col, data.select(x_cols).columns))))
    return data