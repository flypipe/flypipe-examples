from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import String, Float
from nyc_tlc.pipeline.model.train.train import train
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow

@node(
    type="pandas",
    description="Calculate training metrics",
    tags=["model", "train"],
    group="model.train",
    dependencies=[train.select("trip_id", "data_type", "tip_amount", "tip_amount_predicted").alias("data")],
    output=Schema(
        Column("data_type", String(), "data type"),
        Column("metric", String(), "metric name"),
        Column("value", Float(), "metric value"),
    )
)
def metrics(data):
    df = pd.DataFrame(columns=['data_type', 'metric', 'value'])

    for data_type in ['train', 'test']:
        data_ = data[data['data_type'] == data_type]
        row = df.shape[0]
        metric_name = f'{data_type}_mean_square_error'
        metric_value = round(mean_squared_error(data_['tip_amount'], data_['tip_amount_predicted']),4)
        df.loc[row, 'data_type'] = data_type
        df.loc[row, 'metric'] = metric_name
        df.loc[row, 'value'] = metric_value

        mlflow.log_metric(metric_name, metric_value)
        
    return df