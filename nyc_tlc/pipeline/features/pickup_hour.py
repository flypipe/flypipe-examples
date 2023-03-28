from flypipe import node
from flypipe.datasource.spark import Spark
from flypipe.schema import Schema, Column
from flypipe.schema.types import Integer
from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip

@node(
    type="pandas_on_spark",
    description="Pickup hour of the day (24hr clock)",
    tags=["yellow_taxi", "feature"],
    group="pipeline.feature",
    dependencies=[
        silver_yellow_trip.select("trip_id", "pickup_datetime").alias("df")
    ],
    output=Schema(
        silver_yellow_trip.output.get("trip_id") ,
        Column("pickup_hour", Integer(), "Pickup hour of the day (i.e. 23)")
    )
)
def pickup_hour(df):
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    return df
