from flypipe import node
from flypipe.schema import Schema
from flypipe.datasource.spark import Spark
from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip
from nyc_tlc.pipeline.features.pickup_hour import pickup_hour
from nyc_tlc.pipeline.features.payment_type import payment_type

@node(
    type="pandas_on_spark",
    dependencies=[
        silver_yellow_trip.select("trip_id", "tip_amount", "pickup_taxi_zone_id").alias("df"),
        pickup_hour.select("trip_id", "pickup_hour"),
        payment_type.select("trip_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown")
    ],
    output=Schema(
        silver_yellow_trip.output.get("trip_id"),
        silver_yellow_trip.output.get("tip_amount"),
        silver_yellow_trip.output.get("pickup_taxi_zone_id"),
        pickup_hour.output.get("pickup_hour"),
        payment_type.output.get("payment_cash"),
        payment_type.output.get("payment_no_charge"),
        payment_type.output.get("payment_credit_card"),
        payment_type.output.get("payment_dispute"),
        payment_type.output.get("payment_unknown")
        
    )
)
def data(df, pickup_hour, payment_type):
    df = (
        df.merge(pickup_hour, on="trip_id", how="left")
        .merge(payment_type, on="trip_id", how="left")
    )
    return df