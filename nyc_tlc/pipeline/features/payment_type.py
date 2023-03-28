from flypipe import node
from flypipe.datasource.spark import Spark
from flypipe.schema import Schema, Column
from flypipe.schema.types import Boolean
from nyc_tlc.pipeline.data.silver_yellow_trip import silver_yellow_trip
import pyspark.sql.functions as F

@node(
    type="pandas_on_spark",
    description="Payment type of the trip",
    tags=["yellow_taxi", "feature"],
    group="pipeline.feature",
    dependencies=[
        silver_yellow_trip.select("trip_id", "payment_type").alias("df")
    ],
    output=Schema(
        silver_yellow_trip.output.get("trip_id"),
        Column("payment_cash", Boolean(), "Was paid in cash?"),
        Column("payment_no_charge", Boolean(), "Was not charged?"),
        Column("payment_credit_card", Boolean(), "Was paid with credit card?"),
        Column("payment_dispute", Boolean(), "Payment in dispute?"),
        Column("payment_unknown", Boolean(), "Payment is unknown or not provided?"),
    )
)
def payment_type(df):  
         
    df['payment_cash'] = df['payment_type'] == "cash"
    df['payment_no_charge'] = df['payment_type'] == "no charge"
    df['payment_credit_card'] = df['payment_type'] == "credit card"
    df['payment_dispute'] = df['payment_type'] == "dispute"
    df['payment_unknown'] = ~df['payment_type'].isin(["cash", "no charge", "credit card", "dispute"])
    return df

