from flypipe import node
from flypipe.schema.types import String, Integer, DateTime, Float, Decimal, Long
from flypipe.schema import Schema, Column

from urllib import request
import pyspark.sql.functions as F
from pyspark.sql.types import LongType, StringType

from nyc_tlc.pipeline.data.download_load_bronze import download_load_bronze


@node(
    type='pyspark',
    description='Load and transform data into Silver Layer',
    spark_context=True,
    dependencies=[
        download_load_bronze.alias("data")
    ],    
    group="pipeline.data",
    tags=['pipeline', 'silver'],
    output = Schema([
        Column("trip_id", Long(), "Trip Id"),
        Column("metadata_year", Integer(), "Year of the record from NYC TLC data"),
        Column("metadata_month", Integer(), "Month of the record from NYC TLC data"),
        
        Column("provider_id", String(), "A code indicating the TPEP provider that provided the record. 1= Creative Mobile Technologies, LLC; 2= VeriFone Inc."),

        Column("pickup_taxi_zone_id", Integer(), "TLC Taxi Zone in which the taximeter was engaged"),
        Column("pickup_datetime", DateTime(), "The date and time when the meter was engaged"),
        Column("dropoff_taxi_zone_id", Integer(), "TLC Taxi Zone in which the taximeter was disengaged"),
        Column("dropoff_datetime", DateTime(), "The date and time when the meter was disengaged"),
        Column("number_passenger", Integer(), "Number of passengers"),
        Column("trip_distance_miles", Float(), "The elapsed trip distance in miles reported by the taximeter"),
        Column("rate_type", String(), "rate code in effect at the end of the trip."),
        Column("payment_type", String(), "how the passenger paid for the trip."),
        Column("fare_amount", Decimal(13, 2), "The time-and-distance fare calculated by the meter."),

        Column("extra", Decimal(13, 2), "Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges."),
        Column("mta_tax", Decimal(13, 2), "$0.50 MTA tax that is automatically triggered based on the metered rate in use."),
        Column("improvement_surcharge", Decimal(13, 2), "$0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015."),
        Column("tip_amount", Decimal(13, 2), "Tip amount – This field is automatically populated for credit card tips. Cash tips are not included."),
        Column("tolls_amount", Decimal(13, 2), "Total amount of all tolls paid in trip."),
        Column("total_amount", Decimal(13, 2), "The total amount charged to passengers. Does not include cash tips."),
        Column("congestion_surcharge", Decimal(13, 2), "Total amount collected in trip for NYS congestion surcharge."),
        Column("airport_fee", Decimal(13, 2), "$1.25 for pick up only at LaGuardia and John F. Kennedy Airports"),
    ])
)
def silver_yellow_trip(spark, data, year=2023, month=1, write=False):
    
    if not write:
        print("Loading table `nyc_tlc.silver_yellow_trip`")
        return spark.table("nyc_tlc.silver_yellow_trip")



    data = data.rdd.zipWithIndex().toDF()
    data = data.select(F.col("_1.*"), F.col("_2").alias('trip_id'))
    data = data.withColumn("trip_id", F.concat_ws("", F.col("metadata_year").cast(StringType()), F.col("metadata_month").cast(StringType()), F.col('trip_id').cast(StringType())).cast(LongType()))

    data = data.withColumnRenamed("VendorID", "provider_id")
    data = data.withColumn("provider_id", 
        F.when(F.col("provider_id") == 1, F.lit("Creative Mobile Technologies"))
        .otherwise("VeriFone Inc."))

    data = data.withColumnRenamed("tpep_pickup_datetime", "pickup_datetime")
    data = data.withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")
    data = data.withColumnRenamed("Passenger_count", "number_passenger")
    data = data.withColumnRenamed("Trip_distance", "trip_distance_miles")

    data = data.withColumnRenamed("PULocationID", "pickup_taxi_zone_id")
    data = data.withColumnRenamed("DOLocationID", "dropoff_taxi_zone_id")
    data = data.withColumnRenamed("RateCodeID", "rate_type")
    data = data.withColumn("rate_type", 
        F.when(F.col("rate_type") == 1, F.lit("standard rate"))
            .when(F.col("rate_type") == 2, F.lit("JFK"))
            .when(F.col("rate_type") == 3, F.lit("Newark"))
            .when(F.col("rate_type") == 4, F.lit("NewaNassau/Westchesterrk"))
            .when(F.col("rate_type") == 5, F.lit("negotiated rate"))
            .when(F.col("rate_type") == 6, F.lit("group_ride"))
            .otherwise(None))

    data = data.withColumnRenamed("Payment_type", "payment_type")
    data = data.withColumn("payment_type", 
        F.when(F.col("payment_type") == 1, F.lit("credit card"))
        .when(F.col("payment_type") == 2, F.lit("cash"))
        .when(F.col("payment_type") == 3, F.lit("no charge"))
        .when(F.col("payment_type") == 4, F.lit("dispute"))
        .when(F.col("payment_type") == 5, None)
        .when(F.col("payment_type") == 6, F.lit("voided trip"))
        .otherwise(None))

    if spark.catalog.tableExists("nyc_tlc.silver_yellow_trip"):
        spark.sql(f"DELETE FROM nyc_tlc.silver_yellow_trip s WHERE s.metadata_year = {year} AND s.metadata_month = {month}")            


    data.write.mode("append").saveAsTable("nyc_tlc.silver_yellow_trip")
    
    return data