import os
import pickle
import mlflow
from mlflow.models import infer_signature
from flypipe import node
from flypipe.schema import Schema, Column
from flypipe.schema.types import Decimal
from nyc_tlc.pipeline.model.train.scale import scale
from sklearn.ensemble import RandomForestRegressor

@node(
    type="pandas",
    description="Scale trip_amount",
    tags=["model", "train"],
    group="model.train",
    dependencies=[
        scale.select("trip_id", "data_type", "pickup_hour", "pickup_taxi_zone_id", "payment_cash", 
        "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown", "tip_amount").alias("data")
    ],
    output=Schema(
        scale.output.get("trip_id"),
        scale.output.get("data_type"),
        scale.output.get("tip_amount"),
        Column("tip_amount_predicted", Decimal(13,2), "tip amount predicted"),
    )
        
)
def train(data):
    clf = RandomForestRegressor(random_state=12345)
    data_train = data[data['data_type']=="train"]
    x_cols = ["pickup_hour", "pickup_taxi_zone_id", "payment_cash", "payment_no_charge", "payment_credit_card", "payment_dispute", "payment_unknown"]
    X_train = data_train[x_cols]
    y_train = data_train['tip_amount']
    clf = clf.fit(X_train, y_train)

    if mlflow.active_run():
        artifact_path = mlflow.active_run().info.artifact_uri
        artifact_path = artifact_path.replace("dbfs:", "/dbfs")
        artifact_path = os.path.join(artifact_path, 'random_forest', 'artifacts')
        os.makedirs(artifact_path, exist_ok=True)
        pickle.dump(clf, open(os.path.join(artifact_path, 'rf_model.pkl'), 'wb'))

        signature = infer_signature(X_train, y_train)

        mlflow.sklearn.log_model(
            clf,
            artifact_path="random_forest",
            signature=signature,
            input_example=X_train.head(5)
        )

    data['tip_amount_predicted'] = clf.predict(data[x_cols])
    return data