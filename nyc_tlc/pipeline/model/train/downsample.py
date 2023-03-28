from flypipe import node
from nyc_tlc.pipeline.model.data import data

@node(
    type="pandas_on_spark",
    description="Downsample the data to train using 50k training points",
    dependencies=[
        data
    ],
    tags=["model", "train"],
    group="model.train",
    output=data.output
)
def downsample(data):
    return data.sample(frac=1., random_state=12345).head(50000)