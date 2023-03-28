from flypipe import node
from nyc_tlc.pipeline.model.predict.batch_inverse_scaler import batch_inverse_scaler
from nyc_tlc.pipeline.model.predict.online_inverse_scaler import online_inverse_scaler
from nyc_tlc.pipeline.model.train.metrics import metrics


@node(
    type="pandas",
    dependencies=[batch_inverse_scaler, online_inverse_scaler, metrics]
)
def graph(batch_inverse_scaler, online_inverse_scaler, metrics):
    raise NotImplementedError('this is a dummy node, shall not be run')

