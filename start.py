from hotam.datasets import PE
from hotam import Pipeline
from hotam.features import DummyFeature, OneHots
from hotam.nn.models import LSTM_ER
from hotam.nn.default_hyperparamaters import get_default_hps

exp = Pipeline(project="debugging",
               dataset=PE(
                   tasks=["seg+label", "link", "link_label"],
                   prediction_level="token",
                   sample_level="document",
               ),
               encodings=["pos", "deprel", "dephead"],
               features=[DummyFeature(),
                         OneHots("pos"),
                         OneHots("deprel")])

exp.fit(
    model=LSTM_ER,
    hyperparamaters=get_default_hps(LSTM_ER.name()),
    # exp_logger=exp_logger,
    ptl_trn_args=dict(
                    overfit_batches=0.1,
                    ))
