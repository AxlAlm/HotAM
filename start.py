
import segnlp
from segnlp import Pipeline
from segnlp.datasets.am import PE
from segnlp.nn.models.am import JointPN
from segnlp.features import GloveEmbeddings, FlairEmbeddings, BertEmbeddings
from segnlp.nn.default_hyperparamaters import get_default_hps

from segnlp.utils import random_ints, set_random_seed
import numpy as np

from pprint import pprint


exp = Pipeline(
                project="debugging",
                dataset=PE( 
                            tasks=["seg","label","link"],
                            prediction_level="token",
                            sample_level="paragraph",
                            ),
                features =[
                            GloveEmbeddings(),
                            #FlairEmbeddings(),
                            #BertEmbeddings(),
                            ],
                model = JointPN
            )

exp.dataset.info

segnlp.settings["dl_n_workers"] = 0
hps = get_default_hps(JointPN.name())
hps["max_epochs"] = 2
#hps["lr"] = [0.01,0.001]

best_hp = exp.hp_tune(
                        hyperparamaters = hps,
                        n_random_seeds=4,
                        ptl_trn_args=dict(
                                            overfit_batches=0.1,
                                            #gpus=[0]
                                            )
                        )

#exp.test()




# all_splits = []
# for i, splits in exp.dataset.splits.items():
#     for s, ids in splits.items():
#         all_splits.extend(ids.tolist())

# b = np.sort(np.array(all_splits))
# print(len(b), max(b))

# a = np.arange(max(b))
# n = ~np.isin(a,b)
# print(a[n])

# 6949 6982
# [ 624  644 1124 1269 1529 1689 1917 1968 2027 2292 2659 2976 3050 3096
#  3150 3169 3220 3622 3982 4050 4288 4437 4461 4734 4824 5243 6019 6072
#  6221 6247 6533 6696 6738 6867]

# maxs, nr_ids_s = zip(*[(max(s),len(s)) for _,s in list(exp.dataset.splits.items())[0][1].items()])
# nr_ids = sum(nr_ids_s)
# print(nr_ids)
# m = max(maxs)
# print(m)

# print(b[b > nr_ids])

