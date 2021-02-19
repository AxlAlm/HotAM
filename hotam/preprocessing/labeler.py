
#basics
import re
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


#hotam
from hotam.utils import RangeDict
from hotam.utils import timer
from hotam import get_logger

logger = get_logger("LABELER")


class Labeler:


    def _label_spans(self, sample:pd.DataFrame, span_labels:dict):

        def label_f(row, span_labels):
            return span_labels.get(int(row["char_end"]),{})

        sample = pd.concat([sample,sample.apply(label_f, axis=1, result_type="expand", args=(span_labels,))], axis=1)
        return sample


    def _label_tokens(self):
        pass


    def _label_bios(self, sample):
        sample["BIO"] = "O"
        acs = sample.groupby("ac_id")
        for g, ac_df in acs:
            sample.loc[ac_df.index,"BIO"] = ["B"] +  (["I"] * (ac_df.shape[0]-1))
        return sample


    def _label_ams(self, sample):
        sample = self.__ams_as_pre(sample)
        return sample


    def __ams_as_pre(self,sample):
        sample["am_id"] = np.nan
        groups = sample.groupby("sentence_id")

        for sent_id, sent_df in groups:
            
            acs = sent_df.groupby("ac_id")
            prev_ac_end = 0
            for ac_id, ac_df in acs:
                
                ac_start = min(ac_df["char_start"])
                ac_end = max(ac_df["char_end"])

                # more than previous end of ac and less than ac start
                cond1 = sent_df["char_start"] >= prev_ac_end 
                cond2 = sent_df["char_start"] < ac_start
                idxs = sent_df[cond1 & cond2].index
                # self.level_dfs["token"]["am_id"].iloc[idxs] = ac_id
                sample["am_id"].iloc[idxs] = ac_id
                prev_ac_end = ac_end