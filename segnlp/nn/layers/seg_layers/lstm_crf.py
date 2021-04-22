#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#segnlp
from segnlp.nn.layers.rep_layers import LSTM
from segnlp.utils import zero_pad

# use a torch implementation of CRF
from torchcrf import CRF

from segnlp.utils import timer

class LSTM_CRF(nn.Module):

    """
    https://www.aclweb.org/anthology/W19-4501

    """

    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, bidir:bool, output_dim:int, dropout:float=0.0):
        super().__init__()
        self.inference = inference
        self.dropout = nn.Dropout(dropout)
        self.lstm = LSTM(  
                            input_size = input_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            bidirectional = bidir,
                            )

        #for task, output_dim in task_dims.items():
        self.clf = nn.Linear(hidden_dim*(2 if bidir else 1),output_dim)
        self.crf = CRF(    
                        num_tags=output_dim,
                        batch_first=True
                        )
    

    @classmethod
    def name(self):
        return "LSTM_CRF"


    def forward(self, input_embs, mask, length, targets=None):

        lengths = batch["token"]["lengths"]
        mask = batch["token"]["mask"]
        word_embs = batch["token"]["word_embs"]

        input_embs = self.dropout(input_embs)
        lstm_out, _ = self.lstm(input_embs, lengths)
        out = self.clf(lstm_out)

        if targets:
            loss = -crf(    
                        emissions=out, #score for each tag, (batch_size, seq_length, num_tags) as we have batch first
                        tags=targets,
                        mask=mask,
                        reduction='mean'
                        )

        #returns preds with no padding (padding values removed)
        preds = crf.decode( 
                            emissions=dense_out, 
                            mask=mask
                            )

        return {
                    "loss": loss, 
                    "preds": preds,
                }