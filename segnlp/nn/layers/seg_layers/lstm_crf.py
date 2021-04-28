#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


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

    def __init__(self, 
                input_size:int, 
                output_size:int, 
                hidden_size:int=256, 
                num_layers:int=1, 
                bidir:bool=True, 
                dropout:float=0.0,
                ):
        super().__init__()
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = LSTM(  
                            input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            bidir = bidir,
                            )

        self.clf = nn.Linear(hidden_size*(2 if bidir else 1),output_size)
        self.crf = CRF(    
                        num_tags=output_size,
                        batch_first=True
                        )

    @classmethod
    def name(self):
        return "LSTM_CRF"


    def forward(self, input:Tensor, batch:dict):

        lengths = batch["token"]["lengths"]
        mask = batch["token"]["mask"]

        input_embs = self.dropout(input)
        lstm_out, _ = self.lstm(input_embs, lengths)
        out = self.clf(lstm_out)

        loss = None
        if not self.inference:
            targets = batch["token"][self.task]
            targets[targets == -1] = 0

            loss = -self.crf(    
                            emissions=out, #score for each tag, (batch_size, seq_length, num_tags) as we have batch first
                            tags=targets,
                            mask=mask,
                            reduction=self.loss_redu
                            )

        #returns preds with no padding (padding values removed)
        preds = self.crf.decode( 
                                emissions=out, 
                                mask=mask
                                )
        preds = torch.tensor(zero_pad(preds))

        return {
                    "loss": loss, 
                    "preds": preds,
                }