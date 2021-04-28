
 

import torch
import torch.nn as nn
from torch import Tensor

from .token_loss import TokenCrossEntropyLoss



class SimpleCLF(nn.Module):


    def __init__(self, 
                input_size:int,
                output_size:int, 
                loss_redu:str="sum"
                ):
        super().__init__()
        self.output_size = output_size
        self.clf = nn.Linear(input_size, output_size)
        self._level = level
        self.loss = TokenCrossEntropyLoss(reduction=loss_redu, ignore_index=-1)
    
    
    def forward(self,
                input,
                seg_data,
                batch,
                ) -> dict:

        logits = self.clf(input)
        preds = torch.argmax(outputs, dim=-1)
        probs = torch.softmax(outputs, dim=-1)

        return {
                "logits": logits,
                "preds": preds,
                "probs": probs
                }