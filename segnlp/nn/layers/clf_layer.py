
 

import torch
import torch.nn as nn
from torch import Tensor

from .token_loss import TokenCrossEntropyLoss



class SimpleCLF(nn.Module):


    def __init__(self, 
                input_size:int,
                output_size:int, 
                ):
        super().__init__()
        self.output_size = output_size
        self.clf = nn.Linear(input_size, output_size)
    
    
    def forward(self,
                input,
                seg_data,
                batch,
                ) -> dict:

        logits = self.clf(input)
        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        return {
                "logits": logits,
                "preds": preds,
                "probs": probs
                }