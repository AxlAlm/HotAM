
 

import torch
import torch.nn as nn

from .token_loss import TokenCrossEntropyLoss


class SimpleCLF(nn.Module):


    def __init__(self, input_size:int, output_size:int, level:str, loss_redu:str="sum"):
        super().__init__()
        self.clf = nn.Linear(input_size, output_size)
        self._level = level
        if self._level == "unit":
            self.loss = TokenCrossEntropyLoss(reduction=loss_redu, ignore_index=-1)
        else:
            self.loss = nn.CrossEntropyLoss(reduction=loss_redu, ignore_index=-1)

    def forward(self,
                input:torch.tensor, 
                targets:torch.tensor=None,
                mask:torch.tensor=None,
                unit_tok_lengths:torch.tensor=None
                ) -> dict:

        print(input.shape)
        outputs = self.clf(input)

        loss = None
        if targets is not None:

            if self._level == "unit":
                loss = self.loss(
                                unit_inputs=outputs, 
                                unit_mask=mask,
                                unit_tok_lengths=unit_tok_lengths,
                                targets=targets,
                                )
            else:
                loss = self.loss(torch.flatten(outputs, end_dim=-2), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        probs = torch.softmax(outputs, dim=-1)

        return {
                "loss":loss,
                "preds": preds,
                "probs":probs
                }