
 

import torch
import torch.nn as nn



class SimpleCLF(nn.Module):


    def __init__(self, input_size:int, output_size:int, loss_redu:str="sum"):
        self.clf = nn.Linear(input_size, output_size)
        self.loss = nn.CrossEntropyLoss(reduction=loss_redu, ignore_index=-1)

    
    def forward(self, input_embs, targets:torch.tensor=None):
        outputs = self.clf(input_embs)

        loss = None
        if targets is not None:
            loss = self.loss(torch.flatten(output, end_dim=-2), targets.view(-1))
        
        preds = torch.argmax(outputs, dim=-1)
        probs = torch.softmax(outputs, dim=-1)

        return {
                "loss":loss,
                "preds": preds,
                "probs":probs
                }