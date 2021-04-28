

#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp.nn.layers.rep_layers import LSTM

class LLSTMEncoder(nn.Module):

    """
    This is an encoder layer build from x

    """

    def __init__(   
                    self,  
                    input_size:int, 
                    hidden_size:int=256, 
                    num_layers:int=1, 
                    bidir:bool=True,
                    dropout:float=0.0,
                    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.input_layer = nn.Linear(input_size, input_size)
        self.lstm =  LSTM(  
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidir=bidir,
                                )
        self.output_size = hidden_size * (2 if bidir else 1)


    def forward(self, input:Tensor, batch:dict):

        X = self.dropout(input)
        X = torch.sigmoid(self.input_layer(X))
        out, hidden = self.lstm(X, batch["token"]["lengths"])

        return out, hidden

