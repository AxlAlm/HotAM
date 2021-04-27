

#basics
from typing import Tuple

#pytroch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(   
                    self,
                    input_size:int,
                    hidden_size:int=256, 
                    num_layers:int=1, 
                    bidir:bool=True, 
                    dropout:float=0.0,
                    ):
        super().__init__()

        self.lstm = nn.LSTM(     
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers, 
                                bidirectional=bidir,  
                                batch_first=True
                            )

        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size* (2 if bidir else 1)


    def forward(self, 
                input:torch.tensor, 
                lengths:torch.tensor, 
                padding=0.0) -> Tuple[torch.tensor, Tuple[torch.tensor,torch.tensor]]:

        input = self.dropout(input)
        
        pass_states = False
        if isinstance(input, tuple):
           #X, h_0, c_0 = X
            input, *states = input
            pass_states = True

        packed_embs = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)

        if pass_states:
            lstm_packed, hidden = self.lstm(packed_embs, states)
        else:
            lstm_packed, hidden = self.lstm(packed_embs)

        unpacked, lengths = pad_packed_sequence(lstm_packed, batch_first=True, padding_value=0.0)

        return unpacked, hidden