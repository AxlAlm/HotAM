

#pytroch
import torch
import torch.nn as nn

class Encoder(nn.Module):

    """
    This is an encoder layer build from x



    """

    def __init__(   
                    self,  
                    input_size:int, 
                    hidden_size:int, 
                    num_layers:int, 
                    bidirectional:int,
                    dropout:float=0.0,
                    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.input_layer = nn.Linear(input_size, input_size)
        self.lstm =  LSTM(  
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                )
 
  


    def forward(self, X, lengths):

        X = self.dropout(X)
        X = torch.sigmoid(self.input_layer(X))
        out, hidden = self.lstm(X, lengths)

        return out, hidden

