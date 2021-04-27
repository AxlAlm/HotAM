

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

#segnlp
from segnlp.nn.layers.attention import CBAttentionLayer


from ..token_loss import TokenCrossEntropyLoss

class Pointer(nn.Module):

    """
    A pointer is learing attention scores for each position over all possible position. Attention scores
    are probility distributions over all possible units in the input.

    These probabilites are interpreted as to where a units it pointing. E.g. if the attention
    scores for a unit at position n is are hightest at position n+2, then we say that
    n points to n+2.


    The works as follows:

    1) we set first decode input cell state and hidden state from encoder outputs

    then for each LSTMCELL timestep we:

    2) apply a ffnn with sigmoid activation

    3) apply dropout

    4) pass output of 3 to lstm cell with reps from prev timestep

    5) apply attention over given decoder at timestep i and all decoder timestep reps


    """

    def __init__(self, 
                input_size:int, 
                output_size:int, 
                hidden_size:int=256, 
                dropout:float=0.0, 
                ):
        super().__init__()

        self._hidden_size = hidden_size
        self.input_layer = nn.Linear(input_size, input_size)
        self.lstm_cell =  nn.LSTMCell(input_size, hidden_size)
        self.attention = CBAttentionLayer(
                                        input_dim=hidden_size,
                                        )
    
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, 
                encoder_outputs:torch.tensor, 
                mask:torch.tensor,
                ):

        if isinstance(encoder_outputs, tuple):

            encoder_outputs, (encoder_h_s, encoder_c_s) = encoder_outputs
            device = encoder_outputs.device

            seq_len = encoder_outputs.shape[1]
            batch_size = encoder_outputs.shape[0]

            # We get the last hidden cell states and timesteps and concatenate them for each directions
            # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) -> (BATCH_SIZE, HIDDEN_SIZE*NR_DIRECTIONS)
            # The cell state and last hidden state is used to start the decoder (first states and hidden of the decoder)
            # -2 will pick the last layer forward and -1 will pick the last layer backwards
            h_s = torch.cat((encoder_h_s[-2],encoder_h_s[-1]),dim=1)
            c_s = torch.cat((encoder_h_s[-2],encoder_h_s[-1]),dim=1)
            
        else:
            seq_len = encoder_outputs.shape[1]
            batch_size = encoder_outputs.shape[0]
            device = encoder_outputs.device

            h_s = torch.rand((batch_size, self._hidden_size), device=device)
            c_s = torch.rand((batch_size, self._hidden_size), device=device)
     

        decoder_input = torch.zeros(h_s.shape, device=device)
        logits = torch.zeros(batch_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            
            decoder_input = torch.sigmoid(self.input_layer(decoder_input))
            decoder_input = self.dropout(decoder_input)

            h_s, c_s = self.lstm_cell(decoder_input, (h_s, c_s))

            logits[:, i] = self.attention(h_s, encoder_outputs, mask, return_softmax=False)
        
        preds = torch.argmax(logits, dim=-1)

        return {
                "logits": logits,
                "preds":preds,
                "level": unit
                }

