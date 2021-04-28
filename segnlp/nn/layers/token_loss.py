

#pytroch
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F




class TokenCrossEntropyLoss(CrossEntropyLoss):


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def __token_scatter(self, unit_logits:Tensor, lengths:list, length_mask:list, max_nr_token:int):
        
        batch_size = unit_logits.shape[0]
        token_logits = np.zeros((batch_size, max_nr_token))
        for i in range(batch_size):
            token_logits[i] = scatter_repeat(
                                            src = token_logits[i],
                                            values = unit_logits[i], 
                                            lengths = lengths[i], 
                                            length_mask = length_mask, 
                                            )
        return token_logits
        

    def forward(self, 
                unit_logits: Tensor, 
                lengths: Tensor,
                length_mask: list,
                max_nr_token: int,
                targets: Tensor,
                ) -> Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)

        token_logits = self.__token_scatter(
                                            unit_logits = unit_logits, 
                                            lengths  = lengths,
                                            length_mask = length_mask,
                                            max_nr_token = max_nr_token,
                                            )
                                            
        assert token_logits.shape[0] == targets.view(-1).shape[0]

        return F.cross_entropy(
                                token_logits, 
                                targets.view(-1), 
                                weight=self.weight,
                                ignore_index=self.ignore_index, 
                                reduction=self.reduction
                                )
