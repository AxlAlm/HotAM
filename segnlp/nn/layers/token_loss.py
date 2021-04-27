

#pytroch
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F




class TokenCrossEntropyLoss(CrossEntropyLoss):


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, 
                unit_inputs: Tensor, 
                unit_mask: Tensor,
                unit_tok_lengths: Tensor,
                targets: Tensor,
                ) -> Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)

        
        print("OKOK LETT", torch.flatten(unit_inputs, end_dim=-2).shape)

        print(torch.sum(unit_mask.view(-1)))

        flat_unit_inputs = torch.flatten(unit_inputs, end_dim=-2)[unit_mask.view(-1)]

        print("F SHAPE", flat_unit_inputs.shape)

        flat_unit_tok_lengths = torch.tensor([tl for unit in unit_tok_lengths for tl in unit])
        input = torch.repeat_interleave(flat_unit_inputs, flat_unit_tok_lengths, dim=-2)

        print(input.shape)

        targets = targets.view(-1)
        print(targets.shape, self.ignore_index)
        targets = targets[targets != self.ignore_index]

        print(input.shape, targets.shape, sum(flat_unit_tok_lengths))

        return F.cross_entropy(
                                input, 
                                targets, 
                                weight=self.weight,
                                ignore_index=self.ignore_index, 
                                reduction=self.reduction
                                )
