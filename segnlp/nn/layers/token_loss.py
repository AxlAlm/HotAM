

#pytroch
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


#segnlp
from segnlp.nn.utils import scatter_repeat



class TokenCrossEntropyLoss(CrossEntropyLoss):


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def __token_scatter(self, unit_logits:Tensor, span_tok_lengths:list, none_span_mask:list, max_nr_token:int):
        """
        scatters the logits for units over all tokens which the unit comprise of. 
 
        """
        
        device = unit_logits.device
        batch_size = unit_logits.shape[0]
        nr_labels = unit_logits.shape[-1]

        # always show high prob for class at index 0 which should allways be None, or other etc
        default_value = torch.FloatTensor([0.999] + ([0.0001/(nr_labels-1)]*(nr_labels-1)), device=device)
     
        token_logits = torch.zeros((batch_size, max_nr_token, nr_labels), device=device, )
        for i in range(batch_size):
       
            lengths_tok = torch.LongTensor(span_tok_lengths[i], device=device)
            mask = torch.BoolTensor(none_span_mask[i], device=device)
            length = torch.sum(lengths_tok)
            nr_units = torch.sum(mask)
     
            src = torch.zeros((mask.shape[0], nr_labels), device=device)
            src[:] = default_value

            token_logits[i][:length] = scatter_repeat(
                                                        src = src,
                                                        values = unit_logits[i][:nr_units], 
                                                        lengths = lengths_tok, 
                                                        length_mask = mask, 
                                                        )
        return token_logits
        

    def forward(self, 
                unit_logits: Tensor, 
                span_tok_lengths: Tensor,
                none_span_mask: list,
                max_nr_token: int,
                targets: Tensor,
                ) -> Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)

        token_logits = self.__token_scatter(
                                            unit_logits = unit_logits, 
                                            span_tok_lengths = span_tok_lengths,
                                            none_span_mask = none_span_mask,
                                            max_nr_token = max_nr_token,
                                            )

        return F.cross_entropy(
                                torch.flatten(token_logits, end_dim=-2), 
                                targets.view(-1), 
                                weight=self.weight,
                                ignore_index=self.ignore_index, 
                                reduction=self.reduction
                                )
