
#basics
import inspect



#pytorch
import torch.nn as nn
from torch import Tensor
import torch

#segnlp
from segnlp.nn.utils import BIODecoder
from segnlp.nn.layers.token_loss import TokenCrossEntropyLoss


class Layer(nn.Module):

    def __init__(self, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int, 
                output_size:int=None,
                ):
        super().__init__()
        self.inference = False

        input_size = input_size
        output_size = output_size

        params = hyperparams
        params["input_size"] = input_size
       
        if output_size is not None:
            params["output_size"] = output_size

        if isinstance(layer, str):
            pass
        else:
            self.layer = layer(**params)

        self.layer.inference = self.inference

        self.output_size = self.layer.output_size
    

    def forward(self, *args, **kwargs):
        output = self._call(*args, **kwargs)

        assert output is not None

        if "loss" in output:
            if torch.isnan(output["loss"]):
                raise RuntimeError("LAYER create NAN loss")

        return  output     


class RepLayer(Layer):

    def __init__(self, layer:nn.Module, hyperparams:dict, input_size:int):
        super().__init__(layer=layer, hyperparams=hyperparams, input_size=input_size)
        

    def _call(self, input:Tensor, batch:dict):
        return self.layer(
                        input=input,
                        batch=batch
                        )


class SegLayer(Layer):

    def __init__(self, 
                task:str, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                labels:dict,
                encoding_scheme:str="bio",
                loss_redu:str="mean", 
                ignore_index:int=-1
                ):
        super().__init__(
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )
        self.task = task
        self.loss = nn.CrossEntropyLoss(reduction=loss_redu, ignore_index=ignore_index)     

        self.layer.task = task
        self.layer.loss_redu = loss_redu

        if encoding_scheme == "bio":
            self.seg_decoder = BIODecoder(
                                        B = [i for i,l in enumerate(labels) if "B-" in l],
                                        I = [i for i,l in enumerate(labels) if "I-" in l],
                                        O = [i for i,l in enumerate(labels) if "O-" in l],
                                        )
        else:
            raise NotImplementedError(f'"{encoding_scheme}" is not a supported encoding scheme')


    def _call(self, input:Tensor, batch:dict):
        seg_output = self.layer(
                                input=input,
                                batch=batch,
                                )

        if "loss" not in seg_output:
            seg_output["loss"] = self.loss(torch.flatten(seg_output["logits"], end_dim=-2), batch["token"][self.task].view(-1))

        seg_output.update(self.seg_decoder(
                                            batch_encoded_bios = seg_output["preds"], 
                                            lengths = batch["token"]["lengths"], 
                                                            
                                            ))
        return seg_output



class UnitLayer(Layer):

    def __init__(self, 
                task:str, 
                layer:nn.Module, 
                hyperparams:dict, 
                input_size:int,
                output_size:int,
                loss_redu:str="mean", 
                ignore_index:int=-1
                ):
        super().__init__(              
                        layer=layer, 
                        hyperparams=hyperparams,
                        input_size=input_size,
                        output_size=output_size
                        )

        self.task = task
        self.loss = TokenCrossEntropyLoss(reduction=loss_redu, ignore_index=ignore_index)  


    def _call(self, input:Tensor, seg_data:dict, batch:dict):
        unit_output = self.layer(
                                input=input, 
                                seg_data=seg_data, 
                                batch=batch
                                )

        if "loss" not in unit_output:
            unit_output["loss"] = self.loss(
                                            unit_logits = unit_output["logits"], 
                                            span_tok_lengths = seg_data["span"]["lengths_tok"],
                                            none_span_mask = seg_data["span"]["none_span_mask"],
                                            max_nr_token = torch.max(batch["token"]["lengths"]),
                                            targets = batch["token"][self.task],
                                            )

        return unit_output
