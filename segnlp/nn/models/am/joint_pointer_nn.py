
#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.nn.layers.rep_layers import LLSTMEncoder
from segnlp.nn.layers.link_layers import Pointer
from segnlp.nn.layers.clf_layer import SimpleCLF
from segnlp.nn.layers.seg_layers import LSTM_CRF
from segnlp.nn.utils import BIODecoder
from segnlp.nn.utils import agg_emb
from segnlp.nn.utils import create_mask
from segnlp.ptl import PTLBase

from segnlp.nn.layers.base import SegLayer
from segnlp.nn.layers.base import UnitLayer
from segnlp.nn.layers.base import RepLayer


class JointPN(PTLBase):

    """
    Inspiration from paper:
    https://arxiv.org/pdf/1612.08994.pdf

    more on Pointer Networks:
    https://arxiv.org/pdf/1409.0473.pdf
    https://papers.nips.cc/paper/5866-pointer-networks.pdf  

    A quick read:
    https://medium.com/@sharaf/a-paper-a-day-11-pointer-networks-59f7af1a611c

    """
    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        print(kwargs)

        self.seg = SegLayer(
                            task = "seg",
                            layer = LSTM_CRF, 
                            hyperparams =  self.hps.get("lstm_crf", {}), 
                            labels = self.task_labels["seg"],
                            encoding_scheme = self.encoding_scheme,
                            input_size =  self.feature_dims["word_embs"],
                            output_size = self.task_dims["seg"],
                            )
        #self.seg.inference = self.inference

        self.encoder = RepLayer(
                                layer = LLSTMEncoder, 
                                hyperparams = self.hps.get("llstm_encoder", {}),
                                input_size = self.feature_dims["word_embs"] * 3
                                )
        #self.encoder.inference = self.inference


        self.decoder = UnitLayer(
                                task = "link",
                                layer = Pointer,
                                hyperparams = self.hps.get("pointer", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims["link"]
                                )
        #self.decoder.inference = self.inference

        self.label_clf =  UnitLayer(
                                    task = "label",
                                    layer = SimpleCLF,
                                    hyperparams = self.hps.get("simpleclf-label", {}),
                                    input_size = self.encoder.output_size,
                                    output_size = self.task_dims["label"]
                                    )
        #self.label_clf.inference = self.inference


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch, output):


        seg_output = self.seg(
                            input=batch["token"]["word_embs"],
                            batch=batch
                            )
    
        unit_embs = agg_emb(
                            batch["token"]["word_embs"], 
                            lengths = seg_output["unit"]["lengths"],
                            span_indexes = seg_output["unit"]["span_idxs"], 
                            mode = "mix"
                            )

        encoder_out = self.encoder(
                                    input = unit_embs,
                                    batch = batch,
                                    )
    
        label_outputs =  self.label_clf(
                                        input = encoder_out[0],
                                        seg_data = seg_output,
                                        batch = batch
                                        )

        link_output = self.decoder(
                                    input = encoder_out,
                                    seg_data = seg_output,
                                    batch = batch
                                    )

        if not self.inference:
            total_loss = ((1-self.hps["task_weight"]) * link_output["loss"]) + ((1-self.hps["task_weight"]) * label_outputs["loss"])

            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="seg",       data=seg_output["loss"])
            output.add_loss(task="link",        data=link_output["loss"])
            output.add_loss(task="label",       data=label_outputs["loss"])

        output.add_preds(task="label",          level="unit", data=label_outputs["preds"])
        output.add_preds(task="link",           level="unit", data=link_output["preds"])


        return output

