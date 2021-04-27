
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

        lstm_crf_params = self.hps.get("lstm_crf", {})
        encoder_params = self.hps.get("llstm_encoder", {})
        pointer_params = self.hps.get("pointer", {})
        simple_label_l_params = self.hps.get("simpleclf-label", {})


        lstm_crf_params["input_size"] = self.feature_dims["word_embs"]
        lstm_crf_params["output_size"] = self.task_dims["seg"]
        self.seg_layer = LSTM_CRF(**lstm_crf_params)


        encoder_params["input_size"] = self.feature_dims["word_embs"] * 3
        self.encoder = LLSTMEncoder(**encoder_params)


        pointer_params["input_size"] = self.encoder.output_size
        pointer_params["hidden_size"] = self.encoder.output_size
        pointer_params["output_size"] = self.task_dims["link"]
        self.decoder = Pointer(**pointer_params)


        simple_label_l_params["input_size"] = self.encoder.output_size
        simple_label_l_params["output_size"] = self.task_dims["label"]
        simple_label_l_params["level"] = "unit"
        self.label_clf =  SimpleCLF(**simple_label_l_params)

        self.bio_decoder = BIODecoder(
                                        B = self.bio_ids["B"],
                                        I = self.bio_ids["I"],
                                        O = self.bio_ids["O"],
                                        )

    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch, output):

        print("WORD EMB", batch["token"]["word_embs"].shape)

        seg_output = self.seg_layer(
                                    batch["token"]["word_embs"],
                                    mask = batch["token"]["mask"],
                                    lengths = batch["token"]["lengths"],
                                    targets = batch["token"]["seg"] if not self.inference else None
                                    )

        bio_output = self.bio_decoder(
                                        seg_output["preds"],
                                        batch["token"]["lengths"],
                                        )
        unit_mask = create_mask(bio_output["unit"]["lengths"])

            
        unit_embs = agg_emb(
                            batch["token"]["word_embs"], 
                            lengths = bio_output["unit"]["lengths"],
                            span_indexes = bio_output["unit"]["span_idxs"], 
                            mode = "mix"
                            )


        encoder_out = self.encoder(
                                    unit_embs, 
                                    bio_output["unit"]["lengths"]
                                    )


        label_outputs =  self.label_clf(
                                        encoder_out[0],
                                        targets = batch["token"]["label"] if not self.inference else None,
                                        mask = unit_mask,
                                        unit_tok_lengths = bio_output["unit"]["lengths_tok"]

                                        )


        link_output = self.decoder(
                                    encoder_out, 
                                    mask = unit_mask, 
                                    targets = batch["token"]["link"] if not self.inference else None,
                                    unit_tok_lengths = bio_output["unit"]["lengths_tok"]
                                    )

        print("SEG LOSS", seg_output["loss"])
        print("LABEL LOSS", label_outputs["loss"])
        print("LINK LOSS", link_output["loss"])


        if not self.inference:
            total_loss = ((1-self.hps["task_weight"]) * link_output["loss"]) + ((1-self.hps["task_weight"]) * label_outputs["loss"])

            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="seg",       data=seg_output["loss"])
            output.add_loss(task="link",        data=link_output["loss"])
            output.add_loss(task="label",       data=label_outputs["loss"])

        output.add_preds(task="label",          level="unit", data=label_outputs["preds"])
        output.add_preds(task="link",           level="unit", data=link_output["preds"])


        return output

