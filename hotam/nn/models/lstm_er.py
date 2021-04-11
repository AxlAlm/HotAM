from functools import reduce
from operator import iconcat

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from hotam.nn.layers.seg_layers.bigram_seg import BigramSegLayer
from hotam.nn.layers.link_label_layers.dep_pairing_layer import DepPairingLayer
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.utils import get_all_possible_pairs, range_3d_tensor_index, agg_emb
from hotam.nn.utils import util_one_hot
from hotam.nn.schedule_sample import ScheduleSampling
from hotam.nn.bio_decoder import bio_decode


class LSTM_ER(nn.Module):
    def __init__(self, hyperparamaters: dict, task_dims: dict,
                 feature_dims: dict, train_mode: bool):
        super(LSTM_ER, self).__init__()

        # number of arguemnt components
        self.num_ac = task_dims["seg+label"]
        self.num_labels = task_dims["link_label"]  # number of relations

        self.train_mode = train_mode

        self.model_param = nn.Parameter(torch.empty(0))

        self.sub_graph_type = hyperparamaters["sub_graph_type"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        token_embs_size = feature_dims["word_embs"] + feature_dims["pos_embs"]
        label_embs_size = self.num_ac
        dep_embs_size = feature_dims["deprel_embs"]

        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]

        clf_h_size = hyperparamaters["ac_seg_hidden_size"]
        clf_output_size = self.num_ac

        link_label_clf_h_size = hyperparamaters["re_hidden_size"]

        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]

        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.schedule = ScheduleSampling(
                                        schedule="inverse_sig",
                                        k=hyperparamaters["k"]
                                        )

        self.lstm = LSTM_LAYER(
                                input_size=token_embs_size,
                                hidden_size=seq_lstm_h_size,
                                num_layers=seq_lstm_num_layers,
                                bidirectional=lstm_bidirectional,
                                dropout=0.0
                               )

        num_dirs = 2 if lstm_bidirectional else 1
        clf_input_size = label_embs_size + (seq_lstm_h_size * num_dirs)
        self.seg_label_clf = BigramSegLayer(
                                            input_size=clf_input_size,
                                            hidden_size=clf_h_size,
                                            output_size=clf_output_size,
                                            label_emb_dim=label_embs_size,
                                            dropout=dropout,
                                        )

        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        link_label_input_size = tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        self.link_label_clf = DepPairingLayer(
                                                tree_input_size=tree_input_size,
                                                tree_lstm_h_size=tree_lstm_h_size,
                                                tree_bidirectional=tree_bidirectional,
                                                decoder_input_size=link_label_input_size,
                                                decoder_h_size=link_label_clf_h_size,
                                                decoder_output_size=self.num_labels,
                                                dropout=dropout
                                                )

        self.loss_fn = hyperparamaters["loss_fn"].lower()
        self.nll_loss = nn.NLLLoss(
                                    reduction="mean", 
                                    ignore_index=-1
                                    )


    @classmethod
    def name(self):
        return "LSTM_ER"

    def forward(self, batch, output):

        check = True
        tokens_mask = batch["token"]["mask"].type(torch.bool)
        batch_size = batch["token"]["mask"].size(0)

        # 1)
        # pos_word_embs.shape =
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat((
                                    batch["token"]["word_embs"], 
                                    batch["token"]["pos_embs"]
                                    ), 
                                    dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # 2) lstm_out = (batch_size, max_nr_tokens, lstm_hidden)

        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])


        # 3)
        # seg_label_logits = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_probs = (batch_size, max_nr_tokens, nr_labels)
        # seg_label_preds = (batch_size, max_nr_tokens)
        # one_hots = (batch_size, max_nr_tokens, nr_layers)
        clf_output = self.seg_label_clf(lstm_out, batch["token"]["lengths"])
        seg_label_logits, seg_label_probs = clf_output[:2]
        seg_label_preds, seg_label_embs = clf_output[-2:]

        # 4)
        if self.schedule.next(batch.current_epoch):
            preds_used = batch["token"]["seg+label"]
            embs_used = util_one_hot(preds_used, tokens_mask, self.num_ac)
        else:
            preds_used = seg_label_preds
            embs_used = seg_label_embs

        # 5)
        # NOTE It would be better to have the bio_dict during initialization
        # instead of getting the same labels ids each step
        bio_dict = defaultdict(list)
        for (i, label) in enumerate(output.label_encoders["seg+label"].labels):
            bio_dict[label[0]].append(i)
        
        bio_data = bio_decode(
                                batch_encoded_bios=preds_used,
                                lengths=batch["token"]["lengths"],
                                apply_correction=True,
                                B=bio_dict["B"],  # ids for labels counted as B
                                I=bio_dict["I"],  # ids for labels counted as I
                                O=bio_dict["O"],  # ids for labels counted as O
                                only_units = True,
                                )

        # 6)
        all_possible_pairs = get_all_possible_pairs(
                                                    starts=bio_data["unit"]["start"],
                                                    ends=bio_data["unit"]["end"],
                                                    assertion=check
                                                    )

        print(len(all_possible_pairs["start"][0]),len(all_possible_pairs["end"][0]))

        # 7) (batch_size, nr_tokens, feature_dim)
        node_embs = torch.cat((lstm_out, embs_used, batch['token']['deprel_embs']), dim=-1)

        # 8)
        link_label_data = self.link_label_clf(                                                 
                                                node_embs=node_embs,
                                                token_embs=lstm_out,
                                                dependencies=batch["token"]["dephead"],
                                                token_mask=tokens_mask,
                                                roots=output.batch["token"]["root_idxs"],
                                                pairs=all_possible_pairs,
                                                unit_span_idxs=bio_data["unit"]["span_idxs"],
                                                max_units=bio_data["max_units"],
                                                mode="shortest_path",
                                                assertion=check
                                            )

        # # 9
        # link_label_max_logits, link_label_preds, link_preds = link_label_data

        # print(link_label_max_logits)
        # print("LINK_LABEL OUTPUTS",link_label_max_logits.shape)

        # seg_label_truth = batch["token"]["seg+label"]
        # link_label_truth = batch["token"]["link_label"]
        # link_truth = batch["token"]["link"]
        # # negative link_label
        # # Wrong label prediction
        # seg_label_preds[~tokens_mask] = -1  # to avoid falses in below compare
        # label_preds_wrong = seg_label_preds != seg_label_truth
        # # wrong predictions' indices
        # idx_0, idx_1 = torch.nonzero(label_preds_wrong, as_tuple=True)
        # link_label_preds[idx_0, idx_1] = idx_1


        # # link_labels = [PRO-forward , PRO-backwards]
        # if self.train_mode:

        #         self.nll_loss(link_label_max_logits, 
        # #     label_loss = self.nll_loss(seg_label_logits.view(-1, self.num_ac), seg_label_truth.view(-1))
        # #         link_label_loss = self.nll_loss(link_label_max_logits.view(-1, self.num_labels),link_label_truth.view(-1))
        # #         # label_loss = label_loss.view(batch_size, -1).sum(1).mean()
        # #         # link_label_loss = link_label_loss.view(batch_size,
        # #         #                                        -1).sum(1).mean()

        # #     link_preds[~tokens_mask] = -1
        
        
        # total_loss = (alpha *label_loss) + ((1-alpha) *link_label_loss) + link_loss

        # #     output.add_loss(task="total", data=total_loss)
        # #     output.add_loss(task="link_label", data=link_label_loss)
        # #     output.add_loss(task="label", data=label_loss)

        # # output.add_preds(task="seg+label", level="token", data=seg_label_preds)
        # # output.add_preds(task="link", level="token", data=link_preds)
        # # output.add_preds(task="link_label",
        # #                  level="token",
        # #                  data=link_label_preds)



        return output
