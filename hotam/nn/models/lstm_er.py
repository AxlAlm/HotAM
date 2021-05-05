from collections import defaultdict

import torch
import torch.nn as nn

from hotam.nn.layers.seg_layers.bigram_seg import BigramSegLayer
from hotam.nn.layers.link_label_layers.dep_pairing_layer import DepPairingLayer
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.utils import util_one_hot
from hotam.nn.schedule_sample import ScheduleSampling
from hotam.nn.bio_decoder import bio_decode


class LSTM_ER(nn.Module):
    """

    https://www.aclweb.org/anthology/P16-1105.pdf

    """
    def __init__(self, hyperparamaters: dict, task_dims: dict,
                 feature_dims: dict, train_mode: bool):
        super(LSTM_ER, self).__init__()

        # number of arguemnt components
        self.num_seg_labels = task_dims["seg+label"]
        num_link_labels = task_dims["link_label"]  # number of relations

        self.train_mode = train_mode

        self.model_param = nn.Parameter(torch.empty(0))

        self.sub_graph_type = hyperparamaters["sub_graph_type"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        token_embs_size = feature_dims["word_embs"] + feature_dims["pos_embs"]
        dep_embs_size = feature_dims["deprel_embs"]
        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]
        clf_h_size = hyperparamaters["ac_seg_hidden_size"]
        link_label_clf_h_size = hyperparamaters["re_hidden_size"]
        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]
        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        tree_bidirectional = hyperparamaters["tree_bidirectional"]

        dropout = hyperparamaters["dropout"]
        self.dropout = nn.Dropout(dropout)

        self.schedule = ScheduleSampling(schedule="inverse_sig",
                                         k=hyperparamaters["k"])

        self.lstm = LSTM_LAYER(input_size=token_embs_size,
                               hidden_size=seq_lstm_h_size,
                               num_layers=seq_lstm_num_layers,
                               bidirectional=lstm_bidirectional,
                               dropout=0.0)

        num_dirs = 2 if lstm_bidirectional else 1
        clf_input_size = self.num_seg_labels + (seq_lstm_h_size * num_dirs)
        self.seg_label_clf = BigramSegLayer(
            input_size=clf_input_size,
            hidden_size=clf_h_size,
            output_size=self.num_seg_labels,
            label_emb_dim=self.num_seg_labels,
            dropout=dropout,
        )

        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        tree_input_size = seq_lstm_h_size * ns + \
            dep_embs_size + self.num_seg_labels
        link_label_input_size = tree_lstm_h_size * nt + 2 * seq_lstm_h_size * ns
        self.link_label_output_size = 2 * (num_link_labels - 1) + 1
        self.link_label_clf = DepPairingLayer(
            tree_input_size=tree_input_size,
            tree_lstm_h_size=tree_lstm_h_size,
            tree_bidirectional=tree_bidirectional,
            decoder_input_size=link_label_input_size,
            decoder_h_size=link_label_clf_h_size,
            decoder_output_size=self.link_label_output_size,
            dropout=dropout)

        # self.loss_fn = hyperparamaters["loss_fn"].lower()
        self.loss = nn.NLLLoss(reduction="mean", ignore_index=-1)
        # self.ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    @classmethod
    def name(self):
        return "LSTM_ER"

    def forward(self, batch, output):

        check = True
        token_mask = batch["token"]["mask"].type(torch.bool)
        # batch_size = batch["token"]["mask"].size(0)
        # 1)
        # pos_word_embs.shape =
        # (batch_size, max_nr_tokens, word_embs + pos_embs)
        pos_word_embs = torch.cat(
            (batch["token"]["word_embs"], batch["token"]["pos_embs"]), dim=2)
        pos_word_embs = self.dropout(pos_word_embs)

        # 2) lstm_out = (batch_size, max_nr_tokens, lstm_hidden)
        lstm_out, _ = self.lstm(pos_word_embs, batch["token"]["lengths"])

        # 3) seg+label outputs is dict containing the logits, preds, probs and
        # label one_hots
        # (batch_size, max_nr_tokens,  nr_seg_labels)
        # last dim not present in preds
        seg_label_output = self.seg_label_clf(lstm_out,
                                              batch["token"]["lengths"])

        # 5)
        # NOTE It would be better to have the bio_dict during initialization
        # instead of getting the same labels ids each step
        # TODO! NEEDS TO BE ACCESSABLE IN __INIT__
        bio_dict = defaultdict(list)
        for (i, label) in enumerate(output.label_encoders["seg+label"].labels):
            bio_dict[label[0]].append(i)

        # 4)
        if self.train_mode:
            if self.schedule.next(batch.current_epoch):
                preds_used = batch["token"]["seg+label"]
                one_hot_embs = util_one_hot(preds_used, token_mask,
                                            self.num_seg_labels)
            else:
                preds_used = seg_label_output["preds"]
                one_hot_embs = seg_label_output["one_hot"]

        bio_data = bio_decode(
            batch_encoded_bios=preds_used,
            lengths=batch["token"]["lengths"],
            apply_correction=True,
            B=bio_dict["B"],  # ids for labels counted as B
            I=bio_dict["I"],  # ids for labels counted as I
            O=bio_dict["O"],  # ids for labels counted as O
        )

        # 8)
        link_label_outputs = self.link_label_clf(
            token_embs=lstm_out,
            dep_embs=batch["token"]["deprel_embs"],
            one_hot_embs=one_hot_embs,
            roots=batch["token"]["root_idxs"],
            deplinks=batch["token"]["dephead"],
            token_mask=token_mask,
            bio_data=bio_data,
            mode="shortest_path",
            token_label={
                "preds": preds_used,
                "targets": batch["token"]["seg+label"],
                "link_label_targets": batch["token"]["link_label"],
                "link_target": batch["token"]["link"]
            },
            assertion=check)

        if self.train_mode:
            seg_label_probs = seg_label_output["probs"]
            link_label_probs = link_label_outputs["link_label_probs"]

            label_loss = self.loss(
                torch.log_softmax(seg_label_probs,
                                  dim=-1).view(-1, self.num_seg_labels),
                batch["token"]["seg+label"].view(-1))

            link_label_loss = self.loss(
                torch.log(link_label_probs).view(-1,
                                                 self.link_label_output_size),
                link_label_outputs["link_label_target"].view(-1))

            total_loss = label_loss + link_label_loss

            output.add_loss(task="total", data=total_loss)
            output.add_loss(task="link_label", data=link_label_loss)
            output.add_loss(task="seg+label", data=label_loss)

        output.add_preds(task="seg+label",
                         level="token",
                         data=seg_label_output["preds"])
        output.add_preds(task="link",
                         level="unit",
                         data=link_label_outputs["link_preds"])
        output.add_preds(task="link_label",
                         level="unit",
                         data=link_label_outputs["link_label_preds"])

        return output
