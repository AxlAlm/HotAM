"""
A. Reference:
--------------

    This code is based on DGL's tree-LSTM implementation found in the paper [3]
    DGL
    Implementation can be found at
    https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py

B. Papers list:
---------------

    1. Neural End-to-End Learning for Computational Argumentation Mining.
    https://arxiv.org/abs/1704.06104v2

    2. End-to-End Relation Extraction using LSTMs on Sequences and Tree
    Structures
    https://arxiv.org/abs/1601.00770

    3. Improved Semantic Representations From Tree-Structured Long Short-Term
    Memory Networks.
    https://arxiv.org/abs/1503.00075

    4. A Shortest Path Dependency Kernel for Relation Extraction
    https://dl.acm.org/doi/10.3115/1220575.1220666

C. General Description:
-----------------------

    This code implements the LSTM-ER model in [1], based on [2], to classify
    relations between argument components in a document. The LSTM-ER is derived
    from the N-ary treeLSTM architecture found in [3].  The relation extraction
    (RE) module in [2] utilizes the treeLSTM to process a sentence over its
    dependency tree. The dependency tree in nature has a varying number of
    children. However, N-ary design needs a fixed number of child nodes. For
    example, in [3], the N-ary tree is used with constituency binary-tree where
    each node has a left and a right child node.

    In [2], the dependency tree nodes are categorized into two classes: the
    shortest-path nodes are one class, and the other nodes are the second class.
    Nodes that belong to the same class share the same weights. As indicated in
    [4], the shortest path between two entities in the same sentence contains
    all the  information required to identify those relationships. However, as
    stated in [1], 92% of argument components' relationships are across
    different sentences. We cannot identify the path that encodes all the
    required information to identify the relationships between two arguments
    components across different sentences.

D. Abbreviation:
---------------------
    B      : Batch size
    H      : LSTM's Hidden size
    NE-E   : Entity label embedding size
    SEQ    : Max Sequence length
    W-E    : Word embedding size
    NE-OUT : Output dimension for NER module
    Nt     : total number of nodes (Nt) in the batch
    Nchn   : Number of children nodes in the batch

E. TreeLSTMCell Impelementation:
--------------------------------
    For each parent node, `message_func` sends the children's information to
    `reduce_func` function . The following info is sent:
    h:      child nodes' hiddens state      Size: (Nt, Nchn, H)
    c:      child nodes' cell state         Size: (Nt, Nchn, H)
    type_n: child nodes' type               Size: (Nt, Nchn)

    The data is retained in `nodes.mailbox`. The return of `reduce_func`
    function is then sent to the next function, `apply_node_func`.

    We receive h and c in a tensor of size (Nt, Nchn, H). Because the
    number of children in the batch may vary, the `reduce_function`
    collects/groups the information according to the `Nchn` dim. It
    calls itself iteratively to process each group separately.  The
    function then stacks the results vetically and sends them. Thus,
    the dimensions other than Dimension(0) (i.e Nt) must be equal to each
    other. Also, the final number of rows, Dimension(0), after stacking must be
    equal to the number of nodes (batch size); i.e. Nt = number of parent nodes.

    For the leaf nodes, where there is no childen, the code starts at
    `apply_node_func`, The hidden state is initialized, then the the gates
    values are calculated

    E1. The forget gate eqn:
    -----------------------
        Assuming the following:
        1. For nodes in a graph [Ng], the number of nodes = n
        2. For node-t ∈ Ng & 1<=t<=N:
            a. Child nodes of node-t is [Nct]

            b. number of children of node-t: Nchn(t) = ℓ,
            For an arbitry node (node-r), r ≠ t and r ∈ [Ng]: Nchn(r) may not
            be equal to ℓ

            c. the hidden states for the child nodes htl = [hi] where
            1 <= i <= ℓ.
            Each child node is either of type_n0 or type_n1;
            the hidden state for typn_0 is h_t0 and for type_n1 is
            h_t1, where
            h_t0 = Sum([h0])= Sum( [hi | 1 <= j <= ℓ & m(j)=type_n0] )
            h_t1 = Sum([h1])= Sum( [hi | 1 <= j <= ℓ & m(j)=type_n1] )

            e. Node-t have ℓ forget gates; a gate for each child node

        In [1] eqn 4, the second part of the forget gate (Sum(U*h)) could
        be written as follows:
            - For each node-k in the child nodes: The forget gate
            (ftk, 1 <= k <= ℓ) is
            either a type_0 (f0) or (f1).  where:
            f0 = U00 h_t0 + U01 h_t1,  eq(a)
            f1 = U10 h_t0 + U11 h_t1   eq(b)

    E2. i,o,u eqn:
    --------------
        For node_t:
        i_t = U_i0 . h_t0 + U_i1 . h_t1   eq(c)
        o_t = U_o0 . h_t0 + U_o1 . h_t1   eq(d)
        u_t = U_u0 . h_t0 + U_u1 . h_t1   eq(e)

    E3. Example:
    -------------
        - Assuming a node-t = node-1 in a graph:
        - node-1 have 4 child nodes: Nct=[n1, n2, n3, n4].
        - The types of child nodes are as follows [0, 1, 1, 0]
        - Ignoring the fixed parts in the forget gates' equation: Wx & b:
            * the forget gate for each child node will be as follows:
                For node-k that is child of node-t:
                ftk = Um(tk)m(tl) * htl,
                where: tl ∈ Nct, 1 <= tl < 4 & m(lt)=is either 0 or 1
        - For each child node, the equations are:
            child-node-1: f11 = U00 h11 + U01 h12 + U01 h13 + U00 h14
            child-node-2: f12 = U10 h11 + U11 h12 + U11 h13 + U10 h14
            child-node-3: f13 = U10 h11 + U11 h12 + U11 h13 + U10 h14
            child-node-4: f14 = U00 h11 + U01 h12 + U01 h13 + U00 h14
            child-node-5: f15 = U10 h11 + U11 h12 + U11 h13 + U10 h14

        - The equation of child-node 1,4 (type_n0) are equal to each
            other, the same are for child nodes 2,3, (type_n1).

        - Further reduction can be done as follows:
            forget type_0: f0 = U00 (h11 + h14) + U01 (h12 + h13)
            forget type_1: f1 = U10 (h11 + h14) + U11 (h12 + h13)
            h_t0 = (h11 + h14)
            h_t1 = (h12 + h13), see section E1.c above.

            f0 = U00 h_t0 + U01 h_t1
            f1 = U10 h_t0 + U11 h_t1
            where ht_0 is hidden states for type_n0 child nodes and ht_1 is
            hidden states for type_n1 child nodes.

    E4. Impelemntation:
    --------------------
        Step:1 Get ht_0 anf ht_1:
        *************************
            1. Get hidden states for each node type: ht_0, ht_1
                a. Get nodes that are belong to each node type
                    (type: 0 & 1)
                b. Get h and c for each node type "ht_0, ht_1"
                c. If there is no specific node type,
                    the respective ht_0 or ht_1 is zeros

        Step:2 i,o,t gates: based on eqs(c,d,e) Under section D:
        **************************************************
            a. [ht_0, ht_1] [   Uiot   ] = [i, o, t]
                (Nt , 2H)   (2H , 3H)   = (Nt , 3H)

            b. `reduce_func` return [i, o, t]

        Step:3 Forget gate: based on eqs(a,b) Under section C:
        ************************************************
            a. [ht_0, ht_1] [    Uf    ] =  [f0, f1]
                (Nt , 2H)     (2H , 2H)  =  (Nt , 2H)

            b. Then, construct a tensor f_cell (Nt, Nchn, H) ,
                where each tensor at (Nt, Nchn) is either
                f_0 or f_1 according to the type of the respective
                child node. for the example in section C the matrix
                f_cell (1, 4, H) = [f0; f1; f1; f0]

            c. f_tk = sigma( W X_emb + f_cell + b)
                The size of f_tk, [W X_emb] and f_cell = (Nt, Nchn, H)
                The size of b is (1, H)

            d. c_cell = SUM(mailbox(c) . f_tk) over Dimension(Nchn)
                The size of c mailbox(c) = size of f_tk
                c_cell size = (Nt, H)

            e. return c_cell

"""

from collections import defaultdict
import itertools as it

import numpy as np
import dgl

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from hotam.nn.layers.lstm import LSTM_LAYER


class TreeLSTMCell(nn.Module):
    def __init__(self, xemb_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(xemb_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

        self.W_f = nn.Linear(xemb_size, h_size, bias=False)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        self.b_f = nn.Parameter(th.zeros(1, h_size))

    def message_func(self, edges):
        return {
            "h": edges.src["h"],
            "c": edges.src["c"],
            "type_n": edges.src["type_n"],
        }

    def reduce_func(self, nodes):

        c_child = nodes.mailbox["c"]  # (Nt, Nchn, H)
        h_child = nodes.mailbox["h"]  # (Nt, Nchn, H)
        childrn_num = c_child.size(1)
        hidden_size = c_child.size(2)

        # Step 1
        type_n = nodes.mailbox["type_n"]  # (Nt)
        type_n0_id = type_n == 0  # 1.a
        type_n1_id = type_n == 1  # 1.a

        # 1.b: creat mask matrix with the same size of h and c with zeros at
        # either type_0 node ids or type_1 node ids
        mask = th.zeros((*h_child.size()))
        mask[type_n0_id] = 1  # mask one at type_0 nodes
        ht_0 = mask * h_child  # (Nt, Nchn, H)
        ht_0 = th.sum(ht_0, dim=1)  # sum over child nodes => (Nt, H)

        mask = th.zeros((*h_child.size()))  # do the same for type_1
        mask[type_n1_id] = 1
        ht_1 = mask * h_child  # (Nt, Nchn, H)
        ht_1 = th.sum(ht_1, dim=1)  # sum over child nodes => (Nt, H)

        # # Step 2
        h_iou = th.cat((ht_0, ht_1), dim=1)  # (Nt, 2H)

        # Step 3
        # (Nt, 2H) => (Nt, 2, H)
        f = self.U_f(th.cat((ht_0, ht_1), dim=1)).view(-1, 2, hidden_size)
        # 3.b select from f either f_0 or f_1 using type_n as index
        # generate array repeating elements of nodes_id by their number of
        # children. e.g. if we have 3 nodes that have 2 children.
        # select_id = [0, 0, 1, 1, 2, 2]
        select_id = np.repeat(range(c_child.size(0)), c_child.size(1))
        f_cell = f[select_id, type_n.view(-1), :].view(*c_child.size())

        # Steps 3.c,d
        X = self.W_f(nodes.data["emb"])  # (Nt, H)
        X = X.repeat(childrn_num, 1).view(*c_child.size())  # (Nt, Nchn, H)
        f_tk = th.sigmoid(X + f_cell + self.b_f)  # (Nt, Nchn, H)
        c_cell = th.sum(f_tk * c_child, dim=1)  # (Nt, H)

        return {"h": h_iou, "c": c_cell}

    def apply_node_func(self, nodes):
        # The leaf nodes have no child the h_child is initialized.
        h_cell = nodes.data["h"]
        c_cell = nodes.data["c"]

        # Initialization for leaf nodes
        if nodes._graph.srcnodes().nelement() == 0:  # leaf nodes
            # initialize h states, for node type-0 and node type-1
            # NOTE: initialization for node type-0 == node type-1
            h_cell = th.cat((h_cell, h_cell), dim=1)  # (Nt, Nchn*H)

        iou = self.W_iou(nodes.data["emb"]) + self.U_iou(h_cell) + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)  # (Nt x H) for each of i,o,u
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

        c = i * u + c_cell
        h = o * th.tanh(c)

        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        h_size,
        dropout=0,
        bidirectional=True,
    ):

        super(TreeLSTM, self).__init__()

        self.bidirectional = bidirectional
        self.TeeLSTM_cell = TreeLSTMCell(embedding_dim, h_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, root, leaves, h, c):
        """Compute modified N-ary tree-lstm (LSTM-ER) prediction given a batch.
        Parameters.
        ----------
        g : dgl.DGLGraph
            Batch of Trees for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The hidden state of each node.
        """

        # Tree-LSTM (LSTM-ER) according to arXiv:1601.00770v3 sections 3.3 & 3.4
        g.ndata["h"] = h
        g.ndata["c"] = c

        # copy graph
        if self.bidirectional:
            g_copy = g.clone()

        # propagate bottom top direction
        dgl.prop_nodes_topo(
            g,
            message_func=self.TeeLSTM_cell.message_func,
            reduce_func=self.TeeLSTM_cell.reduce_func,
            apply_node_func=self.TeeLSTM_cell.apply_node_func,
        )
        logits = g.ndata.pop("h")[root, :]

        if self.bidirectional:
            # propagate top bottom direction
            dgl.prop_nodes_topo(
                g_copy,
                message_func=self.TeeLSTM_cell.message_func,
                reduce_func=self.TeeLSTM_cell.reduce_func,
                apply_node_func=self.TeeLSTM_cell.apply_node_func,
                reverse=True,
            )
            h_tb = g_copy.ndata.pop("h")[leaves, :]
            # concatenate both tree directions
            logits = th.cat((logits, h_tb), dim=1)

        return logits


class NELabelEmbedding(nn.Module):
    def __init__(self, encode_size):

        super(NELabelEmbedding, self).__init__()
        self.encode_size = encode_size

    def forward(self, prediction_id, device):
        # label prediction, one-hot encoding for label embedding
        batch_size = prediction_id.size(0)
        label_one_hot = th.zeros(batch_size, self.encode_size, device=device)
        label_one_hot[th.arange(batch_size), prediction_id] = 1

        return label_one_hot


class NerModule(nn.Module):
    def __init__(
        self,
        token_embedding_size,
        label_embedding_size,
        h_size,
        ner_hidden_size,
        ner_output_size,
        bidirectional=True,
        num_layers=1,
        dropout=0,
    ):
        super(NerModule, self).__init__()

        self.model_param = nn.Parameter(th.empty(0))
        self.label_embedding_size = label_embedding_size

        # Entity label prediction according to arXiv:1601.00770v3, section 3.4
        # Sequential LSTM layer
        self.seqLSTM = LSTM_LAYER(
            input_size=token_embedding_size,
            hidden_size=h_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # Entity prediction layer: two layers feedforward network
        # The input to this layer is the the predicted label of the previous
        # word and the current hidden state.
        num_dirs = 2 if bidirectional else 1
        ner_input_size = label_embedding_size + (h_size * num_dirs)
        self.ner_decoder = nn.Sequential(
            nn.Linear(ner_input_size, ner_hidden_size),
            nn.Tanh(),
            nn.Linear(ner_hidden_size, ner_output_size),
        )

        self.entity_embedding = NELabelEmbedding(
            encode_size=label_embedding_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                batch_embedded,
                lengths,
                mask,
                pad_id=0.0,
                return_type=0,
                h=None,
                c=None):
        """Compute logits of and predict entities' label.
        ----------
        g :     dgl.DGLGraph
                Tree for computation.
        h :     Tensor
                Intial hidden state for the sequential LSTM
        c :     Tensor
                Cell hidden state for the sequential LSTM

        Returns
        -------
        logits : Tensor
                 (B, SEQ, NE-OUT)
                 logits for entities' label prediction

        label_id_predicted: Tensor
                            (B, SEQ)
                            The predicted labels id
        """
        device = self.model_param.device

        # LSTM layer;
        # dropout input,
        # sort batch according to the sample length
        batch_embedded = self.dropout(batch_embedded)
        lengths_sorted, ids_sorted = th.sort(lengths, descending=True)
        _, ids_original = th.sort(ids_sorted, descending=False)
        if h is not None and c is not None:
            lstm_out, _ = self.seqLSTM(
                (batch_embedded[ids_sorted], h.detach(), c.detach()),
                lengths_sorted)
        else:
            lstm_out, _ = self.seqLSTM(batch_embedded[ids_sorted],
                                       lengths_sorted)
        lstm_out = lstm_out[ids_original]

        # Entity prediction layer
        logits = []  # out_list_len=SEQ, inner_list_of_list_len=(B, NE-OUT)
        prob_dis = []
        label_id_predicted = []  # list_of_list_len=(SEQ, 1)

        # construct initial previous predicted label, v_{t-1}:v_0
        batch_size = lstm_out.size(0)
        seq_length = lstm_out.size(1)
        v_t_old = th.zeros(batch_size,
                           self.label_embedding_size,
                           device=device)
        for i in range(seq_length):  # loop over words
            # construct input, get logits for word_i, softmax, entity prediction
            ner_input = th.cat(
                (lstm_out[:, i, :].view(batch_size, -1), v_t_old), dim=1)
            # QSTN is this order is correct? Is the dim is correct? what is
            # dim=-1?
            logits_i = self.ner_decoder(ner_input)  # (B, NE-OUT)
            logits_i = self.dropout(logits_i)
            logits_i_copy = logits_i.detach().clone()
            mask_i = mask[:, i].type(th.bool)
            logits_i_copy[~mask_i] = float('-inf')
            prob_dist = F.softmax(logits_i_copy, dim=1)  # (B, NE-OUT)
            prediction_id = th.max(prob_dist, dim=1)[1]

            # entity label embedding
            label_one_hot = self.entity_embedding(prediction_id, device)
            # save data
            label_id_predicted.append(prediction_id.detach().tolist())
            logits.append(logits_i.detach().tolist())
            prob_dis.append(prob_dist.detach().tolist())
            v_t_old = label_one_hot  # v_{t-1} <- v_t

        # Reshape logits dimension from (SEQ, B, NE-OUT) to (B, SEQ, NE-OUT)
        # Reshape label_id_predicted from (SEQ, B) to (B, SEQ)
        logits = th.tensor(logits,
                           device=device,
                           dtype=th.float,
                           requires_grad=True).view(batch_size, seq_length, -1)
        prob_dis = th.tensor(prob_dis,
                             device=device,
                             dtype=th.float,
                             requires_grad=True).view(batch_size, seq_length,
                                                      -1)
        label_id_predicted = th.tensor(label_id_predicted,
                                       device=device,
                                       dtype=th.float).view(batch_size, -1)
        # TODO return type: what to return according to the input return_type
        return logits, prob_dis, label_id_predicted


class LSTM_RE(nn.Module):
    def __init__(self, hyperparamaters: dict, task2labels: dict,
                 feature2dim: dict):

        super(LSTM_RE, self).__init__()

        num_ne = len(task2labels["seg_ac"])  # number of named entities
        num_relations = len(task2labels["stance"])  # number of relations
        self.last_tkn_data = last_word_pattern(task2labels["seg_ac"])

        self.model_param = nn.Parameter(th.empty(0))

        self.graph_buid_type = hyperparamaters["graph_buid_type"]
        self.node_type_set = hyperparamaters["node_type_set"]

        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.BATCH_SIZE = hyperparamaters["batch_size"]

        # Embed dimension for tokens
        token_embs_size = feature2dim["word_embs"]
        # Embed dimension for entity labels
        label_embs_size = num_ne + 1
        # Embed dimension for dependency labels
        dep_embs_size = feature2dim["deprel"]
        # Sequential LSTM hidden size
        seq_lstm_h_size = hyperparamaters["seq_lstm_h_size"]
        # Tree LSTM hidden size
        tree_lstm_h_size = hyperparamaters["tree_lstm_h_size"]
        # Entity recognition layer hidden size
        ner_hidden_size = hyperparamaters["ner_hidden_size"]
        # Entity recognition layer output size
        ner_output_size = num_ne
        # Relation extraction layer hidden size
        re_hidden_size = hyperparamaters["re_hidden_size"]
        # Relation extraction layer output size
        re_output_size = num_relations
        # Sequential LSTM number of layer
        seq_lstm_num_layers = hyperparamaters["seq_lstm_num_layers"]
        # Sequential LSTM bidirection
        lstm_bidirectional = hyperparamaters["lstm_bidirectional"]
        # Tree LSTM bidirection
        tree_bidirectional = hyperparamaters["tree_bidirectional"]
        dropout = hyperparamaters["dropout"]

        # Entity recognition module
        self.module_ner = NerModule(token_embedding_size=token_embs_size,
                                    label_embedding_size=label_embs_size,
                                    h_size=seq_lstm_h_size,
                                    ner_hidden_size=ner_hidden_size,
                                    ner_output_size=ner_output_size,
                                    bidirectional=lstm_bidirectional,
                                    num_layers=seq_lstm_num_layers,
                                    dropout=dropout)

        # Relation extraction module
        nt = 3 if tree_bidirectional else 1
        ns = 2 if lstm_bidirectional else 1
        re_input_size = tree_lstm_h_size * nt
        tree_input_size = seq_lstm_h_size * ns + dep_embs_size + label_embs_size
        self.module_re = nn.Sequential(
            TreeLSTM(embedding_dim=tree_input_size,
                     h_size=tree_lstm_h_size,
                     dropout=dropout,
                     bidirectional=tree_bidirectional),
            nn.Linear(re_input_size, re_hidden_size), nn.Tanh(),
            nn.Linear(re_hidden_size, re_output_size))

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    @classmethod
    def name(self):
        return "LSTM_ER"

    def forward(self, batch):
        """Compute logits of and predict entities' label.
        ----------


        Returns
        -------

        """
        device = self.model_param.device
        token_embs = batch["word_embs"]  # (B, SEQ, W-E)
        # token_head_id = batch["dephead"]  # (B, SEQ)
        batch_size = token_embs.size(0)  # B

        # number of tokens in each sample - torch size (B, Sent_len)
        lengths_sent_tok = batch["lengths_sent_tok"]
        lengths_per_sample = batch["lengths_tok"]
        #   I. Sentence_id for each token: token2sent_id
        #    list of list: (B, Sent_len)
        #  II. Character start position for each sentence in each sample:
        #      sent_strt: list of list: (B, Sent_len)
        # III. Character end position for each sentence in each sample:
        #      sent_end: list of list: (B, Sent_len)
        token2sent_id = [
            list(
                it.chain.from_iterable(
                    it.repeat(val, cnt) for val, cnt in enumerate(lengths)))
            for lengths in lengths_sent_tok
        ]
        sent_strt = [[0] + list(it.accumulate(lengths_))[:-1]
                     for lengths_ in lengths_sent_tok]
        sent_end = [[s + len_ - 1 for s, len_ in zip(start, length)]
                    for start, length in zip(sent_strt, lengths_sent_tok)]
        mask = batch["token_mask"]

        # get token head absolute idx in format of list of list
        heads_abs = []
        for i, token_head_sample in enumerate(batch["dephead"].tolist()):
            head_abs_sample = []
            sent_strt_sample = sent_strt[i]
            sent_num = batch["lengths_sent"].tolist()[i]
            for j, heads in enumerate(token_head_sample[0:sent_num]):
                head_sample = heads[0:lengths_sent_tok[i][j]]
                sent_strt_l = [sent_strt_sample[j]] * len(head_sample)
                head_abs_sample += [
                    x + y for x, y in zip(head_sample, sent_strt_l)
                ]
            heads_abs.append(head_abs_sample)

        # NER Module:
        # ===========
        # NOTE Is it important to initialize h, c?
        logitss_ner, prob_ner, ne_predtd = self.module_ner(
            token_embs, lengths_per_sample, mask)

        # Get all possible relations in both direction between the last token
        # of the detected entities.
        graphs = [None] * batch_size
        g_empty = dgl.graph([])
        relations_data = sub_batch(ne_predtd, mask, self.last_tkn_data)
        # NOTE loop is needed because the sentence lengthes differ across
        for r1, r2, sample_ids in relations_data:
            # no relations, build empty graphs
            if r1.nelement() == 0:
                for i in sample_ids:
                    graphs[i] = g_empty
                continue
            # get sentence id for tokens in r1 and r2
            # samples. We use relations as in
            sent_id = th.tensor(token2sent_id[sample_ids],
                                device=device,
                                dtype=th.long)
            # NOTE U is range of
            sent_src_id = th.index_select(sent_id, dim=0, index=r1)
            sent_dist_id = th.index_select(sent_id, dim=0, index=r2)
            # not practical: epoch=0, we have 113050 graph!
            U = [
                list(range(sent_strt[sample_ids][u], sent_end[sample_ids][v]))
                for u, v in zip(sent_src_id, sent_dist_id)
            ]
            V = [[heads_abs[sample_ids][token] for token in U]]


        # Calculation of losses:
        # =======================
        # Entity Recognition Module.
        # (B, SEQ, NE-OUT) --> (B*SEQ, NE-OUT)
        # QSTN Pad token ignoring, should I change it to -1 in the ground
        # truth?
        # QSTN batch change pad value ddoes not work, either in
        # batch["word_embs"] or batch["seg_ac"]
        logitss_ner = logitss_ner.view(-1, logitss_ner.size(2))
        batch.change_pad_value(-1)  # ignore -1 in the loss function
        ground_truth_ner = batch["seg_ac"].view(-1)
        loss_ner = self.loss(logitss_ner, ground_truth_ner)

        # Relation Extraction Module.
        loss_total = loss_ner
        return {
            "loss": {
                "total": loss_total,
            },
            "preds": {
                "seg_ac": ne_predtd,
                # "relation": stance_preds,
                # "stance": stance_preds
            },
            "probs": {
                "seg_ac": prob_ner,
                # "relation": relation_probs,
                # "stance": stance_probs
            },
        }


def last_word_pattern(ne_labels):
    """
    """
    # Get patterns of pair of labels that determine the end of a named entity.
    ne_label_data = defaultdict(list)
    for i, label in enumerate(ne_labels):
        ne_label_data[label[0]].append(i)

    B_B = list(map(list, it.product(ne_label_data["B"], repeat=2)))
    B_O = [[i, j] for i in ne_label_data["B"] for j in ne_label_data["O"]]
    IB_IO = [[i, j] for i in ne_label_data["I"]
             for j in ne_label_data["O"] + ne_label_data["B"]]

    B_Idash_I_Idash = []
    for i, labeli in enumerate(ne_labels):
        for j, labelj in enumerate(ne_labels):
            if labeli[0] == "B" and labelj[0] == "I" or \
               labeli[0] == "I" and labelj[0] == "I":
                if labeli[1:] != labelj[1:]:
                    B_Idash_I_Idash.append([i, j])

    return B_B + B_O + IB_IO + B_Idash_I_Idash



def sub_batch(predictions, token_mask, last_token_pattern: list):
    """
    """
    # 1. Get predicted entities' last token
    #    last_token_pattern contains a list of two consecutive labels that
    #    idintfy the last token. These pairs are as follows:
    #    [B, B], [B, O], [I, B], [I, O],
    #    [B-Claim, I-not-Claim]: any combination of B-I that are not belong to
    #    the same entity class
    #    [I-Claim, I-not-Claim]: ny combination of I-I that are not belong to
    #    the same entity class
    #    Each time one of these pattern appears in the label sequence, the first
    #    label of the pair is the last token.
    #
    #    For example, suppose that we have the following sequence of tags; the
    #    last tokens will be:
    #          ↓  ↓     ↓     ↓        ↓     ↓
    #    B  I  I  B  O  B  B  I  O  I  I  O  B  O  [B] <--- dummy label to pair
    #    ++++  ----  ++++  ++++  ++++  ----  ----           the last label
    #       ++++  ----  ----  ----  ++++  ++++  +++++
    #
    #    Last tokens are marked with ↓, last token pattern are marked with
    #    (---) other pairs are marked with (+++).
    #
    #    Implementation Logic:
    #    --------------------
    #    a. Shift right the predicted label, append [B] at the end
    #    b. Form a list of zip like pairs: c = torch.stack((a,b), dim=2)
    #       c has a shape of (B, SEQ, 2)
    #    c. Search for patterns; (c[:, :, None] == cmp).all(-1).any(-1)

    # 2. for each sample, get all posiible relation in both directions between
    #    last token

    # step: 1, mask last word by true
    d = th.device('cpu')
    last_token_pattern = th.tensor(last_token_pattern, dtype=th.long,
                                   device=d).view(-1, 2)
    B_lbl = th.tensor([last_token_pattern[0][0]] * predictions.size(0),
                      dtype=th.long,
                      device=d).view(-1, 1)
    p = predictions.detach().clone().cpu()
    p_rshifted = th.cat((p[:, 1:], B_lbl), dim=1)
    p_zipped = th.stack((p, p_rshifted), dim=2)
    last_word_mask = (p_zipped[:, :,
                               None] == last_token_pattern).all(-1).any(-1)

    # # step: 2: group tensors that have the same number of last words
    # group_data = defaultdict(list)
    # num_last_tkns = last_word_mask.sum(dim=1)
    # for i, num in enumerate(num_last_tkns.detach().tolist()):
    #     group_data[num].append(i)

    # step: 2
    # NOTE For loop is needed is the number of items are different
    for i in range(last_word_mask.size(0)):
        # get the last token id in each sample
        # select rows in last_word_mask using idx
        # get the lat word id
        # idx_tensor = th.tensor(idx, dtype=th.long)  # sample id to tensor
        sample_mask = last_word_mask[i, :]
        sample_mask = sample_mask[token_mask[i, :]]
        last_word_id = th.nonzero(sample_mask).flatten()
        # get relations per sample
        # list of list of list. The relation in each sample is in list of
        # list.
        # For example, assume in sample-i there are relations betwenn
        # tokens-(1,4) and tokens-(5,9). Then the relation will be in form
        # of [[1, 5], [4, 9]]
        r = list(map(list, zip(*it.combinations(last_word_id.tolist(), 2))))
        r1 = th.tensor(r[0], device=d)
        r2 = th.tensor(r[1], device=d)
        yield r1, r2, i


# - deprel,  e.g. "nmod" etc  (shape = (nr_sentence, max_tok_in_sent  ))

# - dephead, e.g. for each token on which idx is it's head (shape =
#   (nr_sentence, max_tok_in_sent))

# - lengths_sent,  nr sentence is sample

# - lengths_sent_tok, nr token in each sentence in the sample, given this we
#   can get is_sent_end.

# - ac2sentence, which is vector where ac2sentence[ac_idx] given u the idx of
#   the sentence the ac is in

# - sent_ac_mask, which given an ac_idx gives you a mask over a sentence
#   telling you which tokens belong to the ac.