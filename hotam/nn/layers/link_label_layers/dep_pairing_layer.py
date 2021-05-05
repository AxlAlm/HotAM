# basics
from typing import List, Tuple, DefaultDict, Union, Dict
import functools
from itertools import repeat
from functools import reduce
from operator import iconcat
from hotam.utils import nCr

import numpy as np

# pytorch
import torch
from torch import Tensor
import torch.nn as nn
# import torch.nn.functional as F

# networkx
import networkx as nx
from networkx import Graph as nxGraph

# DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo

# hotam
from hotam.nn.layers.type_treelstm import TypeTreeLSTM
from hotam.nn.utils import get_all_possible_pairs
from hotam.nn.utils import agg_emb
from hotam.nn.utils import cumsum_zero


class DepGraph:
    def __init__(self,
                 token_embs: Tensor,
                 deplinks: Tensor,
                 roots: Tensor,
                 token_mask: Tensor,
                 subgraphs: List[List[Tuple]],
                 mode: str,
                 device=None,
                 assertion: bool = False) -> List[DGLGraph]:

        assert mode in set([
            "shortest_path"
        ]), f"{mode} is not a supported mode for DepPairingLayer"
        self.device = device
        batch_size = deplinks.size(0)

        # creat sample graphs G(U, V) tensor on CPU
        U, V, M = self.get_sample_graph(deplinks=deplinks,
                                        roots=roots,
                                        token_mask=token_mask,
                                        assertion=assertion)
        # creat sub_graph for each pair
        dep_graphs = []
        nodes_emb = []
        for b_i in range(batch_size):
            mask = M[b_i]
            u = U[b_i][mask]
            v = V[b_i][mask]
            # creat sample DGLGraph, convert it to unidirection, separate the
            # list of tuples candidate pairs into two lists: (start and end
            # tokens), then create subgraph
            graph = dgl.graph((u, v))
            graph_unidir = graph.to_networkx().to_undirected()
            start, end = list(zip(*subgraphs[b_i]))

            if start == []:  # no candidate pair
                continue

            subgraph_func = functools.partial(self.get_subgraph,
                                              g=graph,
                                              g_nx=graph_unidir,
                                              sub_graph_type=mode,
                                              assertion=assertion)

            dep_graphs.append(dgl.batch(list(map(subgraph_func, start, end))))

            # get nodes' token embedding
            nodes_emb.append(token_embs[b_i, dep_graphs[b_i].ndata["_ID"]])

        # batch graphs, move to model device, update nodes data by tokens
        # embedding
        nodes_emb = torch.cat(nodes_emb, dim=0)
        self.graphs = dgl.batch(dep_graphs).to(self.device)
        self.graphs.ndata["emb"] = nodes_emb

    def assert_graph(self, u: Tensor, v: Tensor, roots: Tensor,
                     self_loop: Tensor) -> None:

        # check that self loops do not exist in places other than roots
        self_loop_check = u == roots[:, None]
        if not torch.all(self_loop == self_loop_check):
            # we have a problem. Get samples ids where we have more than one
            # self loop
            problem_id = torch.where(torch.sum(self_loop, 1) > 1)[0]
            self_loop_id = v[problem_id, :][self_loop[problem_id, :]].tolist()
            theroot = roots[problem_id].tolist()
            error_msg_1 = f"Self loop found in sample(s): {problem_id}, "
            error_msg_2 = f"Node(s) {self_loop_id}. Root(s): {theroot}."
            raise Exception(error_msg_1 + error_msg_2)
            # remove the sample that has the problem

    def get_sample_graph(self, deplinks: Tensor, roots: Tensor,
                         token_mask: Tensor, assertion: bool) -> List[Tensor]:

        batch_size = deplinks.size(0)
        max_lenght = deplinks.size(1)
        device = torch.device("cpu")

        # G(U, V)
        U = torch.arange(max_lenght, device=self.device).repeat(batch_size, 1)
        V = deplinks.clone()
        M = token_mask.clone()

        # remove self loops at root nodes
        self_loop = U == V
        if assertion:
            self.assert_graph(U, V, roots, self_loop)
        U = U[~self_loop].view(batch_size, -1).to(device)
        V = V[~self_loop].view(batch_size, -1).to(device)
        M = M[~self_loop].view(batch_size, -1).to(device)

        return [U, V, M]

    def get_subgraph(self,
                     start: int,
                     end: int,
                     g: DGLGraph,
                     g_nx: nxGraph,
                     sub_graph_type: str,
                     assertion: bool = False) -> DGLGraph:
        """
        """
        if sub_graph_type == "shortest_path":
            thepath = nx.shortest_path(g_nx, source=start, target=end)
            sub_g = dgl.node_subgraph(g, thepath)
            root = list(traverse_topo(sub_g))[-1]

            # initialize node data
            # Node type
            node_type = torch.zeros(sub_g.number_of_nodes(), dtype=torch.long)
            sub_g.ndata["type_n"] = node_type

            # Root, start and end leaves node
            str_mark = torch.zeros(sub_g.number_of_nodes())
            end_mark = torch.zeros_like(str_mark)
            root_mark = torch.zeros_like(str_mark)
            str_mark[0] = 1
            end_mark[-1] = 1
            root_mark[root] = 1
            sub_g.ndata["root"] = root_mark
            sub_g.ndata["start"] = str_mark
            sub_g.ndata["end"] = end_mark

            # check ...
            if assertion:
                assert len(root) == 1
                assert sub_g.ndata["_ID"][0] == start
                assert sub_g.ndata["_ID"][-1] == end
                assert str_mark.sum() == end_mark.sum()
                assert str_mark.sum() == root_mark.sum()

        elif sub_graph_type == 1:
            # get subtree
            pass
        return sub_g


class DepPairingLayer(nn.Module):
    def __init__(self,
                 tree_input_size: int,
                 tree_lstm_h_size: int,
                 tree_bidirectional: bool,
                 decoder_input_size: int,
                 decoder_h_size: int,
                 decoder_output_size: int,
                 dropout: int = 0.0):
        super(DepPairingLayer, self).__init__()

        self.tree_lstm_h_size = tree_lstm_h_size
        self.tree_lstm_bidir = tree_bidirectional

        self.tree_lstm = TypeTreeLSTM(embedding_dim=tree_input_size,
                                      h_size=tree_lstm_h_size,
                                      dropout=dropout,
                                      bidirectional=tree_bidirectional)

        self.link_label_clf_layer = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_h_size), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(decoder_h_size,
                                           decoder_output_size))

    def build_pair_embs(self,
                        token_embs: Tensor,
                        dep_embs: Tensor,
                        one_hot_embs: Tensor,
                        roots: Tensor,
                        token_mask: Tensor,
                        deplinks: Tensor,
                        bio_data: dict,
                        mode: str = "shortest_path",
                        assertion: bool = False):

        # get all possible pairs
        pair_data = get_all_possible_pairs(start=bio_data["unit"]["start"],
                                           end=bio_data["unit"]["end"],
                                           bidir=True)

        # 1) Build graph from dependecy data
        node_embs = torch.cat((token_embs, one_hot_embs, dep_embs), dim=-1)
        G = DepGraph(token_embs=node_embs,
                     deplinks=deplinks,
                     roots=roots,
                     token_mask=token_mask,
                     subgraphs=pair_data["end"],
                     mode=mode,
                     device=self.device,
                     assertion=assertion)

        # 2) Pass graph to a TreeLSTM to create hidden representations
        h0 = torch.zeros(G.graphs.num_nodes(),
                         self.tree_lstm_h_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        tree_lstm_out = self.tree_lstm(G.graphs, h0, c0)

        # construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and tokens in relation
        root_id = (G.graphs.ndata["root"] == 1)
        start_id = G.graphs.ndata["start"] == 1
        end_id = G.graphs.ndata["end"] == 1

        tree_lstm_out = tree_lstm_out.view(-1, 2, self.tree_lstm_h_size)
        tree_pair_embs = torch.cat(
            (
                tree_lstm_out[root_id, 0, :],  # ↑hpA
                tree_lstm_out[start_id, 1, :],  # ↓hp1
                tree_lstm_out[end_id, 1, :]  # ↓hp2
            ),
            dim=-1)

        # 3) create unit embeddings
        unit_embs_flat = agg_emb(
            token_embs,
            bio_data["unit"]["lengths"],
            bio_data["unit"]["span_idxs"],
            mode="average",
            # flat will returned all unit embs with padding removed in a 2D
            # tensor
            flat=True,
        )
        # p1 and p2 are the unit indexs of each pair.
        # p1[k] = token_idx_i_in_sample_k
        # p2[k] = token_idx_i_in_sample_k
        p1 = torch.cat(pair_data["p1"])  # NOTE hstack is avail in v1.8 only
        p2 = torch.cat(pair_data["p2"])

        # p1 and p2 are indexes of each pair, but we want to create a flat
        # tensor with all the pairs by selecting them using indexing for a flat
        # tensor of unit_embeddings.  hence we need to update the indexes so
        # that each unit index in p1/p2, is relative to the global index of all
        # units
        cum_lens = cumsum_zero(torch.LongTensor(bio_data["unit"]["lengths"]))
        global_index = torch.repeat_interleave(cum_lens, pair_data["lengths"])

        p1g = p1 + global_index
        p2g = p2 + global_index

        unit_pair_embs = torch.cat((unit_embs_flat[p1g], unit_embs_flat[p2g]),
                                   dim=-1)

        # combine embeddings from lstm and from unit embeddings
        # dp´ = [dp; s1,s2]
        pair_embs = torch.cat((tree_pair_embs, unit_pair_embs), dim=-1)

        return pair_embs, pair_data

    def get_pair_comb_ids(
            self, pair_data: DefaultDict,
            probs_all: Tensor) -> Tuple[Union[np.array, List[int]]]:
        # Get indices for candidate pairs from left to right (forward, p1->p2)
        # and from right to left (backward, p2->p1) and roots (p1->p1)
        # Candidate pairs are a production of itertools.product(units_ids),
        # thus the upper triangle of a square matrix that have a size of
        # (num_units, num_units) forms the forward direction
        # (u_i -> u_j | i > j). In the same way, the diagonal forms the root
        # (u_i -> u_j | i = j), and the lower triangle forms the backward
        # direction (u_i -> u_j | i < j)
        frwrd_dir = np.array([], dtype=np.long)
        bckwrd_dir = np.array([], dtype=np.long)
        roots = np.array([], dtype=np.long)
        shift = 0
        unit_num = []
        for i, pair_id in enumerate(pair_data['p1']):
            n = pair_id[-1].item() + 1  # number of units in the batch
            unit_num.append(n)
            if n > 1:
                frwrd = np.ravel_multi_index(np.triu_indices(n, k=1),
                                             dims=(n, n)) + shift
                bkwrd = np.stack(np.tril_indices(n, k=-1))
                # sort bkwrd by column id, to align with elements in frwrd
                bkwrd = bkwrd[:, np.lexsort(bkwrd)]
                bkwrd = np.ravel_multi_index(bkwrd, dims=(n, n)) + shift
                self_loop = np.ravel_multi_index(np.diag_indices(n),
                                                 dims=(n, n)) + shift
                frwrd_dir = np.hstack((frwrd_dir, frwrd))
                bckwrd_dir = np.hstack((bckwrd_dir, bkwrd))
                roots = np.hstack((roots, self_loop))
                shift = bkwrd[-1] + 2

        # Get the maximum probability amoung the forward and backword direction
        # (num_pairs, 2, features_size)
        probs_2dir = torch.stack(
            (probs_all[frwrd_dir], probs_all[bckwrd_dir])).permute(1, 0, 2)
        max_dir = torch.argmax(torch.max(probs_2dir, dim=-1)[0], dim=-1)
        max_dir_id = np.stack([frwrd_dir, bckwrd_dir],
                              axis=1)[np.arange(max_dir.size(0)),
                                      max_dir.tolist()]
        select_idx = np.hstack([max_dir_id, roots])

        return select_idx, unit_num

    def split_nested_list(self, nested_list: List[int]) -> Tuple[List[int]]:
        nl_split = reduce(iconcat, nested_list, [])  # type: List[Tuple[int]]
        list_1, list_2 = np.array(list(zip(*nl_split)))
        return list_1, list_2

    def split_2dtensor_start_end(self, matrix: Tensor, start_idx: np.array,
                                 end_idx: np.array, idx0: np.array):
        idx_0 = idx0.copy()
        span_length = end_idx - start_idx
        idx_0 = np.repeat(idx_0, span_length)
        idx_1 = np.hstack(list(map(np.arange, start_idx, end_idx)))

        matrix_split = torch.split(matrix[idx_0, idx_1],
                                   split_size_or_sections=span_length.tolist())

        return matrix_split

    def get_preds(self, ll_probs_all: Tensor, select_idx: np.array,
                  pair_data: DefaultDict) -> Tuple[Tensor]:
        # Form the prediction and probs based on units combinatory
        ll_probs = ll_probs_all[select_idx]
        ll_preds_id = torch.argmax(ll_probs, dim=-1)
        # Get link prediction, i.e. the second unit in the pair if link_label
        # is not none, else link prediction is self loop.
        p1_id = torch.cat(pair_data['p1'])[select_idx]
        l_preds_id = torch.cat(pair_data['p2'])[select_idx]
        ll_preds_none = ll_preds_id == 0
        l_preds_id[ll_preds_none] = p1_id[ll_preds_none]

        return l_preds_id, ll_preds_id, ll_probs

    def get_targets(self, token_label: Dict[str, Tensor], unit_num: List[int],
                    select_idx: np.array,
                    pair_data: DefaultDict) -> Tuple[Tensor]:

        l_target = token_label['link_target']
        ll_target = token_label['link_label_targets']
        batch_size = l_target.size(0)

        # Form the ground truth based on units combinatory
        p1_end, p2_end = self.split_nested_list(pair_data["end"])
        p1_end, p2_end = p1_end[select_idx], p2_end[select_idx]

        num_pair_comb = list(map(nCr, unit_num, repeat(2)))
        idx_0 = np.hstack((np.repeat(np.arange(batch_size), num_pair_comb),
                           np.repeat(np.arange(batch_size), unit_num)))
        l_target = l_target[idx_0, p1_end - 1]
        ll_target = ll_target[idx_0, p1_end - 1]

        return l_target, ll_target

    def get_negative_relations(self, token_label: Dict[str, Tensor],
                               pair_data: DefaultDict, select_idx: np.array,
                               unit_num: List[int]) -> Tensor:
        # Negative relations
        seg_preds = token_label['preds']
        seg_targets = token_label['targets']
        batch_size = seg_preds.size(0)
        seg_targets_pad = seg_targets == -1
        seg_preds[seg_targets_pad] = -1  # set pad token to -1 in predections
        seg_preds_wrong = seg_preds != seg_targets

        # Construct start and end indices for all tokens belongs to each pair
        p1_start, p2_start = self.split_nested_list(pair_data["start"])
        p1_end, p2_end = self.split_nested_list(pair_data["end"])
        p1_start, p2_start = p1_start[select_idx], p2_start[select_idx]
        p1_end, p2_end = p1_end[select_idx], p2_end[select_idx]

        num_pair_comb = list(map(nCr, unit_num, repeat(2)))
        idx_0 = np.hstack((np.repeat(np.arange(batch_size), num_pair_comb),
                           np.repeat(np.arange(batch_size), unit_num)))

        # Get the prediction status (predection == target) for each token in
        # each pair. All tokens need to be predicted correctly
        seg_preds_wrong_p1 = self.split_2dtensor_start_end(
            seg_preds_wrong, p1_start, p1_end, idx_0)
        seg_preds_wrong_p2 = self.split_2dtensor_start_end(
            seg_preds_wrong, p2_start, p2_end, idx_0)

        seg_preds_wrong_p1 = torch.stack(
            [torch.all(pair_seg) for pair_seg in seg_preds_wrong_p1])
        seg_preds_wrong_p2 = torch.stack(
            [torch.all(pair_seg) for pair_seg in seg_preds_wrong_p2])

        negative_relation_pair = torch.logical_and(seg_preds_wrong_p1,
                                                   seg_preds_wrong_p2)

        return negative_relation_pair

    def forward(
        self,
        token_embs: Tensor,
        dep_embs: Tensor,
        one_hot_embs: Tensor,
        roots: Tensor,
        token_mask: Tensor,
        deplinks: Tensor,
        bio_data: dict,
        mode: str = "shortest_path",
        token_label: Dict[str, Tensor] = None,
        assertion: bool = False,
    ):

        self.device = token_embs.device

        # max_units = bio_data["max_units"]

        # essentially, we do 3 things:
        # 1) build a graph
        # 2) pass the graph to lstm to get the dp
        # 3) average token embs to create unit representations
        #
        # we return dp´and the global unit indexes for unit1 and unit2 in pairs
        pair_embs, pair_data = self.build_pair_embs(token_embs=token_embs,
                                                    dep_embs=dep_embs,
                                                    one_hot_embs=one_hot_embs,
                                                    roots=roots,
                                                    token_mask=token_mask,
                                                    deplinks=deplinks,
                                                    bio_data=bio_data,
                                                    mode=mode,
                                                    assertion=assertion)

        # We predict link labels for both directions. Get the dominant pair dir
        # plus roots' probabilities
        ll_probs_all = self.link_label_clf_layer(pair_embs)
        select_idx, unit_num = self.get_pair_comb_ids(pair_data, ll_probs_all)

        # Get predictions and targets
        l_preds_id, ll_preds_id, ll_preds_probs = self.get_preds(
            ll_probs_all, select_idx, pair_data)
        l_target, ll_target = self.get_targets(token_label, unit_num,
                                               select_idx, pair_data)

        # Set negative relations
        neg_rel_bool = self.get_negative_relations(token_label, pair_data,
                                                   select_idx, unit_num)
        # NOTE it is better to acess the none link_label from init
        ll_preds_id[neg_rel_bool] = 0
        ll_preds_probs[neg_rel_bool] = ll_preds_probs.new_tensor(
            [0.999, 0.0005, 0.0005, 0.0005, 0.0005], requires_grad=True)
        p1_id = torch.cat(pair_data['p1'])[select_idx]
        l_preds_id[neg_rel_bool] = p1_id[neg_rel_bool]

        return {
            "link_preds": l_preds_id,
            "link_label_preds": ll_preds_id,
            "link_label_probs": ll_preds_probs,
            "link_label_target": ll_target
        }
