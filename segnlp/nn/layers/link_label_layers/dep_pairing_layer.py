
#basics
from typing import List, Tuple
import functools

#pytorch
import torch
from torch import Tensor
import torch.nn as nn

#networkx
import networkx as nx
from networkx import Graph as nxGraph

#DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo

#segnlp
from segnlp.nn.layers.rep_layers import TypeTreeLSTM



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

        self.label_link_clf = nn.Sequential(
            nn.Linear(decoder_input_size, decoder_h_size), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(decoder_h_size,
                                           decoder_output_size))

        self.__supported_modes = set(["shortest_path"])

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
        M = token_mask.clone().type(torch.bool)

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

    def build_dep_graphs(self, token_embs: Tensor, deplinks: Tensor,
                         roots: Tensor, token_mask: Tensor, token_reps: Tensor,
                         subgraphs: List[List[Tuple]], mode: str,
                         assertion: bool) -> List[DGLGraph]:

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
        dep_graphs = dgl.batch(dep_graphs).to(self.device)
        dep_graphs.ndata["emb"] = nodes_emb

        return dep_graphs

    def forward(self,
                input_embs: Tensor,
                unit_repr: Tensor,
                dependencies: Tensor,
                token_mask: Tensor,
                roots: Tensor,
                pairs: List[List[Tuple[int]]],
                mode: str = "shortest_path",
                assertion: bool = False):

        mode_bool = mode in self.__supported_modes
        assert mode_bool, f"{mode} is not a supported mode for DepPairingLayer"

        self.device = input_embs.device
        dir_n = 2 if self.tree_lstm_bidir else 1

        # 8)
        dep_graphs = self.build_dep_graphs(token_embs=input_embs,
                                           deplinks=dependencies,
                                           roots=roots,
                                           token_mask=token_mask,
                                           token_reps=input_embs,
                                           subgraphs=pairs,
                                           mode=mode,
                                           assertion=assertion)

        # 9)
        h0 = torch.zeros(dep_graphs.num_nodes(),
                         self.tree_lstm_h_size,
                         device=self.device)
        c0 = torch.zeros_like(h0)
        tree_lstm_out = self.tree_lstm(dep_graphs, h0, c0)

        # construct dp = [↑hpA; ↓hp1; ↓hp2]
        # ↑hpA: hidden state of dep_graphs' root
        # ↓hp1: hidden state of the first token in the candidate pair
        # ↓hp2: hidden state of the second token in the candidate pair
        # get ids of roots and token in relation
        root_id = (dep_graphs.ndata["root"] == 1)
        start_id = dep_graphs.ndata["start"] == 1
        end_id = dep_graphs.ndata["end"] == 1
        tree_lstm_out = tree_lstm_out.view(-1, dir_n, self.tree_lstm_h_size)
        tree_logits = tree_lstm_out[root_id, 0, :]  # ↑hpA
        if self.tree_lstm_bidir:
            hp1 = tree_lstm_out[start_id, 1, :]  # ↓hp1
            hp2 = tree_lstm_out[end_id, 1, :]  # ↓hp2
            tree_logits = torch.cat((tree_logits, hp1, hp2), dim=-1)
        # [dp; s]
        link_label_input_emb = torch.cat((tree_logits, unit_repr), dim=-1)
        logits = self.label_link_clf(link_label_input_emb)

        # 10) Here we should format the data to the following structure:
        # t1 = representation of the last token in the first unit of the pair
        # t2 = representation of the last token in the second unit of the pair
        # a = lowest ancestor of t1 and t2
        # pair(unit_i, unit_j) = a+t1+t2 where t1
        # (batch_size, nr_units, nr_units, a+t1+t2)
        # for a sample:
        # [
        #   [
        #    pair(unit0,unit0),
        #       ...
        #     pair(unit0, unitn),
        #   ],
        #    ....
        #   [
        #    pair(unitn,unitn),
        #       ...
        #     pair(unitn, unitn+1),
        #   ],
        #
        # ]
        # pairs = ""

        # now we should get logist  for each link_labels
        # (batch_size, nr_units, nr_units, nr_link_labels)
        #
        # for a sample:
        # [
        #   [
        #    [link_label_0_score, .., link_label_n_score],
        #       ...
        #    [link_label_0_score, .., link_label_n_score]
        #   ],
        #   [
        #    [link_label_0_score, .., link_label_n_score],
        #       ....
        #    [link_label_0_score, .., link_label_n_score]
        #   ],
        # ]
        # link_label_logits = self.link_label_clf(pairs)

        # 11)
        # first we get the index of the unit each unit links to
        # we do this by first get the highest score of the link label
        # for each unit pair. Then we argmax that to get the index of
        # the linked unit.
        # max_link_label_logits = torch.max(link_label_logits, dim=-1)
        # link_preds = torch.argmax(max_link_label_logits, dim=-1)

        # 12)
        # we index the link_label_scores by the link predictions, selecting
        # the logits for the link_labels for the linked pairs
        # top_link_label_logits = index_4D(link_label_logits, index=link_preds)

        # return top_link_label_logits, link_preds
