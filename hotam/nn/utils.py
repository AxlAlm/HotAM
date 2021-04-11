import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

from typing import List, Tuple, DefaultDict
from math import floor, exp
from random import random
from itertools import product, repeat
from collections import defaultdict

def masked_mean(m, mask):
    """
    means the rows in a given matrix based on the masked values.

    example:

    if v == m[1] and mask[i] == [0,1,1,1,0]
    mask_mean(v) = mean(v[1:4])

    Parameters
    ----------
    m : tensor
        matrix to be averaged
    mask : mask
        matrix of where m[i,j] = [0,1]

    Returns
    -------

        [description]
    """
    m_sum = torch.sum(m, dim=2)
    mask_sum = torch.sum(mask, dim=2, keepdim=True)
    masked_mean = m_sum / mask_sum
    masked_mean[masked_mean != masked_mean] = 0.0
    return masked_mean


def multiply_mask_matrix(matrix, mask):
    """
        Tedious way of multiplying a 4D matrix with a 3D mask.

        Example when we need to do this is when we have a matrix of word
        embeddings for paragraphs (batch_size, nr_paragraphs, nr_tok, word_emb)
        and we want to get spans of the paragraphs based on a mask.

        We can then use the mask which signify the spans to turn everything we
        dont want to zeros
    """
    og_shape = matrix.shape

    # # flatten to 2D
    matrix_f = torch.flatten(matrix, end_dim=-2)

    # # 2 flatten mask to 2D, last value needs to be [0] or [1]
    mask_f = mask.view((np.prod([matrix.shape[:-1]]), 1))

    # # 3 turn embs into 0 where mask is 0
    matrix_f_masked = matrix_f * mask_f

    # # 4 return to original shape
    masked_matrix = matrix_f_masked.view(og_shape)

    return masked_matrix


def agg_emb(m, lengths, span_indexes, mode="average"):

    if mode == "mix":
        feature_dim = m.shape[-1]*3
    else:
        feature_dim = m.shape[-1]

    batch_size = m.shape[0]
    device = m.device
    agg_m = torch.zeros(batch_size, torch.max(lengths), feature_dim, device=device)

    for i in range(batch_size):
        for j in range(lengths[i]):
            ii, jj = span_indexes[i][j]

            if mode == "average":
                agg_m[i][j] = torch.mean(m[i][ii:jj], dim=0)

            elif mode == "max":
                v, _ = torch.max(m[i][ii:jj])
                agg_m[i][j] = v

            elif mode == "min":
                v, _ = torch.max(m[i][ii:jj])
                agg_m[i][j] = v

            elif mode == "mix":
                _min, _ = torch.min(m[i][ii:jj],dim=0)
                _max, _ = torch.max(m[i][ii:jj], dim=0)
                _mean = torch.mean(m[i][ii:jj], dim=0)

                agg_m[i][j] = torch.cat((_min, _max, _mean), dim=0)

            else:
                raise RuntimeError(f"'{mode}' is not a supported mode, chose 'min', 'max','mean' or 'mix'")

    return agg_m


# def reduce_and_remove(matrix, mask):

#     """
#     Given a 4D matrix turn it into a 3D by removing 3D dimension while perserving padding.

#     (similar to pytroch.utils.pad_packed_sequences, sort of)

#     for example:
#         given a 4D matrix where dims are (batch_size, nr_paragraphs, nr_spans, feature_dim),
#         if we want to get all words for all paragraphs we need to remove spans and remove padding tokens.
#         we cannot remove all values == n as padding for nr words in paragraphs needs to exists.
#         So, we need to find out max paragraph length, remove all zeros between that length and then after.

#         Given (batch_size, nr_paragraphs, nr_spans, nr_tokens) we get
#     """
#     batch_size, _, _, feature_dim = matrix.shape

#     matrix_f = matrix[mask]
#     lengths = torch.sum(torch.sum(mask, dim=-1),dim=1)

#     new_tensor = torch.zeros((batch_size, max(lengths), feature_dim))

#     start_idx = 0
#     for end_idx in lengths:
#         new_tensor[:end_idx] = matrix_f[start_idx:start_idx+end_idx]
#         start_idx += end_idx

#     return new_tensor


def index_4D(a: torch.tensor, index: torch.tensor):
    """
    a is 4D tensors
    index is 3D tensor

    index will select values/vectors

    link_label_score
    [
        [
            [
                [score10 ,..., scoren0], <- probability that its linked itself
                ...
                [score1n ,..., scorenn] <- probability that its linked to unit at index n
            ]

        ]
    ]



    link_scores
    [
        [
            [
                score1,
                scorenn
            ]
        ]
    ]


    argmax(
            [
                [
                    [
                        score1,
                        scorenn
                    ]
                ]
            ])


        [
            [
                2
            ]
        ]
    """
    b = torch.zeros((a.shape[0], a.shape[1], a.shape[-1]))
    for i in range(index.shape[0]):
        for j, k in enumerate(index[i]):
            b[i][j] = a[i][j][k]
    return b


def get_all_possible_pairs(
                            starts: List[List[int]],
                            ends: List[List[int]],
                            assertion: bool = False
                            ) -> DefaultDict[str, List[List[Tuple[int]]]]:

    all_possible_pairs = defaultdict(list)
    for idx_start, idx_end in zip(starts, ends):

        all_possible_pairs["idx"].append(list(product(list(range(len(idx_start))), repeat=2)))
        all_possible_pairs["start"].append(list(product(idx_start, repeat=2)))
        all_possible_pairs["end"].append(list(product(idx_end, repeat=2)))

        # if assertion:
        #     lens_cal = idx_end - idx_start
        #     span_len = np.array(span)[np.array(mask, dtype=bool)]
        #     assert np.all(lens_cal == span_len)

    return all_possible_pairs


def range_3d_tensor_index(matrix: Tensor,
                          start: List[int],
                          end: List[int],
                          pair_batch_num: List[int],
                          reduce_: str = "none") -> Tensor:

    # to avoid bugs, if there is a sample that does not have a unit the
    # corresponding len should be zero in batch_lens.
    batch_size = matrix.size(0)
    dim_1_size = matrix.size(1)
    new_size = (batch_size, dim_1_size)
    shape_ = len(matrix.size())

    reduce_fn = reduce_ in ["none", "mean", "sum"]
    # assertion messages:
    reduce_msg = f"Function \"{reduce_}\" is not a supported."
    num_msg = "Wrong number of pairs per sample is provided. "
    num_msg += f"Provided {len(pair_batch_num)}, expected {batch_size}."
    assert reduce_fn, reduce_msg
    assert batch_size == len(pair_batch_num), num_msg
    assert shape_ == 3, f"Wrong matrix shape, provided {shape_}, expected 3."

    # change matrix to be 2d matrix (dim0*dim1, dim2)
    # (batch_size, nr_tokens, word_emb)
    # (batch_size*nr_tokens, word_emb)
    mat = matrix.clone().contiguous().view(-1, matrix.size(-1))

    # construct array of indices for dimesion 0, repeating batch_id
    span_len = np.array(end) - np.array(start)
    idx_0 = np.repeat(np.arange(batch_size), pair_batch_num)
    idx_0 = np.repeat(idx_0, span_len)

    # construct array of indices for dimesion 1
    idx_1 = np.hstack(list(map(np.arange, start, end)))

    # Converts idx_0 and idx_1 into an array of indices suitable for the
    # converted 2d tensor
    idx_0_2d = np.ravel_multi_index(np.array([idx_0, idx_1]), new_size)

    # matrix = [[i,j],[i,j]]

    # index 2d tensor using idx_0_2d
    mat = torch.split(mat[idx_0_2d, :], span_len.tolist())
    if reduce_ == "mean":
        mat = torch.stack(list(map(torch.mean, mat, repeat(0))))
    elif reduce_ == "sum":
        mat = torch.stack(list(map(torch.sum, mat, repeat(0))))
    
    #(batch_size, max_units, dim)
    return mat


def util_one_hot(matrix: Tensor, mask: Tensor, num_classes: int):
    # check padding = -1
    thematrix = matrix.clone()  # avoid possible changing of the original Tensor
    pad_emb = thematrix[~mask.type(torch.bool)]
    if torch.all(pad_emb == -1):
        # change padding = 0
        pad_emb = ~mask.type(torch.bool) * 1
        thematrix += pad_emb

    return F.one_hot(thematrix, num_classes=num_classes)


def unfold_matrix(matrix_to_fold: Tensor, start_idx: List[int],
                  end_idx: List[int], class_num_betch: List[int],
                  fold_dim: int) -> Tensor:

    batch_size = matrix_to_fold.size(0)
    # construct array of indices for dimesion 0, repeating batch_id
    span_len = np.array(end_idx) - np.array(start_idx)
    idx_0 = np.repeat(np.arange(batch_size), class_num_betch)
    idx_0 = np.repeat(idx_0, span_len)

    # construct array of indices for dimesion 1
    idx_1 = np.hstack(list(map(np.arange, start_idx, end_idx)))

    # Get unit id for each start, end token
    unit_id = np.hstack(list(map(np.arange, repeat(0), class_num_betch)))
    unit_id = np.repeat(unit_id, span_len)

    # construct the folded matrix
    if len(matrix_to_fold.size()) > 2:
        size = [batch_size, fold_dim, matrix_to_fold.size(-1)]
    else:
        size = [batch_size, fold_dim]
    matrix = matrix_to_fold.new_zeros(size=size)

    # fill matrix


    matrix[idx_0, idx_1] = matrix_to_fold[idx_0, unit_id]

    return matrix
