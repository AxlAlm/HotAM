
from hotam.utils import ensure_numpy, ensure_flat

import re 
import torch
import numpy as np
#torch.nn.Module

# class BIO_Decoder():
#     def __init__(self, B:list, I:list, O:list, apply_correction:bool=True):
#         Bs = "-|".join([str(b) for b in B]) + "-"
#         Is = "-|".join([str(i) for i in I]) + "-"
#         Os = "-|".join([str(o) for o in O]) + "-"
#         self._Os = O
#         self._apply_correction = apply_correction

#         # if we have invalid BIO structure we can correct these
#         #  https://arxiv.org/pdf/1704.06104.pdf, appendix
#         # 1) I follows 0 -> allow OI to be interpreted as a B
#         if self._apply_correction:
#             Bs += f"|(?<=({Os}))({Is})"

#         #self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")
#         self.pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
        

def bio_decode(
                batch_encoded_bios:np.ndarray, 
                lengths:np.ndarray, 
                apply_correction=True,
                B:list=["B"],
                I:list=["I"],
                O:list=["O"],
                only_units:bool = False
                ):
        
    def bio_decode_sample(pattern, encoded_bios):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "<START>-" + "-".join(encoded_bios.astype(str)) + "-"
        all_matches = re.finditer(pattern, encoded_bios_str)

        none_span_mask = []
        span_lengths = []
        for m in all_matches:

            match_string = m.group(0).replace("<START>-", "")
            length = len(match_string.split("-")[:-1])

            if length == 0:
                continue

            span_type = 0 if m.groupdict()["UNIT"] is None else 1

            span_lengths.append(length)
            none_span_mask.append(span_type)
            

        none_span_mask = np.array(none_span_mask, dtype=bool)
        span_lengths = np.array(span_lengths)

        return span_lengths, none_span_mask


    Bs = "-|".join([str(b) for b in B]) + "-"
    Is = "-|".join([str(i) for i in I]) + "-"
    Os = "-|".join([str(o) for o in O]) + "-"
    
    # if we have invalid BIO structure we can correct these
    #  https://arxiv.org/pdf/1704.06104.pdf, appendix
    # 1) I follows 0 -> allow OI to be interpreted as a B
    if apply_correction:
        Bs += f"|(?<=({Os}))({Is})|<START>-"

    #self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")
    pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
    
    batch_size = batch_encoded_bios.shape[0]

    bio_data = {
                "span":{
                        "lengths":[],
                        "span_idxs": [],
                        "none_span_masks":[],
                        "start":[],
                        "end":[],
                        },
                "unit":{
                        "lengths":[],
                        "span_idxs": [],
                        "none_span_masks":[],
                        "start":[],
                        "end":[],
                        },
                "max_units":0
                }
    # sample = [2,3,10,6] where each number indicate the lenght of a spans
    all_span_lengths = []

    # sample = [0,1,0,1,0,1,1] mask which tells us which spans are unit and whihc are not
    all_none_span_mask = []

    # sample = 3, nr of predicted units in the sample
    unit_lengths = []

    #sample = [(0,10),(11,30 ...] 
    all_span_indexes = []

    for i in range(batch_size):
        span_lengths, none_span_mask = bio_decode_sample(
                                                        pattern,
                                                        encoded_bios=batch_encoded_bios[i][:lengths[i]]
                                                        )

        span_ends = np.cumsum(span_lengths)
        span_starts = np.insert(span_ends,0,0)[:-1]
        span_indexes = np.stack((span_starts, span_ends), axis=-1)
        unit_indexes = span_indexes[none_span_mask]
        unit_lengths = span_lengths[none_span_mask]

        bio_data["span"]["lengths"].append(span_lengths.tolist())
        bio_data["span"]["none_span_masks"].append(none_span_mask.tolist())
        bio_data["span"]["span_idxs"].append(span_indexes.tolist())
        bio_data["span"]["start"].append(span_starts.tolist())
        bio_data["span"]["end"].append(span_ends.tolist())
        
        
        bio_data["unit"]["lengths"].append(span_lengths[none_span_mask].tolist())
        bio_data["unit"]["span_idxs"].append(span_indexes[none_span_mask].tolist())
        bio_data["unit"]["start"].append(span_starts[none_span_mask].tolist())
        bio_data["unit"]["end"].append(span_ends[none_span_mask].tolist())

        bio_data["max_units"] = max(sum(none_span_mask), bio_data["max_units"])

    return bio_data
