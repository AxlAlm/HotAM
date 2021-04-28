#basics
import re 
import torch
import numpy as np

#segnlp
from segnlp.utils import ensure_numpy
from segnlp.utils import ensure_flat
from segnlp.nn.utils import create_mask


class BIODecoder:

    def __init__(self, 
                B:list=["B"],
                I:list=["I"],
                O:list=["O"],
                apply_correction:bool = True
                ):

        Bs = "-|".join([str(b) for b in B]) + "-"
        Is = "-|".join([str(i) for i in I]) + "-"
        Os = "-|".join([str(o) for o in O]) + "-"
        
        # if we have invalid BIO structure we can correct these
        #  https://arxiv.org/pdf/1704.06104.pdf, appendix
        # 1) I follows 0 -> allow OI to be interpreted as a B
        if apply_correction:
            Bs += f"|(?<=({Os}))({Is})|<START>-"

        #self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")
        self.pattern = re.compile(f"(?P<UNIT>({Bs})({Is})*)|(?P<NONE>({Os})+)")
        

    def __call__(
                self,
                batch_encoded_bios:np.ndarray, 
                lengths:np.ndarray, 
                ):
    
        batch_size = batch_encoded_bios.shape[0]

        bio_data = {
                    "span":{
                            "lengths_tok":[],
                            "lengths": [],
                            "span_idxs": [],
                            "none_span_mask":[],
                            "start":[],
                            "end":[],
                            },
                    "unit":{
                            "lengths_tok":[],
                            "lengths":[],
                            "span_idxs": [],
                            "none_span_mask":[],
                            "start":[],
                            "end":[],
                            },
                    "max_units":0
                    }


        for i in range(batch_size):
            span_lengths, none_span_mask = self.__decode_sample(
                                                                encoded_bios=batch_encoded_bios[i][:lengths[i]]
                                                                )

            span_ends = np.cumsum(span_lengths)
            span_starts = np.insert(span_ends,0,0)[:-1]
            span_indexes = np.stack((span_starts, span_ends), axis=-1)
            unit_indexes = span_indexes[none_span_mask]
            unit_lengths = span_lengths[none_span_mask]

            bio_data["span"]["lengths"].append(len(span_lengths))
            bio_data["span"]["lengths_tok"].append(span_lengths.tolist())

            bio_data["span"]["none_span_mask"].append(none_span_mask.tolist())
            bio_data["span"]["span_idxs"].append(span_indexes.tolist())
            bio_data["span"]["start"].append(span_starts.tolist())
            bio_data["span"]["end"].append(span_ends.tolist())
            
            unit_length = sum(none_span_mask)
            bio_data["unit"]["lengths"].append(unit_length)
            bio_data["unit"]["lengths_tok"].append(span_lengths[none_span_mask].tolist())
            bio_data["unit"]["span_idxs"].append(span_indexes[none_span_mask].tolist())
            bio_data["unit"]["start"].append(span_starts[none_span_mask].tolist())
            bio_data["unit"]["end"].append(span_ends[none_span_mask].tolist())

            bio_data["max_units"] = max(unit_length, bio_data["max_units"])

        bio_data["unit"]["mask"] = create_mask(bio_data["unit"]["lengths"])
  
        return bio_data


    def __decode_sample(self, encoded_bios):

        encoded_bios = ensure_numpy(encoded_bios)
        encoded_bios_str = "<START>-" + "-".join(encoded_bios.astype(str)) + "-"
        all_matches = re.finditer(self.pattern, encoded_bios_str)

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