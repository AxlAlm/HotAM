

#basics
from typing import Union, List
import numpy as np


#hotam
from hotam.metrics import token_metrics
from hotam.nn import ModelInput
from hotam.utils import ensure_numpy
from hotam.nn.bio_decoder import bio_decode
from hotam.datasets import get_dataset
from hotam.visuals.tree_graph import arrays_to_tree
from hotam.visuals.text_markers import highlight_text

#pytorch
import torch




class ModelOutput:

    def __init__(self, 
                batch:ModelInput,
                return_output:Union[List[int], bool], 
                label_encoders:dict,
                tasks:list,
                all_tasks:list,
                prediction_level:str,
                calc_metrics:bool=True,
                ):

        self.return_output = return_output
        self.calc_metrics = calc_metrics
        self.label_encoders = label_encoders
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.prediction_level = prediction_level
        self.batch = batch
        self._total_loss_added = False

        self.loss = {}
        self.metrics = {}
        self.outputs = {}
        self.pred_spans = {}
        self._pred_span_set = False
        

    def __add_subtask_preds(self, decoded_labels:np.ndarray, lengths:np.ndarray, level:str,  task:str):
        """
        given that some labels are complexed, i.e. 2 plus tasks in one, we can break these apart and 
        get the predictions for each of the task so we can get metric for each of them. 

        for example, if one task is Segmentation+Arugment Component Classification, this task is actually two
        tasks hence we can break the predictions so we cant get scores for each of the tasks.
        """


        #sum_lens = np.sum(lengths)
        max_len = max(lengths)
        size = decoded_labels.shape[0]
        subtasks_predictions = {}
        subtasks = task.split("+")
        for subtask in subtasks:

            subtask_position = subtasks.index(subtask)

            if subtask == "link":
                dtype = np.int16
                decoder = lambda x: int(str(x).split("_")[subtask_position])
            else:
                dtype = decoded_labels.dtype
                decoder = lambda x: str(x).split("_")[subtask_position]


            subtask_preds = np.zeros((size, max_len), dtype=dtype)
            for i in range(size):
                subtask_preds[i][:lengths[i]] = [decoder(x) for x in decoded_labels[i][:lengths[i]] ]


            self.add_preds(
                            task=subtask, 
                            level=level, 
                            data=subtask_preds,
                            decoded=True,
                            )
    

    def __decode_labels(self, preds:np.ndarray, lengths:np.ndarray, task:str):

        size = preds.shape[0]
        decoded_preds = np.zeros((size, max(lengths)), dtype="<U30")
        for i in range(size):
            # print(  
            #         preds[i].shape,
            #         lengths[i],
            #         preds[i][:lengths[i]].shape,
            #         lengths[i],
            #         len(self.label_encoders[task].decode_list(preds[i][:lengths[i]])), 
            #         lengths[i],
            #         decoded_preds[i][:lengths[i]].shape
            #     )
            decoded_preds[i][:lengths[i]] = self.label_encoders[task].decode_list(preds[i][:lengths[i]])
        
        return decoded_preds


    def __decode_token_link_labels(self, preds:np.ndarray, lengths:np.ndarray, span_lengths:np.ndarray, none_spans:np.ndarray):
        
        size = preds.shape[0]
        decoded_preds = np.zeros((size, max(lengths)), dtype=np.int16)
        for i in range(size):
            decoded = self.label_encoders["link"].decode_token_links(
                                                                    preds[i][:lengths[i]], 
                                                                    span_token_lengths=span_lengths[i],
                                                                    none_spans=none_spans[i]
                                                                    )
            decoded_preds[i][:lengths[i]] = decoded
        
        return decoded_preds


    def __handle_complex_tasks(self, data:torch.tensor, level:str, lengths:np.ndarray, task:str):

        # if task is a complex task we get the predictions for each of the subtasks embedded in the 
        # complex task. We will only use these to evaluate
        decoded_labels = self.__decode_labels(
                                                preds=data, 
                                                lengths=lengths,
                                                task=task,
                                                )
        

        subtask_preds = self.__add_subtask_preds(
                                                decoded_labels=decoded_labels,
                                                lengths=lengths,
                                                level=level,
                                                task=task
                                                )
        
        # for stask, sdata in subtask_preds.items():
        #     self.add_preds(
        #                     task=stask, 
        #                     level=level, 
        #                     data=sdata,
        #                     decoded=True,
        #                     )


    def __unfold_span_labels(self, span_labels:np.ndarray, span_indexes:np.ndarray, max_nr_token:int):
        
        batch_size = span_labels.shape[0]
        tok_labels = np.zeros((batch_size, max_nr_token))
        for i in range(batch_size):
            for j,(start,end) in enumerate(span_indexes[i]):
                tok_labels[i][start:end] = span_labels[i][j]
        return tok_labels
        
    
    def __correct_links(self,   links:np.ndarray, 
                                lengths_units:np.ndarray,
                                span_token_lengths:np.ndarray, 
                                none_spans:np.ndarray, 
                                decoded:bool):
        """
        Any link that is outside of the actuall text, e.g. when predicted link > max_idx, is set to predicted_link== max_idx
        https://arxiv.org/pdf/1704.06104.pdf

        """
        new_links = []
        for i in range(links.shape[0]):
            last_unit_idx = max(0, lengths_units[i]-1)
            sample_links = links[i]

            if decoded:
                sample_links = self.label_encoders["link"].encode_token_links(
                                                                                sample_links,
                                                                                span_token_lengths=span_token_lengths[i],
                                                                                none_spans=none_spans[i]
                                                                                )

            
            links_above_allowed = sample_links > last_unit_idx
            sample_links[links_above_allowed] = last_unit_idx

            links_above_allowed = sample_links < 0
            sample_links[links_above_allowed] = last_unit_idx

            sample_links = self.label_encoders["link"].decode_token_links(
                                                                        sample_links,
                                                                        span_token_lengths=span_token_lengths[i],
                                                                        none_spans=none_spans[i]
                                                                        )
            new_links.append(sample_links)
        
        return np.array(new_links)


    def add_loss(self, task:str, data=torch.tensor):

        assert torch.is_tensor(data), f"{task} loss need to be a tensor"
        #assert data.requires_grad, f"{task} loss tensor should require grads"

        if task == "total":
            self.loss["total"] = data
        else:
            self.loss[task] = data

            if not self._total_loss_added:

                if "total" not in self.loss:
                    self.loss["total"] = data
                else:
                    self.loss["total"] += data
    
        self.metrics.update({f"{task}-loss":int(data) if not torch.isnan(data) else 0})


    def add_preds(self, task:str, level:str, data:torch.tensor, decoded:bool=False, sample_ids="same"):

        #assert task in set(self.tasks), f"{task} is not a supported task. Supported tasks are: {self.tasks}"
        assert level in set(["token", "unit"]), f"{level} is not a supported level. Only 'token' or 'unit' are supported levels"
        assert torch.is_tensor(data) or isinstance(data, np.ndarray), f"{task} preds need to be a tensor or numpy.ndarray"
        assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"

        data = ensure_numpy(data)
    
        # If we are prediction on ACs and input is on AC level we can use the information of how many ACs a sample has directly from out input.
        # Same applies if prediction level is token; we get this information explicitly from our preprocessing
        # and same applies if input level i token and prediction level is AC
        # However, if the prediction level is token and the input level is AC, we need to use the prediction lengths derived from our segmentation predictions
        # if level == "unit" and self.prediction_level == "unit":
        #     lengths = self.batch["unit"]["lengths"]

        # elif level == "token" and self.prediction_level == "token":
        #     lengths = self.batch["token"]["lengths"]

        # elif level == "token" and self.prediction_level == "unit":
        #     lengths = self.batch["token"]["lengths"]

        # elif level == "unit" and self.prediction_level == "token":
        #     #lengths = self.seg["lenghts_seq"]
        #     raise NotImplementedError()
            

        if "+" in task:
            self.__handle_complex_tasks(
                                        data=data,
                                        level=level,
                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                        task=task
                                        )
            return


        if level == "unit":
            if self.prediction_level == "unit":
                span_indexes = ensure_numpy(self.batch["unit"]["span_idxs"])
            else:
                raise NotImplementedError()

            # turn labels for each span to labels across all tokens in sample
            data = self.__unfold_span_labels(
                                            span_labels=data,
                                            span_indexes=span_indexes,
                                            max_nr_token=max(ensure_numpy(self.batch["token"]["lengths"])),
                                            )

            if not self._pred_span_set:
                self.pred_spans["lengths"] = self.batch[level]["lengths"]
                self.pred_spans["lengths_tok"] = self.batch["span"]["lengths_tok"]
                self.pred_spans["none_span_mask"] = self.batch["span"]["none_span_mask"]
                self._pred_span_set = True

            level = "token"


        if task == "link":
            data = self.__correct_links(
                                        data,
                                        lengths_units=ensure_numpy(self.pred_spans["lengths"]),
                                        span_token_lengths=ensure_numpy(self.pred_spans["lengths_tok"]),
                                        none_spans=ensure_numpy(self.pred_spans["none_span_mask"]),
                                        decoded=decoded
                                        )


        if not decoded:

            if task == "link" and level == "token":
                decoded_preds = self.__decode_token_link_labels(  
                                                                    preds=data, 
                                                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                                    span_lengths=ensure_numpy(self.pred_spans["lengths_tok"]),
                                                                    none_spans=ensure_numpy(self.pred_spans["none_span_mask"])
                                                                    )
                if self.calc_metrics:
                    decoded_targets = self.__decode_token_link_labels(
                                                                        preds=ensure_numpy(self.batch[level][task]), 
                                                                        lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                                        span_lengths=ensure_numpy(self.batch["span"]["lengths_tok"]),
                                                                        none_spans=ensure_numpy(self.batch["span"]["none_span_mask"])
                                                                        )

            else:
                decoded_preds = self.__decode_labels(
                                                    preds=data, 
                                                    lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                    task=task
                                                    )
                if self.calc_metrics:
                    decoded_targets = self.__decode_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            task=task
                                                            )
        else:
            decoded_preds = data

            if task == "link" and level == "token":
                if self.calc_metrics:
                    decoded_targets = self.__decode_token_link_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            span_lengths=ensure_numpy(self.batch["span"]["lengths_tok"]),
                                                            none_spans=ensure_numpy(self.batch["span"]["none_span_mask"])
                                                            )
            else:
                if self.calc_metrics:
                    decoded_targets = self.__decode_labels(
                                                            preds=ensure_numpy(self.batch[level][task]), 
                                                            lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            task=task
                                                            )


        if task == "seg":
            bio_out = bio_decode(
                                                                batch_encoded_bios=decoded_preds,
                                                                lengths=ensure_numpy(self.batch[level]["lengths"]),
                                                            )
            self.pred_spans["lengths_tok"] = bio_out["span"]["lengths"]
            self.pred_spans["none_span_mask"] = bio_out["span"]["none_span_masks"]
            self.pred_spans["lengths"] = len(bio_out["units"]["lengths"])
        

        if task not in self.outputs:
            self.outputs[task] = []

        self.outputs[task].extend(decoded_preds)


        if self.calc_metrics:
            self.metrics.update(token_metrics(
                                            targets=decoded_targets,
                                            preds=decoded_preds,
                                            mask=ensure_numpy(self.batch["token"]["mask"]),
                                            task=task,
                                            labels=self.label_encoders[task].labels,
                                            )
                                )

   
        
        # if return_output:


        #     if sample_ids == "same":

        #         if isinstance(return_output) == bool:

        #             set(ensure_numpy(self.batch["ids"]).tolist()).union(set(return_output))

        #     else:
        #         raise NotImplementedError()


        # #self.preds[task] = seg_labels


    def add_probs(self, task:str, level:str, data:torch.tensor):

        assert torch.is_tensor(data), f"{task} probs need to be a tensor"
        assert len(data.shape) == 3, f"{task} probs need to be a 3D tensor"
        
        pass
        

    def show_sample(self, prefix=""):
        
        tokens = []
        pred_labels = None
        pred_span_lengths = []
        pred_span_mask = []

        if "label" in self.outputs:
            pred_labels = []

        for idx in  self.batch.oo:
            token_len = self.batch["token"]["lengths"][idx]
            tokens.extend(self.batch[self.prediction_level]["text"][idx][:token_len])
            
            if pred_labels is not None:
                pred_labels.extend(self.outputs["label"][idx][:token_len])

            pred_span_lengths.extend(self.pred_spans["lengths_tok"][idx])
            pred_span_mask.extend(self.pred_spans["none_span_mask"][idx])

        # if "link" in self.outputs:

        #     link_labels = self.outputs.get("link_label", None)
        #     if link_labels is not None:
        #         link_labels = link_labels[idx]


        #     links = self.label_encoders["link"].encode_token_links(
        #                                                                     self.outputs["link"][idx],
        #                                                                     span_token_lengths=self.pred_spans["lengths_tok"][idx],
        #                                                                     none_spans=self.pred_spans["none_span_mask"][idx]
        #                                                                     )

        #     arrays_to_tree(
        #                     self.pred_spans["lengths"][idx], 
        #                     self.pred_spans["lengths_tok"][idx],
        #                     self.pred_spans["none_span_mask"][idx],
        #                     links=links,
        #                     labels=self.outputs["label"][idx],
        #                     tokens=self.batch[self.prediction_level]["text"][idx],
        #                     label_colors=get_dataset("pe").label_colors(),
        #                     link_labels=link_labels,
        #                     )
        
        return highlight_text(
                        tokens=tokens,
                        labels=["Premise", "Claim", "MajorClaim"], 
                        pred_labels=pred_labels,
                        pred_span_lengths=pred_span_lengths,
                        pred_none_spans=pred_span_mask,
                        gold_labels=None,
                        gold_span_lengths=None,
                        gold_none_spans=None,
                        save_path=None, 
                        return_html=True, 
                        prefix=prefix,
                        # show_spans:bool=True, 
                        show_scores=True if pred_labels is not None else False, 
                        # show_legend:bool=True,
                        # font:str="Verdana", 
                        # width:int=1000, 
                        # height:int=800
                        )

    
    # def clear(self):
    #     self.batch = None
    #     self.loss = {}
    #     self.output = {}







    # def __get_segment_labels(labels:list, seg_lengths:list, seg_types:list, max_nr_segs:int, task:str):

    #     nr_samples = labels.shape[0]
    #     seg_labels = torch.zeros(nr_samples, max_nr_segs)
    #     i = 0
    #     for i in range(nr_samples):
    #         floor = 0
    #         type_length = zip(seg_types[i] , seg_lengths[i])
    #         # we dont want segments taht are not Argument Components. These ones we filter out.
    #         sample_seg_lens = [l for t,l in type_length if t is not None] 
    #         nr_segs = len(sample_seg_lens)
    #         for j in range(nr_segs):
                    
    #             seg_labels = labels[i][floor:floor+length]
    #             most_freq_label = Counter(seg_labels).most_common(1)[0][0]

    #             if task == "relation":
                    
    #                 point_to_idx = j + most_freq_label 
    #                 if point_to_idx > nr_segs or point_to_idx < 0:
    #                     most_freq_label = nr_segs - j

    #             seg_labels[i][j] = most_freq_label
    #             floor += length

    #     return seg_labels





    # def __handle_segmentation(self,data);
    #     self.seg["seg_lengths"], self.seg["types"], self.seg["lenghts_seq"] = self.bio.decode(
    #                                                                                         batch_encoded_bios=data, 
    #                                                                                         lengths=self.batch["lengths_tok"]
    #                                                                                         )

    #     # if calc_metrics:
    #     #     seg_results = calc_seg_metrics( 
    #     #                                     target_seg_lens=self.batch["lengths_seq"], 
    #     #                                     pred_seg_lens=self.seg["seg_lengths"]
    #     #                                     )

    #     # self._seg_lens_added = True

    # def add_preds(self, task:str, level:str, data:torch.tensor):

    #     assert task in set(self.dataset.all_tasks), f"{task} is not a supported task. Supported tasks are: {self.dataset.all_tasks}"
    #     assert level in set(["token", "ac"]), f"{level} is not a supported level. Only 'token' or 'ac' are supported levels"
    #     assert torch.is_tensor(data), f"{task} preds need to be a tensor"
    #     assert len(data.shape) == 2, f"{task} preds need to be a 2D tensor"
        
    #     if "seg" in task and not self._seg_lens_added:
    #         self.__handle_segmentation(
    #                                     data=data
    #                                     )
    #         return


    #     # If we are prediction on token level we need to convert the labels to segment labels. For this we need
    #     # the length of each segment predicted by the model.
    #     if level == "token":
    #         if not self._seg_lens_added:
    #             raise RuntimeError("When segmentation is a subtasks it needs to be added first to output")
    
    #     # If we are prediction on ACs and input is on AC level we can use the information of how many ACs a sample has directly from out input.
    #     # Same applies if prediction level is token; we get this information explicitly from our preprocessing
    #     # and same applies if input level i token and prediction level is AC
    #     #
    #     # However, if the prediction level is token and the input level is AC, we need to use the prediction lengths derived from our segmentation predictions
    #     if level == "ac" and self.dataset.prediction_level == "ac":
    #         lengths = self.batch["lengths_seq"]

    #     elif level == "token" and self.dataset.prediction_level == "token":
    #         lengths = self.batch["lengths_tok"]

    #     elif level == "ac" and self.dataset.prediction_level == "token":
    #         lengths = self.seg["lenghts_seq"]

    #     elif level == "token" and self.dataset.prediction_level == "ac":
    #         lengths = self.batch["lengths_tok"]


    #     if "_" in task:
    #         self.__handle_complex_tasks(
    #                                     data=data,
    #                                     level=level,
    #                                     lengths=lengths
    #                                     )
    #         return



    #     if level == "ac":
    #         seg_labels = self.__decode_labels(
    #                                             preds=data, 
    #                                             lengths=lengths
    #                                             )

    #     else:
    #         decoded_labels = self.__decode_labels(
    #                                                 preds=data, 
    #                                                 lengths=lengths
    #                                                 )

    #         seg_labels = self.__get_segment_labels(
    #                                                 labels=decoded_labels, 
    #                                                 seg_lengths=,
    #                                                 seg_types=list,
    #                                                 max_nr_segs=self.dataset.max_nr_segs, 
    #                                                 task=task
    #                                                 )
            
    #     if calc_metrics:

    #         if self.dataset.prediction_level == "ac":
    #             pred_lengths = self.batch["lengths_seq"]
    #         else:
    #             pred_lengths = self.seg["lengths_seq"]

    #         task_metrics = calc_metrics(
    #                                     targets=self.batch["AC_TARGETS FOR TASK"],
    #                                     preds=seg_labels,
    #                                     target_lengths=self.batch["lengths_seq"],
    #                                     pred_lengths=pred_lengths,
    #                                     task=task,
    #                                     prefix=self.current_split,
    #                                     )

    #     self.preds[task] = seg_labels




    # outputs = {}

    # #first we decode the complex tasks, and get the predictions for each of the subtasks
    # complex_tasks = [task for task in self.dataset.tasks if "_" in task]
    # for c_task in complex_tasks:
    #     decoded_labels = decode(preds=preds, lengths=lengths)
    #     subtask_preds = get_subtask_preds(decoded_labels=decoded_labels, task=c_task)
    #     output_dict.update(subtask_preds)


    # for task in self.dataset.subtasks:
        
    #     #if we have the decoded labels
    #     if task not in output_dict:
    #         decoded_labels = outputs[task]
        
    #     #if we dont have the decoded labels
    #     else:
    #         decoded_labels = decode(preds=preds, lengths=lengths)
    #         outputs[task] = decoded_labels


    #     get_segment_labels(decoded_labels, segment_lengths, segment_types, max_nr_segs, task)


    # return outputs
    