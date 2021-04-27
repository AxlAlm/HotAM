
#basics
import numpy as np
import os
import pandas as pd
from typing import List, Dict, Union, Tuple
import re

#pytorch lightning
import pytorch_lightning as ptl

#pytroch
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

#hugginface
from transformers import get_constant_schedule_with_warmup

#am 
from segnlp.utils import ensure_numpy
from segnlp.utils import ensure_flat
from segnlp import get_logger
from segnlp.nn.utils import ModelInput
from segnlp.nn.utils import ModelOutput

logger = get_logger("PTLBase (ptl.LightningModule)")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class MetricContainer(dict):

    def __init__(self):
       
        for k in ["train", "val", "test"]:
            self[k] = []


    def add(self, metrics:dict, split:str):
        """
        metrics = {"metic1": 0.3, "metric2": 0.5, ...., "metricn": 0.9 }
        """
        self[split].append(metrics)

        
    def get_epoch_score(self, split:str):
        epoch_metrics = pd.DataFrame(self[split]).mean()
        
        if epoch_metrics.shape[0] == 0:
            return {}

        epoch_metrics.index = split + "_" + epoch_metrics.index
        return epoch_metrics.to_dict()
   



class PTLBase(ptl.LightningModule):


    def __init__(   self,  
                    hyperparamaters:dict,
                    tasks:list,
                    all_tasks:list,
                    label_encoders:dict,
                    prediction_level:str,
                    task_dims:dict,
                    feature_dims:dict,
                    inference:bool=False
                    ):
        super().__init__()
        self.hps = hyperparamaters
        self.monitor_metric = hyperparamaters.get("monitor_metric", "loss")
        self.feature_dims = feature_dims
        self.task_dims = task_dims
        self.prediction_level = prediction_level
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.label_encoders = label_encoders

        seg_task = [t for t in tasks if "seg" in t][0]
        self.bio_ids = {
                        "B": [i for i,l in label_encoders[seg_task].id2label.items() if "B-" in l],
                        "I": [i for i,l in label_encoders[seg_task].id2label.items() if "I-" in l],
                        "O": [i for i,l in label_encoders[seg_task].id2label.items() if "O-" in l],
                        }
        self.inference = inference
        self.metrics = MetricContainer()
        self.outputs = {"val":[], "test":[]}

        self.OPT = self.hps["optimizer"]
        self.LR = self.hps["lr"]


    def forward(self, batch:ModelInput):
        return self._step(batch, split="test")


    def _step(self, batch:ModelInput, split):
        batch.current_epoch = self.current_epoch
        output = self.forward(
                            batch, 
                            ModelOutput(
                                    batch=batch,
                                    return_output=True, 
                                    label_encoders=self.label_encoders, 
                                    tasks=self.tasks,
                                    all_tasks=self.all_tasks,
                                    prediction_level=self.prediction_level,
                                    inference = self.inference,
                                    )
                            )

        self.metrics.add(output.metrics, split)
        return output.loss.get("total", 0), output
      
    
    def training_step(self, batch, batch_idx):
        loss, output = self._step(batch, "train")
        self.log('train_loss', loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx): 
        loss, output = self._step(batch, "val")
        self.log('val_loss', loss, prog_bar=True)

        if self.monitor_metric != "val_loss":
            self.log(f'val_{self.monitor_metric}', self.metrics["val"][-1][self.monitor_metric.replace("val_","")], prog_bar=True)

        self.outputs["val"].extend(output.to_record())
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        _, output = self._step(batch, "test")
        self.outputs["test"].extend(output.to_record())
        return output


    def _end_of_epoch(self, split):
        #if self.logger is not None:
        epoch_metrics = self.metrics.get_epoch_score(split)

        # if isinstance(self.logger, CometLogger):
        #     for m in epoch_metrics:
        #         if "confusion_matrix" in m:
        #             self.logger.experiment.log_confusion_matrix(labels=["one", "two", "three"],
        #                                                         matrix=[[10, 0, 0],
        #                                                                 [ 0, 9, 1],
        #                                                                 [ 1, 1, 8]])
        self.log_dict(
                        epoch_metrics,
                        on_step=False,
                        on_epoch=True,
                        )
        

    def on_train_epoch_end(self, *args, **kwargs):
        self._end_of_epoch("train")


    def on_validation_epoch_end(self):
        self._end_of_epoch("val")
        self.metrics = MetricContainer()


    def on_test_epoch_end(self):
        self._end_of_epoch("test")


    def configure_optimizers(self):

        if self.OPT.lower() == "adadelta":
            opt = torch.optim.Adadelta(self.parameters(), lr=self.LR)
        elif self.OPT.lower() == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self.LR)
        elif self.OPT.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.LR)
        elif self.OPT.lower() == "rmsprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.LR)
        elif self.OPT.lower() == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.LR)
        else:
            raise KeyError(f'"{self.OPT}" is not a supported optimizer')

        if "scheduler" in self.hps:
            if self.hps["scheduler"].lower() == "rop":
                scheduler = {
                                'scheduler': ReduceLROnPlateau(opt),
                                'monitor': "val_checkpoint_on",
                                'interval': 'epoch',
                                'frequency': 1
                            }
            elif self.hps["scheduler"].lower() == "constant_warmup":
                scheduler = get_constant_schedule_with_warmup(
                                                                optimizer=opt,
                                                                num_warmup_steps=self.hps["num_warmup_steps"],
                                                                last_epoch=self.hps.get("schedular_last_epoch", -1)

                                                                )
            else:
                raise KeyError(f'"{self.hps["scheduler"]} is not a supported learning shedular')

        if "scheduler" in self.hps:
            return [opt], [scheduler]
        else:
            return opt
     








