#basics
import re

#segnlp
from segnlp.preprocessing.dataset_preprocessor import PreProcessedDataset
from segnlp.ptl import PTLBase
from segnlp.ptl import setup_ptl_trainer

#pytorch
from pytorch_lightning.callbacks import ModelCheckpoint

#pytorch
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


#pytorch lightning
from pytorch_lightning import Trainer 


def default(
            model_args:dict,
            trainer:Trainer,
            dataset:PreProcessedDataset,
            save_choice:str,
            ):


    dataset.split_id = 0

    ptl_model = PTLBase(**model_args)
    trainer.fit(    
                    model=ptl_model, 
                    train_dataloader=dataset.train_dataloader(), 
                    val_dataloaders=dataset.val_dataloader()
                    )

    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            # if save_choice == "last":
            #     model_fp = callback.last_model_path
            #     checkpoint_dict = torch.load(model_fp)

            #     print(checkpoint_dict["callbacks"][ModelCheckpoint])
            #     print(lol)

            #     model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
            # else:
            model_fp = callback.best_model_path
            #re.findall(r"(?<=val_f1=)\d+.\d+(?=.ckpt)", model_fp)
            model_score = float(callback.best_model_score)
        
    return model_fp, model_score
