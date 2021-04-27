#basics
from copy import deepcopy

#pytorch
import torch

#pytroch lightning
from pytorch_lightning.callbacks import ModelCheckpoint


class Evaluation:


    def _normal_eval(
                    self,
                    model_args:dict,
                    ):

        self.dataset.split_id = 0
        self.trainer.fit(    
                        model=deepcopy(self.model)(**model_args), 
                        train_dataloader=self.dataset.train_dataloader(), 
                        val_dataloaders=self.dataset.val_dataloader()
                        )

        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if self.save_choice == "last":
                    model_fp = callback.last_model_path
                    checkpoint_dict = torch.load(model_fp)
                    model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
                else:
                    model_fp = callback.best_model_path
                    model_score = float(checkpoint_cb.best_model_score)
                
        return model_fp, model_score


    def _cv_eval(
                self,
                model_args:dict,
                ):

        cv_scores = []
        model_fps = []
        for i, ids in self.splits.items():
            
            cp_callback = None
            for callback in self.trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    new_filename = callback.filename + f"_fold={i}"
                    setattr(model_ckpt_callback, 'filename', new_filename)
                    cp_callback = callback

            self.dataset.split_id = i
            self.trainer.fit(    
                        model=deepcopy(self.model)(**model_args), 
                        train_dataloader=self.dataset.train_dataloader(), 
                        val_dataloaders=self.dataset.val_dataloader()
                        )

            if self.save_choice == "last":
                model_fp = callback.last_model_path
                checkpoint_dict = torch.load(model_fp)
                model_score = float(checkpoint_dict["callbacks"][ModelCheckpoint]["current_score"])
            else:
                model_fp = callback.best_model_path
                model_score = float(checkpoint_cb.best_model_score)

            cv_scores.append(model_score)
            model_fps.append(model_fp)
        
        mean_score = np.mean(cv_scores)

        return model_fps, mean_score
