#basics
from glob import glob
import os
import json
import re
import pandas as pd
from pathlib import Path

#hotam
from hotam import get_logger


logger = get_logger("LocalLogger")


class LocalDB:

    """
    structure

    root_dir/
        experiments/
            <exp_id>_<project>_<dataset>_<model>_<timestamp>/
                config.json
                scores/
                    val/
                        epoch=1.json
                        epoch=n.json
                        ..
                    train/
                        epoch=n.json
                outputs/
                    epoch=1.json
                    epoch=2.json
                    ..


    """

def read_json(fp):
    with open(fp, "r") as f:
        content = json.load(f)
    return content


class LocalDB:


    def __init__(self, root_dir=os.path.join(Path.home(),".hotam/")):
        self.root_dir = root_dir
        logger.info(f"Docking dataset in at {root_dir}")
        self.experiment_dir = os.path.join(self.root_dir, "experiments")


    def get_last_exp(self):
        all_exp_folders = glob(self.experiment_dir+"/*")
        last_exp_folder = sorted(all_exp_folders, key=lambda x: int(x.split("_")[-1]))[-1]
        last_exp_config = os.path.join(last_exp_folder, "config.json")
        with open(last_exp_config, "r") as f:
            config = json.load(f)

        return config


    def get_last_epoch(self, experiment_id):
        try:
            exp_folder = glob(os.path.join(self.experiment_dir, experiment_id+"*"))[0]

            train_scores  = os.path.join(exp_folder, "scores", "train")
            val_scores  = os.path.join(exp_folder, "scores", "val")
            get_epoch = lambda x:int(re.sub(r"[^\d+]", x, ""))
            last_epoch = max(
                                sorted([get_epoch(x) for x in train_scores])[-1],
                                sorted([get_epoch(x) for x in train_scores])[-1]
                                )

            return last_epoch
        except IndexError:
            return -1


    def get_scores(self, experiment_ids:list, epoch:int = None):
        
        scores = []
        #make this into a regex search instead?
        for exp_id in experiment_ids:
            score_files = glob(self.experiment_dir +f"/{exp_id}*/scores**.json")
            scores.extend(read_json(fp) for fp in score_files)

        return scores
    

    def get_outputs(self, experiment_ids):

        outputs = []
        #make this into a regex search instead?
        for exp_id in experiment_ids:
            output_files = glob(self.experiment_dir+f"/{exp_id}*/outputs**.json")
            outputs.extend(read_json(fp) for fp in output_files)

        return {} if output is None else output
    

    def get_exp_config(self, experiment_id):
        config_file = glob(self.experiment_dir + f"/{experiment_id}*/config.json")[0]
        config = read_json(config_file)
        return config


    def get_exp_configs(self, dataset="*", project="*", model="*"):
        configs = [read_json(fp)  for fp in glob(self.experiment_dir + f"/*_{project}_{dataset}_{model}*/config.json")]
        return configs


    def get_exp_ids(self):
        #print(self.experiment_dir)
        #s = os.path.join(self.experiment_dir, "*"))
        exp_ids = [f.rsplit("/",1)[-1].split("_")[0]  for f  in glob(self.experiment_dir+"/*")]
        return exp_ids
    

    def get_projects(self):
        projects = [f.rsplit("/",1)[-1].split("_")[1]  for f  in glob(self.experiment_dir+"/*")]
        return projects

    
    def get_project_tasks(self, project):
        exp_configs = self.get_exp_configs(project=project)
        tasks = sorted(set([t for exp in exp_configs for t in exp["tasks"]]))
        return tasks
    
    def get_project_subtasks(self, project):
        exp_configs = self.get_exp_configs(project=project)
        subtasks = sorted(set([t for exp in exp_configs for t in exp["subtasks"]]))
        return subtasks