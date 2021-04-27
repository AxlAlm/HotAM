#basic
import re
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import random
import os
from urllib.request import urlopen
from zipfile import ZipFile
from zipfile import BadZipFile

import time
from pathlib import Path
from typing import Tuple, List, Dict

# segnlp
from segnlp import get_logger
import segnlp.utils as u
from segnlp.utils import RangeDict
from segnlp.datasets.base import DataSet

#
# string
from string import punctuation
punctuation += "’‘,`'" + '"'

logger = get_logger(__name__)

#data example
''' 

Data example

T1  MajorClaim 503 575  we should attach more importance to cooperation during primary education
T2  MajorClaim 2154 2231    a more cooperative attitudes towards life is more profitable in one's success
T3  Claim 591 714   through cooperation, children can learn about interpersonal skills which are significant in the future life of all students
A1  Stance T3 For
T4  Premise 716 851 What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others
T5  Premise 853 1086    During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred
T6  Premise 1088 1191   All of these skills help them to get on well with other people and will benefit them for the whole life
R1  supports Arg1:T4 Arg2:T3    
R2  supports Arg1:T5 Arg2:T3    
R3  supports Arg1:T6 Arg2:T3    
T7  Claim 1332 1376 competition makes the society more effective
A2  Stance T7 Against
T8  Premise 1212 1301   the significance of competition is that how to become more excellence to gain the victory
T9  Premise 1387 1492   when we consider about the question that how to win the game, we always find that we need the cooperation
T10 Premise 1549 1846   Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could win the game without the training of his or her coach, and the help of other professional staffs such as the people who take care of his diet, and those who are in charge of the medical care
T11 Claim 1927 1992 without the cooperation, there would be no victory of competition
A3  Stance T11 For
R4  supports Arg1:T10 Arg2:T11  
R5  supports Arg1:T9 Arg2:T11   
R6  supports Arg1:T8 Arg2:T7


Dataset Specs:

1) contain 4 subtasks:
    seg
    ac
    relation
    stance

2) Each claim is related to the major claims. MajorClaims are paraphrases of the same majorClaim

3) All relations are within paragraphs


''' 


class PE(DataSet):

    """Class for downloading, reading and parsing the Pursuasive Essay dataset found here: 

    https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays/index.en.jsp

    Does not parse the dataset from the available CONLL format as it contain errors, instead its parsed
    directly from the original dataset.

    """

    def __init__(self,
                tasks:list,
                prediction_level:str="token", 
                sample_level:str="document", 
                dump_path:str="/tmp/"
                ):
        task_labels = {
                            "seg": ["O","B","I"],
                            "label":["None", "MajorClaim", "Claim", "Premise"],
                            # Originally stance labels are For and Against for Claims and MajorClaims
                            # and for premsies supports or attacks. 
                            # However, Against and attacks are functional equivalent so will use CON for both
                            # and for For and supports we will use PRO
                            #"stance":["For", "Against", "supports", "attacks"],
                            "link_label": ["support", "attack", "None"],
                            "link": list(range(-11,12,1))
                            }

        self._stance2new_stance = {
                                    "supports": "support", 
                                    "For": "support", 
                                    "Against": "attack", 
                                    "attacks": "attack",
                                    }
        super().__init__(
                        name="pe",
                        tasks=tasks,
                        prediction_level=prediction_level,
                        sample_level=sample_level,
                        supported_task_labels=task_labels,
                        level="document",
                        supported_tasks=["seg", "label", "link", "link_label"],
                        supported_prediction_levels=["unit", "token"],
                        supported_sample_levels=["document", "paragraph", "sentence"],
                        about="""The corpus consists of argument annotated persuasive essays including annotations of argument components and argumentative relations.""",
                        url="https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp",
                        download_url= "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2422/ArgumentAnnotatedEssays-2.0.zip?sequence=1&isAllowed=y",
                        dump_path=dump_path,
                        )


    @classmethod
    def name(self):
        return "PE"

    def _download_data(self, force=False) -> str:
        """downloads the data from sourse website

        Returns
        -------
        str
            path to downloaded data
        """
    
        zip_dump_path = "/tmp/ArgumentAnnotatedEssays-2.0.zip"
        parent_folder = os.path.join(self.dump_path, "ArgumentAnnotatedEssays-2.0")
        data_folder = os.path.join(parent_folder, "brat-project-final")
        if os.path.exists(data_folder) and not force:
            return data_folder

        zip_dump_path = "/tmp/ArgumentAnnotatedEssays-2.0.zip"
        
        if not os.path.exists(zip_dump_path):
            desc = f"Downloading ArgumentAnnotatedEssays-2.0"
            u.download(url=self.download_url, save_path=zip_dump_path, desc=desc)

        u.unzip(zip_dump_path, self.dump_path)
        u.unzip(data_folder + ".zip", parent_folder)

        return data_folder


    def __read_file(self,file:str) -> str:
        """read data file

        Parameters
        ----------
        file : str
            path do data

        Returns
        -------
        str
            text of file
        """
        with open(file,"r", encoding="utf8") as f:
            text = f.read()
        return text


    def __get_ac(self, annotation_line:str) -> Tuple[str, int, int]:
        """extracts the Argument Unit label and the start and end character position of the span

        Parameters
        ----------
        annotation_line : str
            line of annotation

        Returns
        -------
        Tuple[str, int, int]
            Label, start of span, end of span
        """
        #e.g. "MajorClaim 238 307"
        AC, start_char, end_char = annotation_line.split()
        return AC, int(start_char), int(end_char)


    def __get_stance(self, annotation_line:str) -> Tuple[str, str]:
        """extracts the stance and the id for the Argument Unit the stance belong to.

        Parameters
        ----------
        annotation_line : str
            line of annotation

        Returns
        -------
        Tuple[str, str]
            label, id of Argument Unit/component
        """
        #e.g. "Stance T8 For"
        _, AC_ID, stance = annotation_line.split()
        return stance, AC_ID


    def __get_relation(self, annotation_line:str) -> Tuple[str, str, str]:
        """extracts the relation label and the id for the argument units tied in the relation


        Parameters
        ----------
        annotation_line : str
            line of annotation

        Returns
        -------
        Tuple[str, str, str]
            difference in argument components/units back or forward( + or -), label, ID of argument component that
            supports/attacks/for/against 
        """
        #e.g. supports Arg1:T10 Arg2:T11 
        stance, arg1, arg2 = annotation_line.split()

        arg1_AC_ID = arg1.split(":")[1]
        arg2_AC_ID = arg2.split(":")[1] 

        #get difference in number of AC's 
        #diff_in_acs = int(arg2_AC_ID[1:]) - int(arg1_AC_ID[1:]) 

        #return str(diff_in_acs), stance, arg1_AC_ID
        return stance, arg1_AC_ID, arg2_AC_ID


    def __parse_ann_file(self, ann:str, text_len:int) -> Tuple[dict, dict, dict, dict]:
        """parses the annotation file for each Essay

        creates a dicts for mapping character spans to Argument Component/unit ids
        then dicts for mapping Argument Components to the different labels

        Parameters
        ----------
        ann : str
            string of annotations (see above (line 38) for example)
        text_len : int
            length of the essay

        Returns
        -------
        List[dict, dict, dict]
            dict mapping character span to Argument Components
            dicts for mapping Argument Components to labels
        """
        ac_id2span = {}

        ac_id2ac = {}
        ac_id2stance = {}
        ac_id2relation = {}

        ann_lines = ann.split("\n")
        for ann_line in ann_lines:
            # There are 3 types of lines:
            #['T1', 'MajorClaim 238 307', 'both of studying hard and playing sports are part of life to children']
            #["R5", "supports Arg1:T9 Arg2:T11 "  ]
            #["A3", "Stance T11 For"]

            if not ann_line.strip():
                continue

            segment_ID, annotation_line, *_ = ann_line.split("\t")

            # for MajorClaim, Premise, Claim
            if segment_ID.startswith("T"):
                AC_ID = segment_ID
                ac, start_idx, end_idx = self.__get_ac(annotation_line)
                ac_id2span[AC_ID] = (start_idx,end_idx)
                ac_id2ac[AC_ID] = ac

            #for Stance
            elif segment_ID.startswith("A"):
                stance, AC_ID = self.__get_stance(annotation_line)
                ac_id2stance[AC_ID] = stance

            # for relations + stance
            else:
                #relation, stance, AC_ID = self.__get_relation(annotation_line)
                stance, AC_ID, AC_REL_ID = self.__get_relation(annotation_line)
                ac_id2stance[AC_ID] = stance
                ac_id2relation[AC_ID] = AC_REL_ID

        # sort the span
        ac_id2span_storted = sorted(ac_id2span.items(), key= lambda x:x[1][0])
        ac_id2idx = {AC_ID:i for i, (AC_ID, *_) in enumerate(ac_id2span_storted)}
        ac_id2relation = {AC_ID: ac_id2idx[AC_REL_ID] - ac_id2idx[AC_ID]  for AC_ID, AC_REL_ID in ac_id2relation.items()}
        
        #self.task_labels.extend(list(ac_id2relation.values()))

        # fill in some spans
        prev_span_end = 0
        for (_, (start,end)) in ac_id2span_storted.copy():

            if start-1 != prev_span_end:
                ac_id2span_storted.append((f"None_{start-1}",(prev_span_end,start-1)))

            prev_span_end = end

        #add last Dummy span
        if prev_span_end != text_len:
            ac_id2span_storted.append(("None_LAST",(prev_span_end,999999)))

        # sort again when added the missing spans
        ac_id2span = dict(sorted(ac_id2span_storted,key=lambda x:x[1][0]))
        
        return ac_id2span, ac_id2ac, ac_id2stance, ac_id2relation

    
    def __get_label_dict(self, 
                        ac_id2span:Dict[str,Tuple[int,int]], 
                        ac_id2ac:Dict[str,str], 
                        ac_id2stance:Dict[str,str], 
                        ac_id2relation:Dict[str,str]) -> RangeDict:
        """creates a unified dict for label spans in the essay. Dict constructed contains the span tuple as a key
        and dict of all labels as value. The dict, a RangeDict, fetches the value of the span the key is within.

        e.g. if spans are 0-10, 10-50, 50-65, and key is 5, labels of span 0-10 will be returned

        Parameters
        ----------
        ac_id2span : Dict[str,Tuple[int,int]]
            dict of AC id to character span
        ac_id2ac : Dict[str,str]
            AC id to argument component label
        ac_id2stance : Dict[str,str]
            AC id to stance label
        ac_id2relation : Dict[str,str]
            AC id to relation label

        Returns
        -------
        RangeDict
            dict of characters spans to labels
        """

        span2label = RangeDict()
        current_unit_id = 0
        for i, (ac_id, span) in enumerate(ac_id2span.items()):

            relation = ac_id2relation.get(ac_id, 0)
            #self.__task_labels["link"].add(relation)


            ac = ac_id2ac.get(ac_id,"None")
            stance = self._stance2new_stance.get(ac_id2stance.get(ac_id,"None"), "None")

            if "None" in ac_id:
                unit_id = np.nan 
            else:
                unit_id = current_unit_id
                current_unit_id += 1

            label = {   
                        "label": ac_id2ac.get(ac_id, "None"), 
                        "link_label": self._stance2new_stance.get(ac_id2stance.get(ac_id,"None"), "None"), 
                        "link": relation,
                        "span_id": i,
                        "unit_id": unit_id,
                    }
            

            span2label[span] = label #[label, ac_id]

        return span2label


    def _splits(self) -> Dict[int, Dict[str, np.ndarray]]:
        """creates a dict of split ids from the premade splits

        Returns
        -------
        Dict[int, Dict[str, np.ndarray]]
            a dict of split ids. First level is for CV splits, if one want to split multiple times.
        """

    	# NOTE! we are provided with a test and train split but not a dev split
    	# approach after the original paper:
    	# https://arxiv.org/pdf/1612.08994.pdf (section 4) - join pointer model
    	# https://www.aclweb.org/anthology/P19-1464.pdf (section 4.1 -Dataset) - LSTM-Dist
    	# report that they randomly select 10% from the train set as validation set
    	# there is no reference a dev or validation split in 
    	# https://arxiv.org/pdf/1704.06104.pdf - (LSTM-ER, LSTM-CNN-CRF)
    	# 
    	# For only segmentation:
    	# https://www.aclweb.org/anthology/W19-4501.pdf (LSTM-CRF + flair, bert etc),
    	# report that they use the following samples for dev set:
    	#
    	# 13, 38, 41, 115, 140, 152, 156, 159, 162, 164, 201, 257,291, 324, 343, 361, 369, 371, 387, 389, 400 
    	# 
    	# however 21/322 = 6%, thus using smaller dev set
    	#
    	# We will randomly select a 10% as dev set

       	train_set = []
       	test_set = []

        try:
            split_path = str(list(Path(self.dump_path).rglob("train-test-split.csv"))[0])
        except IndexError as e:
            self._dataset_path = self._download_data(force=True)
            split_path = str(list(Path(self.dump_path).rglob("train-test-split.csv"))[0])


       	with open(split_path, "r") as f:
       		for i,line in enumerate(f):

       			if not i:
       				continue

       			essay, split = line.replace('"',"").strip().split(";")
       			essay_id = int(re.findall(r"(?!0)\d+", essay)[0])

       			if split == "TRAIN":
       				train_set.append(essay_id)
       			else:
       				test_set.append(essay_id)

        #dev_set = [13, 38, 41, 115, 140, 152, 156, 159, 162, 164, 201, 257,291, 324, 343, 361, 369, 371, 387, 389, 400]
        dev_set = []

        if dev_set:
            for i in dev_set:
                train_set.remove(i)
        else:
            dev_size = int(len(train_set)*0.1)
            dev_set = []
            while len(dev_set) != dev_size:
                i = random.choice(train_set)
                train_set.remove(i)
                dev_set.append(i)

        random.shuffle(train_set)
        random.shuffle(dev_set)
        random.shuffle(test_set)

        # as pe start on 1 we shift all ids so it start at 0
        splits = {
                    0:{
                        "train":np.array(train_set)-1, 
                        "val":  np.array(dev_set)-1,
                        "test": np.array(test_set)-1
                    }
                }
       	return splits


    def _process_data(self, path_to_data):
        """loads the Pursuasive Essay data and parse it into a DataSet. Also dumps the dataset 
        locally so that one does not need to parse it again, only load the parsed data.

        if the dumppath exist data will be loaded from the pkl file and no parsing will be done

        Returns
        -------
        DataSet
            
        """
        ann_files = sorted(glob(path_to_data+"/*.ann"))
        text_files = sorted(glob(path_to_data+"/*.txt"))
        number_files = len(ann_files) + len(text_files)

        data = []
        grouped_files = list(zip(ann_files, text_files))
        for ann_file, txt_file in grouped_files:

            # -1 one as we want index 0 to be sample 1
            file_id = int(re.sub(r'[^0-9]', "", ann_file.rsplit("/",1)[-1])) -1

            text = self.__read_file(txt_file)
            ann = self.__read_file(ann_file)

            # extract annotation spans
            ac_id2span, ac_id2ac,ac_id2stance, ac_id2relation = self.__parse_ann_file(ann, len(text))

            span2label = self.__get_label_dict(ac_id2span, ac_id2ac,ac_id2stance, ac_id2relation)
                        
            data.append({
                            "sample_id":file_id,
                            "text":text, 
                            "text_type":"document",
                            "span_labels": span2label
                            })
 
        return pd.DataFrame(data)
        
    
    @classmethod
    def label_colors(self):
        return {
                "PRO":"#14ac1f",
                "CON":"#ff0000",
                "MajorClaim": "#83bde4",
                "Claim": "#94edaf",
                "Premise": "#f1a8ee",
                }