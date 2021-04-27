#/Users/xalmax/phd/datasets/arg-microtexts/corpus/en
import xml.etree.ElementTree as ET
import glob
import re
import os
import pandas as pd
import numpy as np
import shutil

#segnlp
from segnlp.utils import RangeDict
from segnlp.datasets.base import DataSet

#sklearn
from sklearn.model_selection import KFold

#git
from git import Repo

#<?xml version='1.0' encoding='UTF-8'?>
# <arggraph id="micro_b001" topic_id="waste_separation" stance="pro">
#   <edu id="1"><![CDATA[Yes, it's annoying and cumbersome to separate your rubbish properly all the time.]]></edu>
#   <edu id="2"><![CDATA[Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins.]]></edu>
#   <edu id="3"><![CDATA[But still Germany produces way too much rubbish]]></edu>
#   <edu id="4"><![CDATA[and too many resources are lost when what actually should be separated and recycled is burnt.]]></edu>
#   <edu id="5"><![CDATA[We Berliners should take the chance and become pioneers in waste separation!]]></edu>
#   <adu id="1" type="opp"/>
#   <adu id="2" type="opp"/>
#   <adu id="3" type="pro"/>
#   <adu id="4" type="pro"/>
#   <adu id="5" type="pro"/>

#   <edge id="c6" src="1" trg="1" type="seg"/>
#   <edge id="c7" src="2" trg="2" type="seg"/>
#   <edge id="c8" src="3" trg="3" type="seg"/>
#   <edge id="c9" src="4" trg="4" type="seg"/>
#   <edge id="c10" src="5" trg="5" type="seg"/>

#   <edge id="c1" src="1" trg="5" type="reb"/>
#   <edge id="c2" src="2" trg="1" type="sup"/>
#   <edge id="c3" src="3" trg="c1" type="und"/>
#   <edge id="c4" src="4" trg="c3" type="add"/>
# </arggraph>


class MTC(DataSet):

    def __init__(self, 
                tasks:list,
                prediction_level:str="token", 
                sample_level:str="paragraph", 
                dump_path="/tmp/mtc"
                ):
        task_labels = {     
                            "seg":  ["O","B","I"],
                            "label":["None", "pro", "opp"],
                            "link_label": ["None", "sup", "exa", "add", "reb", "und"],
                            "link": list(range(-6,6,1))
                            }

        super().__init__(
                        name = "pe",
                        tasks = tasks,
                        prediction_level = prediction_level,
                        sample_level = sample_level,
                        supported_task_labels = task_labels,
                        level = "paragraph",
                        supported_tasks = [ "seg", "label", "link", "link_label"],
                        supported_prediction_levels = ["unit", "token"],
                        supported_sample_levels = ["paragraph", "sentence"],
                        about = """The arg-microtexts corpus features 112 short argumentative texts. All texts were originally written in German and have been professionally translated to English. """,
                        url = "https://github.com/peldszus/arg-microtexts",                        
                        download_url = "https://github.com/peldszus/arg-microtexts",
                        dump_path = dump_path,
                        )

    @classmethod
    def name(self):
        return "MTC"


    def _download_data(self):

        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
            Repo.clone_from(self.download_url, self.dump_path)
        
        if len(glob.glob(self.dump_path+"/corpus/en/*.xml")) == 0:
            shutil.rmtree(self.dump_path)
            self._download_data()
          
        return os.path.join(self.dump_path,"corpus", "en")


    def _splits(self):
        ### MTC normaly uses Cross Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        ids = np.arange(len(self))
        splits = {i:{"train": train_index,  "test":test_index} for i, (train_index, test_index) in enumerate(kf.split(ids))}
        return splits


    def _process_data(self, path_to_data):
        
        def sort_info(x):
            letter = re.findall(r"(?<=micro_)\w", x.rsplit("/",1)[1])[0]
            nr = int(re.findall(r"(?!0)\d+", x.rsplit("/",1)[1])[0])
            return letter, nr

        xml_files = sorted(glob.glob(path_to_data + "/*.xml"), key=sort_info)
        text_files = sorted(glob.glob(path_to_data + "/*.txt"), key=sort_info)
            
        data = []
        for i in range(len(xml_files)):
            # WE assume that each number in each ID is refering to an unique EDU.
            # for each "id" for any tag type EXCEPT (tag==edge and tag.attrib["type"] == seg),
            # we remove the letter, making all ids numbers. e.g. a1 -> 1, e1 -> 1, c1 -> 1
            # This is bacause the number always refers to a particular EDU EXCEPT in the above mentioned case where 
            # any EDGE ID c{i>max(EDU_ID)} are all of type SEG and only map ei to ai.             
            #
            # follow test will not break
            #
            # if c.tag == "edge":
            #   if c.attrib["type"] == "seg":
            #      assert c.attrib["src"][1:], c.attrib["trg"][1:]
            #   assert c.attrib["id"][1:], c.attrib["src"][1:]
            
            file_id = int(re.findall(r"(?!0)\d+", xml_files[i])[0])
                
            edus = {}
            
            
            with open(text_files[i]) as f:
                text = f.read()
                
            tree = ET.parse(xml_files[i])
            root = tree.getroot()
            
            start = 0
            for i,c in enumerate(root):
                
                if c.tag == "edu":
                    ID = c.attrib["id"][1:]
                    text_len = len(c.text)
                    edus[ID] = {
                                        "label": None, 
                                        "link":int(ID)-1, 
                                        "link_label":None,
                                        "span_id": int(ID)-1,
                                        "unit_id": None,
                                        "span": (start, start+text_len)
                                        }
                    
                    start += (text_len+1)
                
                elif c.tag == "adu":
                    ID = c.attrib["id"][1:]
                    edus[ID]["label"] = c.attrib["type"]
                    edus[ID]["unit_id"] = int(ID)-1
                                        
                elif c.tag == "edge":
                    link_label = c.attrib["type"]
                    
                    if c.attrib["type"] == "seg":
                        assert c.attrib["src"][1:], c.attrib["trg"][1:]
                        continue
                    
                    assert c.attrib["id"][1:], c.attrib["src"][1:]
                
                    ID = c.attrib["id"][1:]
                    edus[ID]["link_label"] = c.attrib["type"]

                    link = (int(c.attrib["trg"][1:])-1) - (int(ID)-1)
                    edus[ID]["link"] = link

            
            assert len(text) == start-1, f"text={len(text)},  start={start}"
            
            span_labels = RangeDict({e.pop("span"):e for _,e in edus.items()})

            data.append({
                        "sample_id":file_id,
                        "text":text, 
                        "text_type":"document",
                        "span_labels": span_labels
                        })

        return pd.DataFrame(data)