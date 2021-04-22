
#basic
import numpy as np
import os
from typing import List, Union
import _pickle as pkl

#sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


#segnlp
from segnlp.features.base import FeatureModel
import segnlp.utils as u
from segnlp.resources.corpus import BNC
from segnlp import get_logger
#from segnlp.preprocessing.resources.vocab import vocab


logger = get_logger("BOW")


# http://www.natcorp.ox.ac.uk/
# https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554
# https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2554/2554.zip?sequence=3&isAllowed=y


def space_tok(string):
    return string.split(" ")


class BOW(FeatureModel):

    def __init__(self, corpus:Union[dict,str]="BNC",  dim:int=200, group:str="doc_embs"):

        #self.vocab = vocab
        self._name = "tfidf"
        self._level = "doc"
        self._feature_dim = dim
        self._dtype = np.float32
        self._group = self._name if group is None else group
        self.__init_model(corpus)


    def __init_model(self, corpus):

        if isinstance(corpus, str):
            corpus_name = corpus.lower()
        else:
            corpus_name = corpus["name"]

        path_to_tfidf_m = f"/tmp/{corpus_name}_tfidf.pkl"
        path_to_svd_m = f"/tmp/{corpus_name}_{self._feature_dim}_svd.pkl"

        train_tfidf = False
        train_svd = False

        if os.path.exists(path_to_tfidf_m):
            logger.info(f"Loading a TF-IDF model from {path_to_tfidf_m}")

            try:
                with open(path_to_tfidf_m, "rb") as f:
                    self._tfidf = pkl.load(f)
            except EOFError as e:
                logger.info(f"Loading TFIDF model from {path_to_tfidf_m} failed: {e}")
                train_tfidf = True
        else:
            train_tfidf = True

        if os.path.exists(path_to_svd_m):

            try:
                logger.info(f"Loading a SVD model from {path_to_svd_m}")
                with open(path_to_svd_m, "rb") as f:
                    self._svd = pkl.load(f)
            except EOFError as e:
                logger.info(f"Loading SVD model from {path_to_svd_m} failed: {e}")
                train_svd = True
        else:
            train_svd = True
            

        if train_tfidf or train_svd:
            logger.info("Will download data and train BOW-tfidf model ... ")
            corpus_data = self.__get_corpus_data(corpus)

            if train_tfidf:
                self._tfidf = self.__train_tfidf(corpus_data, dump_path=path_to_tfidf_m)

            if train_svd:
                self._svd = self.__train_svd(corpus_data, tfidf=self._tfidf, dump_path=path_to_svd_m)

    
    def __get_corpus_data(self, corpus):

        if isinstance(corpus, str):
           
            if corpus.lower() == "bnc":
                bnc = BNC()
                logger.info("Collecting data from BNC-corpus ... (might take some time)")
                corpus_data = [d for d in bnc]
            else:
                raise ValueError(f"{corpus} is not a supported Corpus")
        else:
            corpus_data = cropus["data"]

        return corpus_data


    def __train_tfidf(self, corpus_data, dump_path:str):
        logger.info("Training a TFIDF model ...")
        tfidf = TfidfVectorizer(tokenizer=space_tok, max_df=0.7, min_df=5)
        tfidf.fit(corpus_data)

        with open(dump_path, "wb") as f:
            pkl.dump(tfidf, f)

        return tfidf


    def __train_svd(self, corpus_data, tfidf:int, dump_path:str):
        logger.info("Training a SVD model ...")
        svd = TruncatedSVD(n_components=self._feature_dim, random_state=42)
        X = tfidf.transform(corpus_data)
        svd.fit(X)

        with open(dump_path, "wb") as f:
            pkl.dump(svd, f)
        
        return svd

    #@feature_memory
    def extract(self, df):
        text = " ".join(df["text"].to_numpy())
        x = self._svd.transform(self._tfidf.transform([text]))
        return x[0]

