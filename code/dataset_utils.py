from functools import lru_cache
from tqdm import tqdm
import requests, os
from sklearn.datasets import fetch_openml
from skmultilearn.dataset import load_dataset
import pandas as pd
import numpy as np

def fetch(url,path="dataset"):
    """Downloads the file if it is not already downloaded and returns the path to the downloaded file"""
    
    #create the path if it does not exist
    os.makedirs(path, exist_ok=True)
    
    #get the file name from the url
    file_name = url.split("/")[-1]
    
    #create the file path
    file_path = os.path.join(path, file_name)
    
    #download the file if it does not exist
    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        r = requests.get(url, stream=True)
        assert r.status_code == 200
        progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=url)        
        with open(file_path+".tmp", "wb") as f:
            for x in r.iter_content(chunk_size=16384):
                f.write(x)
                progress_bar.update(len(x))
            
        os.rename(file_path+".tmp", file_path)
        
    return file_path


    
@lru_cache(maxsize=1)    
def fetch_openml_dataset(name="yeast",version=4):
    match name:
        case "birds":
            X, Y = fetch_openml(data_id=41464, return_X_y=True,parser="auto")
        case "yeast":
            X,Y =fetch_openml(name,return_X_y=True,version=version,parser="auto")
        case "reuters":
            X, Y = fetch_openml(
                data_id=41470, return_X_y=True,parser="auto")
        case "emotions":
            X, Y = fetch_openml(
                data_id=41465, return_X_y=True,parser="auto")
        #FIXME: something not right with this dataset
        # case "news20":
        #     X, Y = fetch_openml(
        #         data_id=1594, return_X_y=True,parser="auto")
        case "scene":
            X, Y = fetch_openml(
                data_id=41471, return_X_y=True,parser="auto")
        case "image":
            X, Y = fetch_openml(
                data_id=41468, return_X_y=True,parser="auto")
            
        # FIXME: throws error for this dataset because only one class in a training split
        case "genbase":
            X, Y = fetch_openml(
                data_id=41467, return_X_y=True,parser="auto")
            mappping={"NO":0.0,"YES":1.0}
            X=X.applymap(lambda x: mappping.get(x) if x in mappping else x)
        # FIXME: throws error for this dataset because only one class in a training split
        case "enron": # 53 labels
            X, Y = fetch_openml(
                data_id=41466, return_X_y=True,parser="auto")
        # FIXME: throws error for this dataset because only one class in a training split
        case "langlog":
            X, Y = fetch_openml(
                data_id=41469, return_X_y=True,parser="auto")
        # FIXME: throws error for this dataset because only one class in a training split
        case "slashdot":
            X, Y = fetch_openml(
                data_id=41472, return_X_y=True,parser="auto")
            
        case "tmc2007_500":
            X, Y, feature_names, label_names = load_dataset('tmc2007_500', 'undivided')
            X=pd.DataFrame(X.todense())
            Y=pd.DataFrame(np.array(Y.todense(),dtype=bool))
            return X,Y
        #FIXME: my cache system does not work with that many labels
        case "rcv1subset1":
            X, Y, feature_names, label_names = load_dataset('rcv1subset1', 'undivided')
            X=pd.DataFrame(X.todense())
            Y=pd.DataFrame(np.array(Y.todense(),dtype=bool))
            return X,Y
        
        # FIXME: throws error for this dataset because only one class in a training split
        case "medical":
            X, Y, feature_names, label_names = load_dataset('medical', 'undivided')
            X=pd.DataFrame(X.todense())
            Y=pd.DataFrame(np.array(Y.todense(),dtype=bool))
            return X,Y
        
        case _:
            raise ValueError(f"Dataset {name} not found")
        
    Y = Y == "TRUE"
    return X, Y
            


if __name__ == "__main__":
    # fetch("http://iesl.cs.umass.edu/downloads/bibtex/bibtex.tar.gz")
    X,Y=fetch_openml_dataset("yeast")
    Y = Y == "TRUE"
    print(Y[:10])