from xml.parsers.expat import model

import torch
import os
import pickle
from typing import Dict, Tuple, List, Optional
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import re
import numpy as np



""" This function will split the feature names into a list of tokens"""
def split_words(col_name, dt="5G3E"):
    words=[]
    if dt=="cisco":
        for j in re.split('[/ \- __ :]',col_name):
            if j not in words:
                    words.append(j)
    elif dt=="swat":
        for j in re.split('[- ; . ( ) ,]',col_name):
            if j not in words:
                    if j=='':
                        continue
                    else:
                        words.append(j)
    elif dt=="wadi":
        for j in re.split('[- ; . ( ) / ,]',col_name):
            if j not in words:
                    if j=='':
                        continue
                    else:
                        words.append(j)
    else:
        raise ValueError("dataset is not implemented yet")
    
    return words

def split_all(features, dt="5G3E"):
    txt=[]
    for i in range(len(features)):
        a=split_words(features[i], dt=dt)
        txt.append(a)
    return txt

"""This function will find the frequence of each words and give a dictionary with key each words, frequance as value"""
def find_frequence_of_words(list_col, dt=None):
    words_freq={}
    words_dico=[]
    for k in range(len(list_col)):
        if dt=="cisco":
            for j in re.split('[/ \- __ :]',list_col[k]):
                if j not in words_freq:
                    words_freq[j]=1
                    words_dico.append(j)
                else:
                    words_freq[j]+=1
        elif dt=="swat":
            for j in re.split('[- ; . ( ) ,]',list_col[k]):
                if j not in words_freq:
                    words_freq[j]=1
                    words_dico.append(j)
                else:
                    words_freq[j]+=1
        elif dt=="wadi":
            for j in re.split('[- ; . ( ) / ,]',list_col[k]):
                if j not in words_freq:
                    words_freq[j]=1
                    words_dico.append(j)
                else:
                    words_freq[j]+=1          
        else:
            raise ValueError("dataset is not implemented yet")
    
    
    return words_freq, words_dico

"""estimate the total number of term in all features"""
def estimate_total_number_of_word(features, dt=None):
    num=0
    for name, number in find_frequence_of_words(features, dt=dt)[0].items():
        num+=number
    return num

""" brute frequency TF score """
def TF_brute_version(word, freq_each, number_total):
    number_of_this_word=freq_each[word]
    idf2=number_total/number_of_this_word
    return idf2


"""COmpute cosine similarity score"""
def cosine_similarity(seq1, seq2):
    # Convert sequences to numpy arrays
    vec1 = np.array(seq1)
    vec2 = np.array(seq2)
    
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    similarity = dot_product / (mag1 * mag2)
    
    return similarity

#compute similarity score
def score_similarity(columns, model_word2vec, dt=None, verbose=0):

    dico=find_frequence_of_words(columns, dt=dt)[0]
    total_number=estimate_total_number_of_word(columns, dt=dt)

    mat_nlog={}
    vector_names={}
    for coli in range(len(columns)):
        if verbose==1:
            print(f'for feature : {columns[coli]}')

        for colj in range(len(columns)):
            if coli==colj:
                continue
            if verbose==1:
                print(f'for feature : {columns[colj]}')
            words_i=split_words(columns[coli], dt=dt)
            words_j=split_words(columns[colj], dt=dt)

            score_i_3=0
            
            for word1 in words_i:
                 idfi=TF_brute_version(word1,dico,total_number)
                 score_i_3+=model_word2vec.wv[word1]*idfi

            score_i_3=score_i_3/len(words_i)

            score_j_3=0

            for word2 in words_j:   
                idfj=TF_brute_version(word2,dico,total_number)
                score_j_3+=model_word2vec.wv[word2]*idfj

            score_j_3=score_j_3/len(words_j)

            vector_names[columns[colj]]=score_j_3

            similarity_nlog=cosine_similarity(score_i_3, score_j_3)

            mat_nlog[(columns[coli], columns[colj])]= similarity_nlog

    return mat_nlog, vector_names
                    

def load_columns(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() != ""]


def parse_txt_to_dict_swat(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            if not line.strip():
                continue  # skip empty lines
            parts = line.strip().split(' ', 1)  # split only at the first space
            if len(parts) == 2:
                key = int(parts[0])  # assuming the key is always an integer
                value = parts[1]
                data_dict[key] = value
    return data_dict


def get_word_from_dic_swat(dictionary_swat):
    Words=[]
    for k in dictionary_swat.keys():
        features_full_desc=[]
        features_full_desc=re.split("[- ; . ( )]", dictionary_swat[k])
        new_f=[s for s in features_full_desc if s != '']

        #print(new_f)
        Words.append(new_f)
    
    return Words


def load_word2vec_model(path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=True)



def load_or_compute_semantic_dict(
    model_path: str,
    cache_path: Optional[str],
    dataset_name: Optional[str],
    description_file: Optional[str]
) -> Dict[Tuple[str, str], float]:

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)


    # Load pretrained Word2Vec
    model = load_word2vec_model(model_path)


    if dataset_name=="swat":
        if description_file is not None:
            descriptions_dic = parse_txt_to_dict_swat(description_file)
            new_col_names=[]
            for k in descriptions_dic.keys():
                new_col_names.append(descriptions_dic[k])

        else:
            raise ValueError("this dataset require a description file")
        
        #split into words
        words=get_word_from_dic_swat(descriptions_dic)

    if dataset_name == "cisco":
        if description_file is not None:
            new_col_names = load_columns(description_file)
            words = split_all(new_col_names, dt="cisco")
            print(f'len of columns: {len(new_col_names)}')

        else:
            raise ValueError("this dataset require a description file")
    if dataset_name == "wadi":
        if description_file is not None:
            new_col_names = load_columns(description_file)
            words = split_all(new_col_names, dt="wadi")
            print(f'len of columns: {len(new_col_names)}')

        else:
            raise ValueError("this dataset require a description file")

    model_train = Word2Vec(vector_size=300, min_count=1)

    model_train.build_vocab([list(model.key_to_index.keys())], update=False)
    model_train.build_vocab(words, update=True)
    n_total = len(model_train.wv.key_to_index)
    new_vectors = np.random.randn(n_total, 300).astype(np.float32) * 0.01

    for word, new_idx in model_train.wv.key_to_index.items():
        if word in model.key_to_index:
            new_vectors[new_idx] = model[word] 

    model_train.wv.vectors = new_vectors 

    model_train.train(words, total_examples=len(words), epochs=10)


    cosine_similarities, _ =score_similarity(new_col_names, model_train, dt=dataset_name)

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(cosine_similarities, f)

    return cosine_similarities



def compute_correlation_matrix(X: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    corr = (X.T @ X) / (n - 1)
    corr = torch.nan_to_num(corr, nan=0.0)
    return corr


def tensor_corr_to_dict(corr_tensor: torch.Tensor, col_names: list) -> dict:
    a = corr_tensor.numpy().copy()
    
    a[np.tril_indices(a.shape[0], 0)] = np.nan
    
    out = {}
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not np.isnan(a[i, j]):
                out[(col_names[i], col_names[j])] = a[i, j]
    
    return out

"""give a score of proximity between two features, based on name proximity : sim_score, and correlation : corr (COSI-*) """
def build_score_product(corr, sim):
    new_dictionnary={}
    for (name1, name2), score in corr.items():
        new_dictionnary[(name1,name2)]=abs(sim[(name1, name2)]*score)
    
    return new_dictionnary


def get_edges_topK(cosin_score, threshold, topK=200):
    scored_of_all = {}
    for (name1, name2), score in cosin_score.items():
        if score > threshold:
            scored_of_all.setdefault(name1, []).append(score)

    for name, score_list in scored_of_all.items():
        if len(score_list) > topK:
            scored_of_all[name] = sorted(score_list, reverse=True)[:topK]

    edges = []
    for (name1, name2), score in cosin_score.items():
        if score > threshold:
            if name1 in scored_of_all and score in scored_of_all[name1]:
                edges.append((name1, name2))

    return edges



def build_cosi_matrix(
    X: torch.Tensor,
    w2v_path: str,
    device: str = "cpu",
    cache_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    description_file: Optional[str] = None,
    top_k: int = 200
) -> torch.Tensor:

    X = X.to(device)

    sem_dict = load_or_compute_semantic_dict(
        dataset_name=dataset_name,
        model_path=w2v_path,
        cache_path=cache_path,
        description_file=description_file
    )
    col_names = list(dict.fromkeys(name for pair in sem_dict.keys() for name in pair))
    corr = compute_correlation_matrix(X)

    corr_dic=tensor_corr_to_dict(corr, col_names)

    cosi_dict = build_score_product(corr_dic, sem_dict)

    cosi= get_edges_topK(cosi_dict, threshold=0, topK=top_k)

    col_names = list(dict.fromkeys(name for pair in cosi_dict.keys() for name in pair))
    node_to_idx = {name: idx for idx, name in enumerate(col_names)}

    edge_index = torch.tensor(
        [[node_to_idx[n1], node_to_idx[n2]] for n1, n2 in cosi],
        dtype=torch.long
    ).T 

    return edge_index