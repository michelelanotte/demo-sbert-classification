# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:15:07 2021

@author: Utente
"""

import string
import nltk
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
    

STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


def cleaning_URLs(data):
    pattern_re = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
    return re.sub(pattern_re, ' ', str(data))


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return text

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text


#dataset_series è un dato di tipo Series, nonchè un vettore monodimensionale di Pandas con assi etichettati
def preprocessing(dataset_series):
    dataset_series = dataset_series.str.lower() 
    #dataset_series = dataset_series.apply(lambda text: cleaning_stopwords(text))
    #dataset_series = dataset_series.apply(lambda x: cleaning_punctuations(x))
    dataset_series = dataset_series.apply(lambda x: cleaning_repeating_char(x))
    dataset_series = dataset_series.apply(lambda x: cleaning_URLs(x))
    #dataset_series = dataset_series.apply(lambda x: cleaning_numbers(x))
    
    
    """dataset_series = dataset_series.apply(word_tokenize)
    dataset_series = dataset_series.apply(lambda x: stemming_on_text(x))
    dataset_series = dataset_series.apply(lambda x: lemmatizer_on_text(x))
    
    dataset_series = dataset_series.apply(lambda x: " ".join(x))"""
    
    return dataset_series


def model_Evaluate(model, X_test, X_test_vecs, output_file, y_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test_vecs)
    model_name = str(model).split("(")[0]
    
    save_output_tsv(model_name, X_test, y_pred, output_file)
    # Print the evaluation metrics for the dataset.
    #if y_test:
    print(classification_report(y_test, y_pred))
    
    
def csr_matrix_to_arrays(X):
    vectors = []
    for row in X:
        vec_tmp = row.toarray()[0]
        vectors.append(vec_tmp)
    return vectors


def save_output_tsv(model, tweets, predictions, output_file):
    with open(output_file + model + ".tsv", mode = "w", encoding = "ISO-8859-1") as out_file:
        for i, tweet in enumerate(tweets):
            if predictions[i] == 1:
                sentiment = "POSITIVE"
            else: #se la prediction è 0 allora il tweet è negativo
                sentiment = "NEGATIVE"
  
            out_file.writelines(tweet + "\t" + sentiment + "\n")
    