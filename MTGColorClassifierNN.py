#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:10:40 2022

@author: Grant Bowling

Credit Sryfall database
"""

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os
import re
import shutil
import string

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization


def mod(row):
    if(row['legalities']['modern'] == 'legal'):
        return 1
    else:
        return 0
    
def crea(row):
    if(isinstance(row["type_line"], float)):
        return 0
    elif(row['type_line'].split()[0] == "Creature"):
        return 1
    else:
        return 0
    
def d(num1, num2):
    return num1/num2;

def star(row):
    if('*' in str(row['power']) or '*' in str(row['toughness'])):
        return 1
    else:
        return 0
    
def colorless(row):
    if('W' in row['colors'] or 'U' in row['colors'] or 'B' in row['colors'] or 'R' in row['colors'] or 'G' in row['colors']):
        return 0
    else:
        return 1
    
def compound(row):
    if('//' in row["name"]):
        return 1
    else:
        return 0

def orac_len(row):
    n =  len(row["oracle_text"])
    return n

def has_keyw(row, keyword):
    return keyword in row['keywords']

def com(row):
    if(row['rarity'] == 'common'):
        return True
    else:
        return False
    
def rar(row, rarity):
    if(row['rarity'] == rarity):
        return 1
    else:
        return 0

def col_allow(row, color_num):
    if(len(row['colors']) > color_num):
        return False
    else:
        return True
    
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def make_oracle_dict(row):
    orac_list = row['oracle_text'].split()
    bag = {}
    for word in orac_list:
        if (not word in bag):
            bag[word] = 1
        else:
            bag[word] += 1
    return bag

def oracle_feats(row, abis):
    for ability in abis:
        for abil_word in ability:
            if (not abil_word in row['oracle_text']):
                return 0.0
    return 1.0

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

   
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=2)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    
    origData = pd.read_json('oracle-cards.json')
    df = origData
    
    
    #drop all columns but the columns for basic classifier   
    for col in df.columns:
        if (not col in ["object", "name", "rarity", "oracle_text", "oracle_id", "cmc", "type_line", "power", "toughness", "colors", "keywords", "legalities", "set_name"]):
            df.drop(col, axis = 1, inplace=True)
    
    #check if modern legal and drop illegal cards
    df['is_mod_legal'] = df.apply(mod, axis=1)
    df = df[df.is_mod_legal == 1]
    
    
    #check if creature and drop non-creatures    
    #df['is_creature'] = df.apply(crea, axis=1)
    #df = df[df.is_creature == 1]

    #check compound // name and drop
    df['is_compound'] = df.apply(compound, axis=1)
    df = df[df.is_compound == 0]
    
    #check * power or toughness and drop
    df['is_star'] = df.apply(star, axis=1)
    df = df[df.is_star == 0]
    
    #drop colorless cards
    df['is_colorless'] = df.apply(colorless, axis=1)
    df = df[(df.is_colorless == 0)]
    
    #only use up to certain number of color combos equal to color_num
    df['is_color_allowed'] = df.apply(col_allow, axis=1, color_num=1)
    df = df[df.is_color_allowed == True]
    

    
    #delete the above extra 4 columns used for cleaning
    for col in ["is_mod_legal", "is_star", "is_colorless", "is_compound", 'is_color_allowed']:
        df.drop(col, axis = 1, inplace=True)
    
    
    df['power'] = pd.to_numeric(df['power'])
    df['toughness'] = pd.to_numeric(df['toughness'])
    
    df['pt_ratio'] = df['power']/ df['toughness']
    df['p+t/cmc'] = (df['power'] + df['toughness']) / df['cmc']
    
    df = df[df['oracle_text'] != '']
    
    #clean up the oracle text list for bag-of-words
    df['oracle_text'] = df['oracle_text'].str.lower()
    df['oracle_text'] = df['oracle_text'].str.replace(':','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('.','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace(',','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('—',' ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace(';',' ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('+',' plus ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('-',' minus ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('\n',' ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace("\(.*?\)",'', regex=True)
    
    x_fin = df['oracle_text']
    y = df['colors']

    color_num = {'W' : 0, 'U' : 1, 'B' : 2, 'R' : 3, 'G' : 4} 

    y_num = []
    for i in y:
        y_num.append(color_num[i[0]])
    y_fin = pd.DataFrame(y_num)
    
    max_features = 6000
    sequence_length = 64

    vectorize_layer = layers.TextVectorization(
        standardize= 'strip_punctuation',
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    
    
    ds = tf.data.Dataset.from_tensor_slices((x_fin, y_fin))
    text_train_ds, text_val_ds, text_test_ds = get_dataset_partitions_tf(ds, len(list(ds)), train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000)
    
    # Make a text-only dataset (without labels), then call adapt
    train_text = text_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    train_ds = text_train_ds.map(vectorize_text)
    val_ds = text_val_ds.map(vectorize_text)
    test_ds = text_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim = 14
    
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Conv1D(32, 4, padding="valid", activation="relu", strides=1),
        layers.Dropout(0.25),
        layers.GlobalMaxPooling1D(),   
        layers.Dense(50, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(5,activation='softmax')])

    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, batch_size = 64)

    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)