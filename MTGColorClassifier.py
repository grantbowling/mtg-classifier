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
from sklearn.datasets import make_classification

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
        

if __name__ == '__main__':
    
    origData = pd.read_json("oracle-cards.json")
    df = origData
    
    
    #drop all columns but the columns for basic classifier   
    for col in df.columns:
        if (not col in ["object", "name", "rarity", "oracle_text", "oracle_id", "cmc", "type_line", "power", "toughness", "colors", "keywords", "legalities", "set_name"]):
            df.drop(col, axis = 1, inplace=True)
    
    #check if modern legal and drop illegal cards
    df['is_mod_legal'] = df.apply(mod, axis=1)
    df = df[df.is_mod_legal == 1]
    
    
    #check if creature and drop non-creatures    
    df['is_creature'] = df.apply(crea, axis=1)
    df = df[df.is_creature == 1]

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
    df['is_color_allowed'] = df.apply(col_allow, axis=1, color_num=5)
    df = df[df.is_color_allowed == True]
    

    
    #delete the above extra 4 columns used for cleaning
    for col in ["is_mod_legal", "is_creature", "is_star", "is_colorless", "is_compound", 'is_color_allowed']:
        df.drop(col, axis = 1, inplace=True)
    
    
    df['power'] = pd.to_numeric(df['power'])
    df['toughness'] = pd.to_numeric(df['toughness'])
    
    df['pt_ratio'] = df['power']/ df['toughness']
    df['p+t/cmc'] = (df['power'] + df['toughness']) / df['cmc']
    
    #add oracle text list for bag-of-words
    df['oracle_text'] = df['oracle_text'].str.lower()
    df['oracle_text'] = df['oracle_text'].str.replace(':','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('.','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace(',','', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace('\n',' ', regex=False)
    df['oracle_text'] = df['oracle_text'].str.replace("\(.*?\)",'', regex=True)
    df['oracle_text'] = df['oracle_text'].str.replace("\{.*?\}",'', regex=True)
    df['oracle_text'] = df['oracle_text'].str.split()
    
    #add column of length of Oracle text and drop text
    df['oracle_length'] = df.apply(orac_len, axis=1)
    
    
    #delete the rest of the extra columns not needed for modeling
    for col in ['object','type_line', 'legalities', 'set_name', 'oracle_id','oracle_text']:
        df.drop(col, axis = 1, inplace=True)
    
    # print(df.head())
    # print(df['oracle_text'])
    # print(df['oracle_text'][12])
    # print(df['oracle_text'][25023])
    
    #TODO: turn the list (into bag of words?) into features using color pie website
    #TODO: download new cards from scryfall!
    
    #finding the most common keywords to one-hot encode 
    
    key_dict = {}
    for key_list in df['keywords']:
        for keyw in key_list:
            if (not keyw in key_dict):
                key_dict[keyw] = 1
            else:
                key_dict[keyw] += 1
    
    sorted_dict = {}
    sorted_keys = sorted(key_dict, key=key_dict.get)
    
    for w in sorted_keys:
        sorted_dict[w] = key_dict[w]
    
    #storing in keyw_valid after deciding cutoff  
    keyw_valid = []
    for keys in sorted_dict.keys():
        if (sorted_dict[keys] > 10):
            keyw_valid.append(keys)
    #one-hot encoding the keys in keyw_valid
    col_add = []
    for keys in keyw_valid:
        df['has_' + keys] = df.apply(has_keyw, axis =1, keyword = keys)
        col_add.append('has_' + keys)
        
    for col in col_add:
        df[col] = df[col].astype(int)
    df.drop("keywords", axis=1, inplace=True)
        
    i = list(df.columns)
    a, b = i.index("cmc"), i.index("colors")
    i[b], i[a] = i[a], i[b]
    df = df[i]
    
    #one-hot encode the rarity
    df['is_common'] = df.apply(rar, axis=1, rarity='common')
    df['is_uncommon'] = df.apply(rar, axis=1, rarity='uncommon')
    df['is_rare'] = df.apply(rar, axis=1, rarity='rare')
    df['is_mythic'] = df.apply(rar, axis=1, rarity='mythic')
    df.drop('rarity', axis = 1, inplace = True)
    
    pd.set_option('display.max_columns', None) 
    #clean data to prepare for model
    x = df.iloc[:, 2:]
    y = df[['colors']]
    
    clean_x = clean_dataset(x)
    new_df = clean_x.merge(y, left_index = True, right_index = True)
    
    y = new_df['colors'].astype(str)
    x = new_df.drop('colors', axis =1 , inplace=False)
    
    print(x)
    
    color_dict = {}
    for i in y:
        if(not i in color_dict):
            color_dict[i] =1
        else:
            color_dict[i] += 1
    
    #build model
    model = RandomForestClassifier(max_depth = 17, n_estimators=1000)
    #TODO: out of bag?
    model.fit(x,y)
    
    #evaluate model
    n_scores = cross_val_score(model, x, y)
    print(n_scores)
    
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
