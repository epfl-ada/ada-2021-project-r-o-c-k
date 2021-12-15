'''
File name: mappings.py
Author: Karim, Oskar, Celinna
Date created: 04/12/2021
Date last modified: 04/12/2021
Python Version: 3.8
'''
import seaborn as sns

# dictionaries for lexical features
pronoun_dict = {
    'me' :'self',
    'myself' : 'self',
    'ours' : 'union',
    'ourselves' : 'union',
    'us' : 'union',
    'hers' : 'other',
    'herself' : 'other',
    'him' : 'other',
    'himself' : 'other',
    'hisself' : 'other',
    'one' : 'other',
    'oneself' : 'other',
    'she' : 'other',
    'thee' : 'other',
    'theirs' : 'other',
    'them' : 'other',
    'themselves' : 'other',
    'they' : 'other',
    'thou' : 'other',
    'thy' : 'other',
    'mine' : 'self',
    'my' : 'self',
    'our' : 'union',
    'ours' : 'union',
    'her' : 'other',
    'his' : 'other',
    'their' : 'other',
    'they' : 'other', 
    'your' : 'other',
    'we' : 'union',
    'i': 'self',
    'he': 'other',
    'she': 'other',
    'you': 'other',
    'yourself': 'other'
}

verb_tag_dict = {
    'VB': 'base', 
    'VBD': 'past', 
    'VBN': 'past', 
    'VBG': 'pres', 
    'VBP': 'pres', 
    'VBZ': 'pres'
}

adj_tag_dict = {
    'JJ': 'ordinal',
    'JJR': 'comparative',
    'JJS': 'superlative'
}


# dictionary for clean data loading
dtypes ={'quoteID': str,
      'qid': str,
       'sentence_count':int, 
       '._per_sentence':float, 
       ',_per_sentence':float, 
       '!_per_sentence':float,
       '?_per_sentence':float, 
       ':_per_sentence':float, 
       ';_per_sentence':float, 
       'sign_per_token':float,
       'punctuation_per_sentence':float, 
       'approx_word_count': int, 
       'token_count': int,
       'adj_per_word':float, 
       'ordinal_ratio':float, 
       'comparative_ratio':float,
       'superlative_ratio':float, 
       'verb_per_word':float, 
       'base_ratio':float, 
       'pres_ratio':float,
       'past_ratio':float, 
       'pronoun_per_word':float, 
       'self_ratio':float, 
       'union_ratio':float,
       'other_ratio':float, 
       'sentiment':float
        }


# Dictionaries used during regression model training
gender_dict = {
    'Male': 0,
    'Female': 1,
    'Other': 2
}

religion_dict = {
    'Christian': 0,
    'Hindus': 1, 
    'Muslim': 2, 
    'Jewish': 3,
    'Other': 4
}

degree_dict = {
    'Bachelor': 0, 
    'Master': 1,
    'Doctorate': 2,
    'Other': 3
}

occupation_dict = {
    'Politics': 0, 
    'Arts': 1, 
    'Military': 2, 
    'Sciences': 3, 
    'Business': 4,
    'Sports': 5, 
    'Religion': 6,
    'Other': 7
}

continent_dict = {
    'NA': 0, 
    'AS': 1, 
    'EU': 2, 
    'OC': 3, 
    'SA': 4, 
    'AF': 5
}

generation_dict = {
    '30s': 0, 
    '50s': 1, 
    '70s': 2, 
    '90s': 3, 
    '00s': 4
}

# unique color palettes
colors = ["#AD2646","#234473","#F4D7DB","#512E3B","#126EA8","#A76662","#D09790","#331F1C"]

palette_gender = dict(zip(gender_dict.keys(), sns.color_palette(colors)))
palette_religion = dict(zip(religion_dict.keys(), sns.color_palette(colors)))
palette_degree = dict(zip(degree_dict.keys(), sns.color_palette(colors)))
palette_occupation = dict(zip(occupation_dict.keys(), sns.color_palette(colors)))
palette_continent = dict(zip(continent_dict.keys(), sns.color_palette(colors)))
palette_generation = dict(zip(generation_dict.keys(), sns.color_palette(colors)))