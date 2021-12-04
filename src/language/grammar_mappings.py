
'''
File name: grammar_mappings.py
Author: Karim, Oskar
Date created: 04/12/2021
Date last modified: 04/12/2021
Python Version: 3.8
'''
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