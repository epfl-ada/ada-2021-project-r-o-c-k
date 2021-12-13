#Imports
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import itertools
import pycountry_convert as pc
from scipy.stats import mannwhitneyu

path = '/home/karim/Downloads/Newmerged_data_2020.csv.gzip'  #Modify

df_quotes_chunks = pd.read_csv(path, chunksize = 40000, low_memory = False, compression='gzip')
samples = []
for i, chunk in enumerate(df_quotes_chunks):
    # get sample and append it to samples
    sample = chunk.sample(1000)
    samples.append(sample)

sample = pd.concat(samples)
sample.drop_duplicates(inplace=True)
sample.reset_index()
sample.shape

def GroupBirthDates(sample):
    birth = sample.copy()
    birth["date_of_birth"] = birth["date_of_birth"].astype(int)
    birth["date_of_birth"] = birth["date_of_birth"].apply(lambda x: '30s' if x < 1940 else '50s' if x < 1960 else '70s' if x < 1980  else '90s' if x < 2000 else '00s' if x < 2020 else x)
    return birth

def GroupNationalities(sample):
    nat = sample.copy()
    for idx,i in enumerate(nat["nationality"]):
        try:
            nat["nationality"].iloc[idx] = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(i, cn_name_format="default"))
        except:
            nat["nationality"].iloc[idx] = None
    return nat

def TestSignificance(df,lexical_features,speaker_features,tresh=0.05):
    Grid = np.zeros(shape=(len(lexical_features),len(speaker_features)))
    for idx_i,i in enumerate(speaker_features):                                      #Iterate over
        for idx_j,j in enumerate(lexical_features):                                  #Iterate over
            unique = df[i].unique()                  #Iterate over the different attributes
            unique = unique[unique != np.array(None)]
            unique = np.array(unique, dtype=str)
            unique = unique[unique != 'nan']
            pairs = list(itertools.combinations(unique,2))
            min_p = 1
            for k in pairs:
                try:
                    N, p = mannwhitneyu(df[df[i]==k[0]][j], df[df[i]==k[1]][j])
                    if p<tresh:
                        print('A statistically significant difference with {} in {} between {} and {}'.format(j,i,k[0],k[1]))
                        print('The p-value is {}'.format(p))
                        Grid[idx_j,idx_i] = 1
                except:
                    print('Test failed for {},{}'.format(i,j))
    return Grid, lexical_features, speaker_features

Test = GroupBirthDates(sample)
Test = GroupNationalities(Test)

selected_feats = ['sentence_count',
       'approx_word_count', 'token_count',
       'adj_per_word', 'ordinal_ratio', 'comparative_ratio',
       'superlative_ratio', 'verb_per_word', 'base_ratio', 'pres_ratio',
       'past_ratio', 'pronoun_per_word', 'self_ratio', 'union_ratio',
       'other_ratio', 'sentiment']
speaker_feats = ['date_of_birth','nationality', 'gender', 'occupation', 'academic_degree', 'religion']

Grid, lexical_features, speaker_features = TestSignificance(Test,selected_feats,speaker_feats,tresh=0.05)
print(Grid)