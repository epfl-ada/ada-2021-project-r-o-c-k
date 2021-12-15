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
from statistics import median
import matplotlib.colors
from matplotlib.ticker import PercentFormatter
from operator import itemgetter

path = '/home/karim/Downloads/Newmerged_data_2020.csv.gzip'  #Modify

df_quotes_chunks = pd.read_csv(path, chunksize = 40000, low_memory = False, compression='gzip')
samples = []
for i, chunk in enumerate(df_quotes_chunks):
    # get sample and append it to samples
    sample = chunk.sample(400)
    samples.append(sample)

sample = pd.concat(samples)
sample.drop_duplicates(inplace=True)
sample.reset_index()
sample.shape
sample.head(10)

def TestSignificance(df,lexical_features,speaker_features, tresh=0.05, verbose=False):
    Grid_max = np.zeros(shape=(len(lexical_features),len(speaker_features)))
    Grid_median = np.zeros(shape=(len(lexical_features),len(speaker_features)))
    Significant_list = []
    Significant_p =[]
    for idx_i,i in enumerate(speaker_features):                                      #Iterate over
        for idx_j,j in enumerate(lexical_features):                                  #Iterate over
            unique = df[i].unique()                  #Iterate over the different attributes
            unique = unique[unique != np.array(None)]
            unique = np.array(unique, dtype=str)
            unique = unique[unique != 'nan']
            unique = unique[unique != 'Other']
            pairs = list(itertools.combinations(unique,2))
            significant_list = []
            significant_p = []
            min_p = 1
            res = []
            for idx,k in enumerate(pairs):
                try:
                    N, p = mannwhitneyu(df[df[i]==k[0]][j], df[df[i]==k[1]][j])
                    Significant_list.append([k[0],k[1],j,i])
                    Significant_p.append(p)
                    if p < min_p:
                        min_p = p
                        min_x = k[0]
                        min_y = k[1]
                        param = j
                        speak = i
                        Grid_max[idx_j,idx_i] = 1/min_p
                    res.append(p)   
                except:
                    print('Test failed for {},{}'.format(i,j))
            if verbose:
                print('The maximum statistically significant difference with {} in {} is between {} and {}'.format(j,i,k[0],k[1]))
                print('The p-value is {}'.format(min_p))
            Grid_median[idx_j,idx_i] = 1/median(res)
    indices, Significant_p = zip(*sorted(enumerate(Significant_p), key=itemgetter(1))) #Sort the scores
    Significant_list = [Significant_list[i] for i in indices]
    return Grid_median, Grid_max, lexical_features, speaker_features, Significant_list, Significant_p

def GroupBirthDates(sample):
    birth = sample.copy()
    birth["date_of_birth"] = birth["date_of_birth"].astype(int)
    birth["date_of_birth"] = birth["date_of_birth"].apply(lambda x: '30s' if x < 1940 else '50s' if x < 1960 else '70s' if x < 1980  else '90s' if x < 2000 else '00s' if x < 2020 else x)
    return birth

def flatten(t):
    return [item for sublist in t for item in sublist]

def GroupNationalities(sample):
    nat = sample.copy()
    for idx,i in enumerate(nat["nationality"]):
        try:
            nat["nationality"].iloc[idx] = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(i, cn_name_format="default"))
        except:
            nat["nationality"].iloc[idx] = None
    return nat

Test = sample.copy()
Test = GroupBirthDates(Test)
Test = GroupNationalities(Test)

#selected_feats = ['sentence_count', 'approx_word_count','adj_per_word','verb_per_word', 'base_ratio', 'pres_ratio',
#       'past_ratio', 'pronoun_per_word', 'self_ratio', 'union_ratio',
#       'other_ratio', 'sentiment']
selected_feats = ['sentence_count', '._per_sentence', ',_per_sentence', '!_per_sentence',
       '?_per_sentence', ':_per_sentence', ';_per_sentence', 'sign_per_token',
       'punctuation_per_sentence', 'approx_word_count', 'token_count',
       'adj_per_word', 'ordinal_ratio', 'comparative_ratio',
       'superlative_ratio', 'verb_per_word', 'base_ratio', 'pres_ratio',
       'past_ratio', 'pronoun_per_word', 'self_ratio', 'union_ratio',
       'other_ratio', 'sentiment']
speaker_feats = ['date_of_birth','nationality', 'gender', 'occupation', 'academic_degree', 'religion']
Grid_mean, Grid_max, lexical_features, speaker_features, Significant_list, Significant_p = TestSignificance(Test,selected_feats,speaker_feats,verbose=False)

def plotDistribution(sample, lexical, speaker, binwidth, include_only=[]):
    if include_only is None:
        ax = sns.kdeplot(x=lexical,hue=speaker,common_norm=False, data=sample, cut=0)
    else:
        tmp = sample[sample[speaker].isin(include_only)]
        ax = sns.kdeplot(x=lexical,hue=speaker,common_norm=False, data=tmp, cut=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1/binwidth))
    plt.ylabel("Probability", fontweight='semibold')
    plt.xlabel(ax.get_xlabel(), fontweight='semibold')
    plt.show()
    
def getMostSignificant(Significant_list,Significant_p,rank):
    return Significant_list[:rank], Significant_p[:rank]
def getLeastSignificant(Significant_list,Significant_p,rank):
    return Significant_list[-rank:], Significant_p[-rank:]

Most_Significant_list, p_list = getMostSignificant(Significant_list,Significant_p,4)
Least_Significant_list, p_list = getLeastSignificant(Significant_list,Significant_p,4)

for i in Most_Significant_list:
    plotDistribution(sample,i[2],i[3],binwidth=0.01,include_only=i[:2])