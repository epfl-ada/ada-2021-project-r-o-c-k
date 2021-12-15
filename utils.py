'''
File name: utils.py
Author: Oskar, Karim, Celinna
Date created: 13/12/2021
Date last modified: 14/12/2021
Python Version: 3.8
'''
import numpy as np
import pandas as pd
import seaborn as sns
import pycountry_convert as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.colors
from sklearn.metrics import silhouette_score
from sklearn import cluster as skc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, cluster
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from mappings import *
import itertools
from scipy.stats import mannwhitneyu
from statistics import median
from operator import itemgetter

def group_nation_by_continent(df):
    '''
    Replaces nation in column of nationalities by continent
    
    :param df: df with a column named nationality, containing country names
    :return df_new: df with a column named nationality, containing continents
    '''
    df_new = df.copy()
    for idx, nation in enumerate(df_new["nationality"]):
        try:
            df_new["nationality"].iloc[idx] = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(nation, cn_name_format="default"))
        except:
            df_new["nationality"].iloc[idx] = np.NaN
    return df_new


def group_generations(sample):
    birth = sample.copy()
    birth["date_of_birth"] = birth["date_of_birth"].astype(int)
    birth["date_of_birth"] = birth["date_of_birth"].apply(lambda x: '30s' if x < 1940 else '50s' if x < 1960 else '70s' if x < 1980  else '90s' if x < 2000 else '00s' if x < 2020 else x)
    
    return birth


def normalize_min_max(df):
    '''
    Normalises each column through
    
    min-max normalization
    :param df: df to normalize
    :return: normalized df
    '''
    return (df - df.min())/(df.max() - df.min())


def plot_speaker_attr_within_clusters(df_attributes, attributes, n_clusters):
    '''
    Plots distribution of attributes within each cluster
    
    :param df_attributes: df with attributes and cluster column named 'cluster'
    :param attributes: list of attributes to plot, should exist in df_attributes
    :param n_clusters: number of clusters to plot
    '''
    n_speaker_feats = len(attributes)
    fig, axes = plt.subplots(n_speaker_feats, n_clusters, figsize = (n_clusters * 3, n_speaker_feats * 2), sharey = False)
    
    palettes = [palette_generation, palette_continent, palette_gender, palette_occupation, palette_degree, palette_religion]
    
    for speaker_feat_idx, speaker_feat in enumerate(attributes):
        clusters_curr_feat = df_attributes.groupby(['cluster', speaker_feat]).size().reset_index(level=[0,1])

        if speaker_feat == 'date_of_birth':
            use_sns = False
        else:
            use_sns = True
            
        for cluster_i in range(n_clusters):
            axis = axes[speaker_feat_idx, cluster_i]
            cluster = clusters_curr_feat[clusters_curr_feat['cluster'] == cluster_i]
            values = cluster[0]
            labels = cluster[speaker_feat]
            if not values.isnull().all():
                if use_sns:
                    sns.barplot(x = labels, y = values, ax = axis, palette = palettes[speaker_feat_idx])
                    axis.set_xticklabels(labels, rotation = 45, fontsize = 10)
                else:
                    axis.bar(labels, values)
            axis.set_title("Cluster {} - {}".format(cluster_i, speaker_feat))
            axis.set(ylabel = None, xlabel = None)

    fig.tight_layout()
    fig.text(0.4,0, "Speaker attributes of quotated speakers")
    fig.text(0,0.4, "Number of quotated speakers", rotation = 90)
    fig.suptitle("Distribution of speaker attributes per cluster", fontsize = 16)
    plt.subplots_adjust(top=0.92)


def cluster_with_best_threshold(cluster_features, return_attributes, thresholds, n_clusters_min = 2):
    '''
    Cluster with different distance thresholds and choose clusters with highest silhouette score
    
    :param cluster_features: df with features to use for clustering
    :param return_attributes: df to add cluster label column to
    :param thresholds: list of thresholds to evaluate, sorted in increasing order
    :param n_clusters_min: minimum amount of clusters to generate
    :return: (df with return_attributes and cluster label column, number of clusters for best clustering)
    '''    
    best_labels = []
    best_score = -1
    best_cluster_n = 0

    normalized_features = normalize_min_max(cluster_features)

    for i, threshold in enumerate(thresholds):
        model = skc.AgglomerativeClustering(n_clusters = None, linkage = 'complete', distance_threshold = threshold)
        model.fit(normalized_features)
        n_clusters = model.n_clusters_
        try:
            model_score = silhouette_score(normalized_features, model.labels_)
        except ValueError:
            print("Threshold {} only gave one cluster".format(threshold))
            break
        print("Threshold {} gave {} clusters and silhouette score {}.".format(threshold, n_clusters, model_score))
        
        # stop looking if we already have a valid clustering and n_clusters is smaller than n_clusters_min
        if n_clusters_min > n_clusters and best_score != - 1:
            break
        elif best_score < model_score:
            best_score = model_score
            best_labels = model.labels_
            best_cluster_n = n_clusters

    print("{} clusters was generated with silhouette score {}.".format(best_cluster_n, best_score))

    # add best cluster labels to speaker attributes
    return_attributes['cluster'] = best_labels

    return return_attributes, best_cluster_n


def mean_boxplots(df, plot_by, features, palette, order=None):
    """
    Plots the means of a given feature with respect to a given parameter
    
    param df: dataframe to be analyzed
    param plot_by: list of parameters in string format to be compared
    param features: list of features in string format to be plotted
    param palette: specific palette to plot with
    param order: order of classes to be plotted (only used for generation)
    """
    
    num = int(math.sqrt(len(features)))
    fig, axs = plt.subplots(num,num,figsize=(25, 15))
    fig.subplots_adjust(hspace=0.4)
    for i,feat in enumerate(features):
        if i<num**2:
            x_idx = int(np.floor(i/num))
            y_idx = int(i%num)
            sns.boxplot(x=plot_by, y=feat, data=df, ax=axs[x_idx,y_idx], order=order, palette=palette)
            axs[x_idx,y_idx].tick_params(axis='x', rotation=45)
            
    plt.show()
    
            
def sample_dataset(years, sample_size, save=False):
    '''
    Sample lines from datasets of every year
    
    param years: years to sample from
    param sample_size: sample per chunk to take
    param save: whether to save the sample
    '''
    generate_path = './../../data/merged_data/'
    file_prefix_r = 'merged_data_'
    file_suffix_r = '.csv.gzip'
    
    chunksize = 40000
    samples = []
    # get a sample of the df from each chunk from each year
    for year in years:
        print("Year: " + str(year))
        # read the file from year and filter away features that are not selected
        file_name_r = file_prefix_r + year + file_suffix_r
        df_quotes_chunks = pd.read_csv(generate_path + file_name_r, chunksize = chunksize, low_memory = False, compression='gzip')

        for i, chunk in enumerate(df_quotes_chunks):
            # get sample and append it to samples
            sample = chunk.sample(sample_size)
            samples.append(sample)

            if i % 100 == 0:
                print("Chunk: " + str(i))

    sample = pd.concat(samples)
    sample.drop_duplicates(inplace=True)
    sample.reset_index()
    sample.shape

    sample.to_csv(join(generate_path, 'sample.csv'))
    
    
def merge(df_quotes, df_speaker):
    '''
    Merges quote features and speaker features
    
    param df_quotes: quotes feature df
    param df_speaker: speaker feature df
    return df: merged dataframe 
    '''
    df_quotes = df_quotes.set_index('qid')
    
    # drop duplicate rows (mostly data headers)
    df_quotes.drop_duplicates(inplace=True)
    df_quotes = df_quotes[df_quotes.quoteID.str.contains('quoteID') == False]
    
    # change value type
#     for field in df_quotes.columns:
#         df_quotes[field] = df_quotes[field].astype(dtypes[field], errors = 'raise')
    
    # merge df
    df = df_quotes.merge(df_speaker.set_index('id'), left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)
    
    return df


def tree_feature_select(X, y):
    '''
    Does feature selection using decision trees
    
    :param X: features (n_rows, n_features)
    :param y: target (n_rows,)
    :return clf: extra trees classifiers
    :return model: model with reduced features
    '''
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    
    return clf, model


def select_predictors(df, ft_language, target):
    '''
    Extracts portion of the df and extracts the most relevant features based on a target speaker attribute
    
    :param df: merged dataframe with language and speaker features
    :param ft_language: list of language features
    :param target: target speaker attribute to predict
    :return clf: extra trees classifier
    :return model: selected model
    :return shape: shape of the data used (some target speaker features have nan which needs to be removed)
    :return X: features used to get model
    :return y: targets used to get model
    '''
    temp = df[ft_language].copy()
    temp[target] = df[target]
    temp.dropna(inplace=True)
    
    X = temp[ft_language]
    y = temp[target]
    clf, model = tree_feature_select(X, y)
    clf.feature_importances_
    
    return clf, model, temp.shape, X, y



def get_balanced_sample(target, target_dict, sample_size=50):
    '''
    Get a balanced data sample from the data of all years
    
    param target: target speaker feature to balance
    param target_dict: dictionary of the speaker feature
    param sample_size: number of values to sample per chunk
    
    return df_new: the new dataframe with balanced classes
    '''
    # file location
    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    generate_path = './../../data/merged_data/'
    file_prefix_r = 'merged_data_'
    file_suffix_r = '.csv.gzip'

    df_new = pd.DataFrame() # initialize values
    
    # load and parse data in chunks
    chunksize = 50000
    for year in years:
        print("Year: " + str(year))
        # read the file from year and filer away features that are not selected
        file_name_r = file_prefix_r + year + file_suffix_r
        chunks = pd.read_csv(generate_path + file_name_r, chunksize = chunksize, compression='gzip', low_memory=False)
        for i, chunk in enumerate(chunks):
            size = sample_size
            # check for smallest sample
            for key in target_dict.keys():
                sample = chunk.loc[chunk[target] == key]
                if sample.shape[0] < size:
                    size = sample.shape[0]

            # add sample to df
            if size > 0:
                for key in target_dict.keys():
                    sample = chunk.loc[chunk[target] == key].sample(size, replace=False, random_state=1)
                    df_new = pd.concat([df_new, sample])

            if i % 100 == 0:
                print("Chunk: " + str(i), df_new.shape)
    
    return df_new


def train_GBR_model(X, y):
    '''
    Training a gradient boosting regressor
    
    param X: np matrix of (n_rows, n_features)
    param y: np array of (n_rows)
    
    return model: GBR model
    return r2: model residual squared (represents accuracy)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train, y_train)
    reg.predict(X_test)
    r2 = reg.score(X_test, y_test)
    
    return reg, r2


def explained_variance(ipca):
    '''
    Check how many principle components results in 80% of the variance
    
    param ipca: incremental PCA
    '''
    k = 0.0
    weight = ipca.explained_variance_ratio_*100
    print('Explained variances: {}'.format(weight*100))
    idx = -1
    for i in range(len(weight)):
        if k < 80.0:
            k = k + weight[i]
            idx = i

    print(f'\nTo explain {round(k,2)} % of the variance we need the first {idx+1} principal components')
    
    
def setProperties():
    '''
    Set color properties for plots
    '''
#     custom_palette = sns.set_palette(sns.color_palette(mappings.colors))
    
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "#234473"],
              [norm( 1.0), "#AD2646"]]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    matplotlib.rcParams['font.family'] = "monospace"
    matplotlib.rcParams['font.weight'] = "semibold"
    
    return cmap
    
    
def test_significance(df, lexical_features, speaker_features, tresh=0.05, verbose=False):
    '''
    Get significances between features
    
    param df: dataframe
    param lexical_features: lexical features
    param speaker_features: speaker features
    param tresh:  p-value threshold
    param verbose: whether to print values
    
    return Grid_median: median significance values
    return Grid_max: max significance values
    return Significant_list: list of significant values
    return Significant_p: list of p-values
    '''
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
                        
                        if min_p == 0:
                            Grid_max[idx_j,idx_i] = 1/1E-200
                        else:    
                            Grid_max[idx_j,idx_i] = 1/min_p
                    res.append(p)   
                except:
                    print('Test failed for {},{}'.format(i,j))
           
            Grid_median[idx_j,idx_i] = 1/median(res)
            
            if verbose:
                print('The maximum statistically significant difference with {} in {} is between {} and {}'.format(j,i,k[0],k[1]))
                print('The p-value is {}'.format(min_p))
                  
    indices, Significant_p = zip(*sorted(enumerate(Significant_p), key=itemgetter(1))) #Sort the scores
    Significant_list = [Significant_list[i] for i in indices]
    
    return Grid_median, Grid_max, Significant_list, Significant_p


def plot_distribution(sample, lexical, speaker, binwidth, include_only=[]):
    '''
    To plot distributions between lexical and speaker attributes
    
    param sample: input dataframe
    param lexical: which lexical feature to plot (only 1)
    param speaker: which speaker attribute to plot (only 1)
    param binwidth: histogram bin width
    param include_only: which classes within the speaker attribute to include
    '''
    if include_only is None:
        ax = sns.kdeplot(x=lexical,hue=speaker,common_norm=False, data=sample, cut=0)
    else:
        tmp = sample[sample[speaker].isin(include_only)]
        ax = sns.kdeplot(x=lexical,hue=speaker,common_norm=False, data=tmp, cut=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1/binwidth))
    plt.ylabel("Probability", fontweight='semibold')
    plt.xlabel(ax.get_xlabel(), fontweight='semibold')
    plt.show()
    ax.get_figure().savefig("Distribution_{}_with_{}_and_{}.png".format(speaker,include_only[0], include_only[1]),dpi=300,bbox_inches="tight")
    
    
def get_most_significant(Significant_list,Significant_p,rank):
    '''
    Get most significant lexical and speaker attribute pairs
    
    param significant_list: list of significance
    param significant_p: list of p values
    param rank: number of pairs to output, ranked by most significant first
    return: significant_list and significant_p of the first rank pairing
    '''
    return Significant_list[:rank], Significant_p[:rank]


def get_least_significant(Significant_list,Significant_p,rank):
        '''
    Get most significant lexical and speaker attribute pairs
    
    param significant_list: list of significance
    param significant_p: list of p values
    param rank: number of pairs to output, ranked by least significant first
    return: significant_list and significant_p of the first rank pairing
    '''
    return Significant_list[-rank:], Significant_p[-rank:]

