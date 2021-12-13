'''
File name: utils.py
Author: Oskar, Karim 
Date created: 13/12/2021
Date last modified: 14/12/2021
Python Version: 3.8
'''
import numpy as np
import seaborn as sns
import pycountry_convert as pc
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import cluster as skc


def group_nation_by_continent(df):
    '''
    Replaces nation in column of nationalities 
    by continent
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

    for speaker_feat_idx, speaker_feat in enumerate(attributes):
        clusters_curr_feat = df_attributes.groupby(['cluster', speaker_feat]).size().reset_index(level=[0,1])

        if speaker_feat == 'date_of_birth':
            use_sns = False
        else:
            use_sns = True
            # map each label of the speaker attribute to a certain colour
            palette = dict(zip(clusters_curr_feat[speaker_feat].unique(), sns.color_palette()))
            
        for cluster_i in range(n_clusters):
            axis = axes[speaker_feat_idx, cluster_i]
            cluster = clusters_curr_feat[clusters_curr_feat['cluster'] == cluster_i]
            values = cluster[0]
            labels = cluster[speaker_feat]
            if not values.isnull().all():
                if use_sns:
                    sns.barplot(x = labels, y = values, ax = axis, palette = palette)
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