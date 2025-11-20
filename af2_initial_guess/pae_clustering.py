#!/usr/bin/env python3

import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing
import subprocess
import shutil
import os
import re
from clean_af2 import clean_checkpoint, repair_pae_script

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Globals for worker processes (set by initializer)
_GLOBAL_AF2SCORES = None
_GLOBAL_AF2PAE = None
_GLOBAL_CONTIGMAP = None
_GLOBAL_HOTSPOTS = None
_GLOBAL_KMEANS_N_INIT = 1

def _init_worker(af2scores, af2pae, contigmap, hotspots, kmeans_n_init):
    """Initializer to bind large objects once per process (no per-task pickling)."""
    global _GLOBAL_AF2SCORES, _GLOBAL_AF2PAE, _GLOBAL_CONTIGMAP, _GLOBAL_HOTSPOTS, _GLOBAL_KMEANS_N_INIT
    _GLOBAL_AF2SCORES = af2scores
    _GLOBAL_AF2PAE = af2pae
    _GLOBAL_CONTIGMAP = contigmap
    _GLOBAL_HOTSPOTS = hotspots
    _GLOBAL_KMEANS_N_INIT = kmeans_n_init

def process_pae_index(i):
    """Lightweight wrapper calling original logic using globals."""
    return process_pae(
        i,
        _GLOBAL_AF2SCORES,
        _GLOBAL_AF2PAE,
        _GLOBAL_CONTIGMAP,
        _GLOBAL_HOTSPOTS,
    )

def safe_int(x):
    try:
        return int(float(str(x)))
    except:
        return np.nan

def hotspot_mapping(hotspots, contigmap, binderlen):
    """
    Maps the given hotspots to their positions according to the contigmap.

    Args:
        hotspots (str): A string containing the hotspots to be mapped.
        contigmap (str): A string containing the contigmap from RFdiffusion.
        binderlen (int): The length of the binder.

    Returns:
        list: A list of residue positions corresponding to the mapped hotspots.

    """
    # Rest of the code...
    ranges = re.findall(r'[AB](\d+)-(\d+)/0', contigmap)
    numbers = []
    for start, end in ranges:
            numbers.extend(range(int(start), int(end) + 1))
    hotspots = re.findall(r'[AB](\d+)', hotspots)
    hotspots = [int(number) for number in hotspots]
    # Map each hotspot to its position in numbers
    hotspot_positions = {}
    for hotspot in hotspots:
        if hotspot in numbers:
            position = numbers.index(hotspot)
            hotspot_positions[hotspot] = position
        else:
            hotspot_positions[hotspot] = None
    hotspot = list(hotspot_positions.values())
    residues = [hotspot + binderlen + 1 for hotspot in hotspot]
    return residues

def process_pae(i, af2scores, af2pae, contigmap, hotspots):
    """
    Process the PAE (Protein-Antigen Interface Energy) for a given index and performs KMeans clustering.
    
    Args:
        i (int): The index of the PAE to process.
        af2scores (pandas.DataFrame): The DataFrame containing AF2 scores.
        pae (pandas.DataFrame): The DataFrame containing PAE values.
        contigmap (str): The contigmap parameter from RFdiffusion.
        hotspots (str): The hotspots parameter from RFdiffusion.
    
    Returns:
        dict: A dictionary containing the following information:
            - 'min_pae' (float): The minimum mean PAE score from the 8 clusters after KMeans clustering.
            - 'max_pae' (float): The maximum mean PAE score from the 8 clusters after KMeans clustering.
            - 'min_pae_size' (int): The cluster size (row x column) for the cluster with the minimum mean PAE.
            - 'min_pae_size_fraction' (float): The fraction of the cluster size for the cluster with the minimum mean PAE.
            - 'min_pae_shape' (tuple): The shape (row x column) for the cluster with the minimum mean PAE.
            - 'min_pae_cluster' (str): The cluster number for the cluster with the minimum mean PAE.
            - 'hotspot_min_pae' (float): The minimum mean PAE score from the 4 clusters after KMeans clustering of hotspot residues.
            - 'hotspot_max_pae' (float): The maximum mean PAE score from the 4 clusters after KMeans clustering of hotspot residues.
            - 'hotspot_min_pae_size' (int): The cluster size (row x column) for the cluster with the minimum mean PAE of hotspot residues.
            - 'hotspot_min_pae_size_fraction' (float): The fraction of the cluster size for the cluster with the minimum mean PAE of hotspot residues.
            - 'hotspot_min_pae_shape' (tuple): The shape (row x column) for the cluster with the minimum mean PAE of hotspot residues.
            - 'hotspot_min_pae_cluster' (str): The cluster number for the cluster with the minimum mean PAE of hotspot residues.
            - 'pae_sample' (str): The description for the cluster with the minimum mean PAE.
    """

    try:
        binderlen_raw = af2scores.at[i, 'binderlen'] if 'binderlen' in af2scores.columns else np.nan
        binderlen = safe_int(binderlen_raw)

        af2scores_sample = str(af2scores.at[i, 'description']).strip()
        af2scores_sample_clean = af2scores_sample.replace(' ', '')

        if pd.isna(binderlen) or af2scores_sample_clean == 'nan':
            return {
                'min_pae': np.nan,
                'max_pae': np.nan,
                'weighted_score': np.nan,
                'min_pae_size': np.nan,
                'min_pae_size_fraction': np.nan,
                'min_pae_shape': np.nan,
                'min_pae_cluster': np.nan,
                'hotspot_min_pae': np.nan,
                'hotspot_max_pae': np.nan,
                'hotspot_weighted_score': np.nan,
                'hotspot_min_pae_size': np.nan,
                'hotspot_min_pae_size_fraction': np.nan,
                'hotspot_min_pae_shape': np.nan,
                'hotspot_min_pae_cluster': np.nan,
                'pae_sample': np.nan
            }

        # Clean PAE description column once
        if 2 in af2pae.columns:
            af2pae[2] = af2pae[2].astype(str).str.replace(' ', '').str.strip()
        else:
            return {
                'min_pae': np.nan, 'max_pae': np.nan, 'weighted_score': np.nan,
                'min_pae_size': np.nan, 'min_pae_size_fraction': np.nan,
                'min_pae_shape': np.nan, 'min_pae_cluster': np.nan,
                'hotspot_min_pae': np.nan, 'hotspot_max_pae': np.nan,
                'hotspot_weighted_score': np.nan, 'hotspot_min_pae_size': np.nan,
                'hotspot_min_pae_size_fraction': np.nan, 'hotspot_min_pae_shape': np.nan,
                'hotspot_min_pae_cluster': np.nan, 'pae_sample': np.nan
            }

        # Direct match (avoid idxmax on boolean)
        match_rows = af2pae[af2pae[2] == af2scores_sample_clean]
        if match_rows.empty:
            return {
                'min_pae': np.nan, 'max_pae': np.nan, 'weighted_score': np.nan,
                'min_pae_size': np.nan, 'min_pae_size_fraction': np.nan,
                'min_pae_shape': np.nan, 'min_pae_cluster': np.nan,
                'hotspot_min_pae': np.nan, 'hotspot_max_pae': np.nan,
                'hotspot_weighted_score': np.nan, 'hotspot_min_pae_size': np.nan,
                'hotspot_min_pae_size_fraction': np.nan, 'hotspot_min_pae_shape': np.nan,
                'hotspot_min_pae_cluster': np.nan, 'pae_sample': np.nan
            }

        j = match_rows.index[0]

        complex = af2pae[[1]].iloc[j].str.split(r'\s+|,\s*', expand=True)
        pae_sample = af2pae.at[j, 2]
        
        complex = complex.dropna(how='all', axis=1)
        complex_list = complex.values.flatten().tolist()
        complex_size = int(np.sqrt(len(complex_list)))

        # KMeans clustering of the interaction interface PAE values
        try:
            complex_matrix = np.reshape(complex_list, (complex_size, complex_size))
            #extract pae of binder
            binder_AB = complex_matrix[0:binderlen, binderlen:]
            binder_AB_transpose = np.transpose(binder_AB)
            
            binder_BA = complex_matrix[binderlen:, 0:binderlen]
            binder_BA_transpose = np.transpose(binder_BA)

            #perform kmeans clustering
            try:
                kmeans_AB_rows = KMeans(n_clusters=2, random_state=0, n_init=_GLOBAL_KMEANS_N_INIT).fit(binder_AB)
            except Exception as e:
                print(f"KMeans_AB_rows failed at index {i}: {e}")
                kmeans_AB_rows = np.nan
            try:
                kmeans_AB_cols = KMeans(n_clusters=2, random_state=0, n_init=_GLOBAL_KMEANS_N_INIT).fit(binder_AB_transpose)
            except Exception as e:
                print(f"KMeans_AB_cols failed at index {i}: {e}")
                kmeans_AB_cols = np.nan
            try:
                kmeans_BA_rows = KMeans(n_clusters=2, random_state=0, n_init=_GLOBAL_KMEANS_N_INIT).fit(binder_BA)
            except Exception as e:
                print(f"KMeans_BA_rows failed at index {i}: {e}")
                kmeans_BA_rows = np.nan
            try:
                kmeans_BA_cols = KMeans(n_clusters=2, random_state=0, n_init=_GLOBAL_KMEANS_N_INIT).fit(binder_BA_transpose)
            except Exception as e:
                print(f"KMeans_BA_cols failed at index {i}: {e}")
                kmeans_BA_cols = np.nan

            # Check if KMeans clustering gives a result
            if (kmeans_AB_rows is not None and not isinstance(kmeans_AB_rows, float) and
                kmeans_AB_cols is not None and not isinstance(kmeans_AB_cols, float) and
                np.unique(kmeans_AB_rows.labels_).size > 1 and np.unique(kmeans_AB_cols.labels_).size > 1):            
                try:
                    rows0 = kmeans_AB_rows.labels_ == 0
                    cols0 = kmeans_AB_cols.labels_ == 0
                    rows1 = kmeans_AB_rows.labels_ == 1
                    cols1 = kmeans_AB_cols.labels_ == 1

                    cluster1 = binder_AB[np.ix_(rows0, cols0)]
                    cluster2 = binder_AB[np.ix_(rows1, cols1)]
                    cluster3 = binder_AB[np.ix_(rows0, cols1)]
                    cluster4 = binder_AB[np.ix_(rows1, cols0)]
                    
                    mean_AB = pd.DataFrame([np.mean(cluster1.astype(np.float64)), 
                                        np.mean(cluster2.astype(np.float64)),
                                        np.mean(cluster3.astype(np.float64)),
                                        np.mean(cluster4.astype(np.float64))], columns = ['means'])
                    mean_AB['cluster'] = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']
                    mean_AB['size'] = [cluster1.size, cluster2.size, cluster3.size, cluster4.size]
                    mean_AB['shape'] = [cluster1.shape, cluster2.shape, cluster3.shape, cluster4.shape]
                    total_matrix_size_AB = binder_AB.shape[0] * binder_AB.shape[1]
                    mean_AB['size_weight'] = mean_AB['size'] / total_matrix_size_AB
                    mean_AB['weighted_score'] = mean_AB['means'] * (1 / mean_AB['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for clustered AB interface at index {i}: {e}")
            else:
                try:
                    mean_AB = np.mean(binder_AB.astype(np.float64))
                    mean_AB['cluster'] = 'KMeans_failed'
                    mean_AB['size'] = binder_AB.size
                    mean_AB['shape'] = binder_AB.shape
                    total_matrix_size_AB = binder_AB.shape[0] * binder_AB.shape[1]
                    mean_AB['size_weight'] = mean_AB['size'] / total_matrix_size_AB
                    mean_AB['weighted_score'] = mean_AB['means'] * (1 / mean_AB['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for AB interface at index {i}: {e}")

            if (kmeans_BA_rows is not None and not isinstance(kmeans_BA_rows, float) and 
                kmeans_BA_cols is not None and not isinstance(kmeans_BA_cols, float) and
                np.unique(kmeans_BA_rows.labels_).size > 1 and np.unique(kmeans_BA_cols.labels_).size > 1):
                try:
                    rows0 = kmeans_BA_rows.labels_ == 0
                    cols0 = kmeans_BA_cols.labels_ == 0
                    rows1 = kmeans_BA_rows.labels_ == 1
                    cols1 = kmeans_BA_cols.labels_ == 1

                    cluster5 = binder_BA[np.ix_(rows0, cols0)]
                    cluster6 = binder_BA[np.ix_(rows1, cols1)]
                    cluster7 = binder_BA[np.ix_(rows0, cols1)]
                    cluster8 = binder_BA[np.ix_(rows1, cols0)]
                    
                    mean_BA = pd.DataFrame([np.mean(cluster5.astype(np.float64)), 
                                        np.mean(cluster6.astype(np.float64)),
                                        np.mean(cluster7.astype(np.float64)),
                                        np.mean(cluster8.astype(np.float64))], columns = ['means'])
                    mean_BA['cluster'] = ['Cluster_5', 'Cluster_6', 'Cluster_7', 'Cluster_8']
                    mean_BA['size'] = [cluster5.size, cluster6.size, cluster7.size, cluster8.size]
                    mean_BA['shape'] = [cluster5.shape, cluster6.shape, cluster7.shape, cluster8.shape]
                    total_matrix_size_BA = binder_BA.shape[0] * binder_BA.shape[1]
                    mean_BA['size_weight'] = mean_BA['size'] / total_matrix_size_BA
                    mean_BA['weighted_score'] = mean_BA['means'] * (1 / mean_BA['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for clustered BA interface at index {i}: {e}")
            else:
                try:
                    mean_BA = np.mean(binder_BA.astype(np.float64))
                    mean_BA['cluster'] = 'KMeans_failed'
                    mean_BA['size'] = binder_BA.size
                    mean_BA['shape'] = binder_BA.shape
                    total_matrix_size_BA = binder_BA.shape[0] * binder_BA.shape[1]
                    mean_BA['size_weight'] = mean_BA['size'] / total_matrix_size_BA
                    mean_BA['weighted_score'] = mean_BA['means'] * (1 / mean_BA['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for BA interface at index {i}: {e}")

            pae_score = pd.concat([mean_AB, mean_BA], axis=0).reset_index(drop=True)
            min_pae = np.min(pae_score['means'].astype(np.float64))
            max_pae = np.max(pae_score['means'].astype(np.float64))
            pae_weighted = pae_score['weighted_score'].min()
            min_pae_size = pae_score.loc[pae_score['means'] == min_pae, 'size'].sum()
            min_pae_size_fraction = min_pae_size / (pae_score.loc[pae_score['means'] != min_pae, 'size'].sum())
            min_pae_shape = pae_score.loc[pae_score['means'].idxmin(), 'shape']
            min_pae_cluster = pae_score.loc[pae_score['means'].idxmin(), 'cluster']


            # Map hotspots to their positions in the trimmed PDB file according to the contigmap parameter
            residues = hotspot_mapping(hotspots, contigmap, binderlen)

            #extract pae of hotspot residues
            hotspot_matrix_0 = complex_matrix[residues, 0:binderlen]
            hotspot_matrix_1 = complex_matrix[0:binderlen, residues]

            #perform kmeans clustering of hotspot residues
            try:
                hotspot_cluster_0_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(np.transpose(hotspot_matrix_0))
                hotspot_cluster_1_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit((hotspot_matrix_1))
            except:
                print('KMeans clustering failed')

            try:
                hotspot_cluster1 = hotspot_matrix_0[:, hotspot_cluster_0_rows.labels_==0]
                hotspot_cluster2 = hotspot_matrix_0[:, hotspot_cluster_0_rows.labels_==1]
                hotspot_cluster3 = hotspot_matrix_1[hotspot_cluster_1_rows.labels_==0, :]
                hotspot_cluster4 = hotspot_matrix_1[hotspot_cluster_1_rows.labels_==1, :]
                
            except:
                print('Cluster filtering failed')

            if hotspot_cluster1 is not None and hotspot_cluster2 is not None:
                try:
                    mean_hotspot_0 = pd.DataFrame([np.mean(hotspot_cluster1.astype(np.float64)), 
                                        np.mean(hotspot_cluster2.astype(np.float64))], columns = ['means'])
                    mean_hotspot_0['cluster'] = ['Cluster_1', 'Cluster_2']
                    mean_hotspot_0['size'] = [hotspot_cluster1.size, hotspot_cluster2.size]
                    mean_hotspot_0['shape'] = [hotspot_cluster1.shape, hotspot_cluster2.shape]
                    if isinstance(mean_hotspot_0, pd.DataFrame):
                        total_hotspot_size_0 = hotspot_matrix_0.shape[0] * hotspot_matrix_0.shape[1]
                        mean_hotspot_0['size_weight'] = mean_hotspot_0['size'] / total_hotspot_size_0
                        mean_hotspot_0['weighted_score'] = mean_hotspot_0['means'] * (1 / mean_hotspot_0['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")
            else:
                try:
                    mean_hotspot_0 = np.mean(hotspot_matrix_0.astype(np.float64))
                    mean_hotspot_0['cluster'] = 'KMeans_failed'
                    mean_hotspot_0['size'] = hotspot_matrix_0.size
                    mean_hotspot_0['shape'] = hotspot_matrix_0.shape
                    if isinstance(mean_hotspot_0, pd.DataFrame):
                        total_hotspot_size_0 = hotspot_matrix_0.shape[0] * hotspot_matrix_0.shape[1]
                        mean_hotspot_0['size_weight'] = mean_hotspot_0['size'] / total_hotspot_size_0
                        mean_hotspot_0['weighted_score'] = mean_hotspot_0['means'] * (1 / mean_hotspot_0['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")

            if hotspot_cluster3 is not None and hotspot_cluster4 is not None:
                try:
                    mean_hotspot_1 = pd.DataFrame([np.mean(hotspot_cluster3.astype(np.float64)), 
                                        np.mean(hotspot_cluster4.astype(np.float64))], columns = ['means'])
                    mean_hotspot_1['cluster'] = ['Cluster_3', 'Cluster_4']
                    mean_hotspot_1['size'] = [hotspot_cluster3.size, hotspot_cluster4.size]
                    mean_hotspot_1['shape'] = [hotspot_cluster3.shape, hotspot_cluster4.shape]
                    if isinstance(mean_hotspot_1, pd.DataFrame):
                        total_hotspot_size_1 = hotspot_matrix_1.shape[0] * hotspot_matrix_1.shape[1]
                        mean_hotspot_1['size_weight'] = mean_hotspot_1['size'] / total_hotspot_size_1
                        mean_hotspot_1['weighted_score'] = mean_hotspot_1['means'] * (1 / mean_hotspot_1['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")
            else:
                try:
                    mean_hotspot_1 = np.mean(hotspot_matrix_1.astype(np.float64))
                    mean_hotspot_1['cluster'] = 'KMeans_failed'
                    mean_hotspot_1['size'] = hotspot_matrix_1.size
                    mean_hotspot_1['shape'] = hotspot_matrix_1.shape
                    if isinstance(mean_hotspot_1, pd.DataFrame):
                        total_hotspot_size_1 = hotspot_matrix_1.shape[0] * hotspot_matrix_1.shape[1]
                        mean_hotspot_1['size_weight'] = mean_hotspot_1['size'] / total_hotspot_size_1
                        mean_hotspot_1['weighted_score'] = mean_hotspot_1['means'] * (1 / mean_hotspot_1['size_weight'])
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")

            hotspot_pae_score = pd.concat([mean_hotspot_0, mean_hotspot_1], axis=0).reset_index(drop=True)
            hotspot_min_pae = np.min(hotspot_pae_score['means'].astype(np.float64))
            hotspot_max_pae = np.max(hotspot_pae_score['means'].astype(np.float64))
            hotspot_weighted_pae = hotspot_pae_score['weighted_score'].min()
            hotspot_min_pae_size = hotspot_pae_score.loc[hotspot_pae_score['means'] == hotspot_min_pae, 'size'].sum()
            hotspot_min_pae_size_fraction = hotspot_min_pae_size / (hotspot_pae_score.loc[hotspot_pae_score['means'] != hotspot_min_pae, 'size'].sum())
            hotspot_min_pae_shape = hotspot_pae_score.loc[hotspot_pae_score['means'].idxmin(), 'shape']
            hotspot_min_pae_cluster = hotspot_pae_score.loc[hotspot_pae_score['means'].idxmin(), 'cluster']
            
        except Exception as e:
            pae_score = np.nan
            min_pae = np.nan
            max_pae = np.nan
            min_pae_size = np.nan
            min_pae_size_fraction = np.nan
            min_pae_shape = np.nan
            min_pae_cluster = np.nan
            hotspot_pae_score = np.nan
            hotspot_min_pae = np.nan
            hotspot_max_pae = np.nan
            hotspot_min_pae_size = np.nan
            hotspot_min_pae_size_fraction = np.nan
            hotspot_min_pae_shape = np.nan
            hotspot_min_pae_cluster = np.nan
            print(f"Error calculating PAE score at index {i}: {e}. Setting to NaN")

        return {
            'min_pae': round(min_pae, 3), 
            'max_pae': round(max_pae, 3), 
            'weighted_score': round(pae_score['weighted_score'].min(), 3),
            'min_pae_size': min_pae_size, 
            'min_pae_size_fraction': min_pae_size_fraction,
            'min_pae_shape': min_pae_shape, 
            'min_pae_cluster': min_pae_cluster,
            'hotspot_min_pae': round(hotspot_min_pae, 3),
            'hotspot_max_pae': round(hotspot_max_pae, 3),
            'hotspot_weighted_score': round(hotspot_weighted_pae, 3),
            'hotspot_min_pae_size': hotspot_min_pae_size,
            'hotspot_min_pae_size_fraction': hotspot_min_pae_size_fraction,
            'hotspot_min_pae_shape': hotspot_min_pae_shape,
            'hotspot_min_pae_cluster': hotspot_min_pae_cluster,
            'pae_sample': pae_sample
        }
    except Exception as e:
        print(f"Error at index {i}: {e}")

def parallel_process_pae(af2scores, af2pae, contigmap, hotspots, num_cores, kmeans_n_init=1, no_parallel=False, batch_size=None):
    """
    Perform parallel processing of the process_pae function on the given af2scores and pae arrays using multiple cores.
    
    Parameters:
    - af2scores (numpy.ndarray): Array of shape (n, m) representing the af2scores from the '.sc' file when running alphafold initial guess.
    - pae (numpy.ndarray): Array of shape (n,) representing the pae values from the '.pae' file when running alphafold initial guess.
    - contigmap (str): The contigmap parameter from RFdiffusion.
    - hotspots (str): The hotspots parameter from RFdiffusion.
    - num_cores (int): Number of CPU cores to be used for parallel processing.
    
    Returns:
    - final_results (dict): A dictionary containing the following keys:
        - 'pae' (pandas.Series): The minimum mean PAE score from the 8 clusters after KMeans clustering.
        - 'size' (pandas.Series): The cluster size (row x column) for the cluster with the minimum mean PAE.
        - 'shape' (pandas.Series): The shape (row x column) for the cluster with the minimum mean PAE.
        - 'cluster' (pandas.Series): The cluster number for the cluster with the minimum mean PAE.
        - 'pae_description' (pandas.Series): The description for the cluster with the minimum mean PAE.
        - 'max_pae_cluster' (pandas.Series): The maximum mean PAE score from the 8 clusters after KMeans clustering.
        - 'hotspot_pae_cluster' (pandas.Series): The minimum mean PAE score from the 4 clusters after KMeans clustering of hotspot residues.
        - 'hotspot_max_pae_cluster' (pandas.Series): The maximum mean PAE score from the 4 clusters after KMeans clustering of hotspot residues.
        - 'hotspot_size' (pandas.Series): The cluster size (row x column) for the cluster with the minimum mean PAE of hotspot residues.
        - 'hotspot_shape' (pandas.Series): The shape (row x column) for the cluster with the minimum mean PAE of hotspot residues.
        - 'hotspot_cluster' (pandas.Series): The cluster number for the cluster with the minimum mean PAE of hotspot residues.
        - 'kmeans_n_init' (int): Number of KMeans initializations (lower reduces memory)
        - 'batch_size' (int or None): Process this many rows per pool instantiation.
        - 'no_parallel' (bool): If True, disables parallel processing and runs sequentially.
    
    Note:
    - The process_pae function is called in parallel for each row of af2scores and pae.
    """

    indices = list(range(af2scores.shape[0]))
    results = []

    def _consume(batch):
        if no_parallel:
            for idx in batch:
                results.append(process_pae_index(idx))
        else:
            with multiprocessing.Pool(
                processes=num_cores,
                initializer=_init_worker,
                initargs=(af2scores, af2pae, contigmap, hotspots, kmeans_n_init)
            ) as pool:
                for r in pool.imap_unordered(process_pae_index, batch, chunksize=10):
                    results.append(r)

    if batch_size is None:
        _consume(indices)
    else:
        for start in range(0, len(indices), batch_size):
            _consume(indices[start:start + batch_size])

    # Build columns as simple lists (avoid per-row Series overhead)
    def col(name):
        return [r.get(name, np.nan) if r is not None else np.nan for r in results]

    final_results = {
        'pae_cluster': col('min_pae'),
        'pae_weighted': col('weighted_score'),
        'size': col('min_pae_size'),
        'size_fraction': col('min_pae_size_fraction'),
        'shape': col('min_pae_shape'),
        'cluster': col('min_pae_cluster'),
        'pae_description': col('pae_sample'),
        'max_pae_cluster': col('max_pae'),
        'hotspot_pae_cluster': col('hotspot_min_pae'),
        'hotspot_weighted_pae': col('hotspot_weighted_score'),
        'hotspot_max_pae_cluster': col('hotspot_max_pae'),
        'hotspot_size': col('hotspot_min_pae_size'),
        'hotspot_size_fraction': col('hotspot_min_pae_size_fraction'),
        'hotspot_shape': col('hotspot_min_pae_shape'),
        'hotspot_cluster': col('hotspot_min_pae_cluster'),
    }
    return final_results

if __name__ == '__main__':
    #################################
    # Parse Arguments
    #################################

    parser = argparse.ArgumentParser()

    # I/O Arguments
    parser.add_argument( "-score" , type=str, default=None, help='The path of a file of af2-initial guess scores' )
    parser.add_argument( "-pae", type=str, default=None, help='The path of a file of af2-initial guess pae values' )
    parser.add_argument( "-checkpoint", type=str, default=None, help='The path of the checkpoint file' )
    parser.add_argument( "-contigmap", type=str, default=None, help='The RFdiffusion contigmap parameter' )
    parser.add_argument( "-hotspots", type=str, default=None, help='The RFdiffusion hotspots parameter' )
    parser.add_argument( "-num_cores", type=int, default=None, help='The number of CPU cores to be used for parallel processing' )
    parser.add_argument( "-repair_pae", action='store_true', help='Repair the pae file before processing' )
    parser.add_argument('-kmeans_n_init', type=int, default=1, help='Number of KMeans initializations (lower reduces memory)')
    parser.add_argument('-no_parallel', action='store_true', help='Force serial execution to reduce RAM')
    parser.add_argument('-batch_size', type=int, default=None, help='Process this many rows per pool instantiation')

    args = parser.parse_args()

    #################################

    # Find default files
    if args.score is None:
        matching_sc = [f for f in os.listdir() if f.endswith("af2.sc")]
        if len(matching_sc) == 1:
            args.score = matching_sc[0]
            print(f"\nDefault score file found: {args.score} \n")
            continue_sc = input("Continue with default score file? (y/n): \n")
            if continue_sc.lower() == 'n':
                exit()
            elif continue_sc.lower() == 'y':
                pass
        elif len(matching_sc) > 1:
            print("Multiple score files found. Please specify the score file. Exiting. \n")
            exit()
        else:
            print("No score file found. Please specify the score file. Exiting. \n")
            exit()
    if args.pae is None:
        matching_pae = [f for f in os.listdir() if f.endswith("af2.pae")]
        if len(matching_pae) == 1:
            args.pae = matching_pae[0]
            print(f"\nDefault pae file found: {args.pae} \n")
            continue_pae = input("Continue with default PAE file? (y/n): \n")
            if continue_pae.lower() == 'n':
                exit()
            elif continue_pae.lower() == 'y':
                pass
        elif len(matching_pae) > 1:
            print("Multiple PAE files found. Please specify the PAE file. Exiting. \n")
            exit()
        else:
            print("No PAE file found. Please specify the PAE file. Exiting. \n")
            exit()
    if args.checkpoint is None:
        matching_checkpoint = [f for f in os.listdir() if f.endswith("af2_check.point")]
        if len(matching_checkpoint) == 1:
            args.checkpoint = matching_checkpoint[0]
            print(f"\nDefault checkpoint file found: {args.checkpoint} \n")
            continue_checkpoint = input("Continue with default checkpoint file? (y/n): \n")
            if continue_checkpoint.lower() == 'n':
                exit()
            elif continue_checkpoint.lower() == 'y':
                pass
        elif len(matching_checkpoint) > 1:
            print("Multiple checkpoint files found. Please specify the checkpoint file. Exiting. \n")
            exit()
        else:
            print("No checkpoint file found. Please specify the checkpoint file. Exiting. \n")
            exit()

    # Read in the AF2 initial guess score file
    print("Writing backup file \n")
    shutil.copy(args.score, args.score + '.backup')
    print("Read in the AF2 initial guess score file \n")
    af2scores = pd.read_csv(args.score, sep='\s+(?![^()]*\))', engine='python', index_col=False, usecols=range(12))
    if 'description' in af2scores.columns:
        af2scores['description'] = af2scores['description'].astype(str).str.strip()
    if 'binderlen' in af2scores.columns:
        af2scores['binderlen'] = af2scores['binderlen'].astype(str).str.strip()

    # Repair the pae
    if args.repair_pae:
        print("Checking pae file \n")
        repair_pae_script(args.pae)
    
    # Read in the pae file
    chunksize=10**3
    print("Read in the pae file \n")
    csv = pd.read_csv(args.pae, sep = '\s+', header=None, engine='python', nrows=1) 
    if csv.map(lambda cell: any(substring in str(cell) for substring in ['[', ']'])).any().any():
        print("Old pae file format \n")
        pae = pd.DataFrame()
        for chunk in pd.read_csv(args.pae, sep = '\[ |\]|\s+\]', index_col=False, header=None, engine='python', usecols=[1, 2], chunksize=chunksize):
            pae = pd.concat([pae, chunk], ignore_index=True)
    elif csv.map(lambda cell: any(substring in str(cell) for substring in ['tag:'])).any().any():
        print("New pae file format \n")
        pae = pd.DataFrame()
        for chunk in pd.read_csv(args.pae, sep = r'pae:\s+|\s+tag:\s+', index_col=False, header=None, engine='python', chunksize=chunksize):
            pae = pd.concat([pae, chunk], ignore_index=True)
    else:
        raise ValueError("The PAE file does not contain the expected format. \n")

    # KMeans clustering of the pae values in parallel
    print("KMeans clustering is performed on the pae matrix in parallel \n")
    if args.num_cores is not None:
        num_cores = args.num_cores
    else:
        num_cores = multiprocessing.cpu_count()
    print("This may take a while. The number of cores used is :", num_cores, "\n")
    #from af2_initial_guess.pae_clustering import parallel_process_pae, process_pae
    #pae_results = parallel_process_pae(af2scores, pae, args.contigmap, args.hotspots, num_cores)
    pae_results = parallel_process_pae(
        af2scores, pae, args.contigmap, args.hotspots,
        num_cores=args.num_cores if args.num_cores else multiprocessing.cpu_count(),
        kmeans_n_init=args.kmeans_n_init,
        no_parallel=args.no_parallel,
        batch_size=args.batch_size
    )
    
    # Add the pae scores to the af2scores dataframe and write a file for missing pae values
    print('Adding the clustered pae scores to the af2.sc file \n')
    pae_results_df = pd.DataFrame(pae_results)
    merged_df = pd.merge(af2scores, pae_results_df,
                         left_on=['description'], right_on=['pae_description'],
                         how='left')
    merged_df.to_csv(args.score, index=False, sep="\t")
    
    missing_pae = merged_df[merged_df['pae_description'].isna()]['description']  
    if missing_pae.size > 0:
        print('Writing a file for missing pae values to the af2.sc.missing file')
        missing_pae.to_csv(args.score + '.missing', index=False, header=False)
        print(f'Removing the missing pae values from the checkpoint file {args.checkpoint} \n')
        clean_checkpoint(merged_df['pae_description'], args.checkpoint)