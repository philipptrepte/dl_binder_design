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
        j = i
        while True:
            binderlen = af2scores[['binderlen']].iloc[i]
            af2scores_sample = af2scores[['description']].iloc[i]
            af2scores_sample = af2scores_sample.iloc[0]
            if af2scores_sample is None:
                print(f"AF2 sample is None at index {i}. Changing to NaN")
                af2scores_sample = np.nan
                i += 1
            binderlen = int(binderlen.iloc[0])
            try: 
                j = (af2pae[[2]] == af2scores_sample).idxmax().values[0]
            except Exception as e:
                print(f"PAE sample not found in PAE file at index {i}. Changing to NaN")
                complex = np.nan
            complex = af2pae[[1]].iloc[j].str.split(r'\s+|,\s*', expand=True)
            pae_sample = af2pae[[2]].iloc[j]
            if pae_sample.isna().any():
                print(f"PAE sample is None at index {i}. Changing to NaN")
                pae_sample = np.nan
            else:
                pae_sample = pae_sample.iloc[0].replace(' ', '')
            if af2scores_sample != pae_sample:
                i += 1
                continue
            break
        
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
                kmeans_AB_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_AB)
            except Exception as e:
                print(f"KMeans_AB_rows failed at index {i}: {e}")
                kmeans_AB_rows = np.nan
            try:
                kmeans_AB_cols = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_AB_transpose)
            except Exception as e:
                print(f"KMeans_AB_cols failed at index {i}: {e}")
                kmeans_AB_cols = np.nan
            try:
                kmeans_BA_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_BA)
            except Exception as e:
                print(f"KMeans_BA_rows failed at index {i}: {e}")
                kmeans_BA_rows = np.nan
            try:
                kmeans_BA_cols = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_BA_transpose)
            except Exception as e:
                print(f"KMeans_BA_cols failed at index {i}: {e}")
                kmeans_BA_cols = np.nan

            # Check if KMeans clustering gives a result
            if (kmeans_AB_rows is not None and kmeans_AB_cols is not None and
                np.unique(kmeans_AB_rows.labels_).size > 1 and np.unique(kmeans_AB_cols.labels_).size > 1):
                try:
                    sorted_AB = binder_AB[kmeans_AB_rows.labels_.argsort(), :]
                    sorted_AB = binder_AB[:, kmeans_AB_cols.labels_.argsort()]
                    sorted_AB_cluster1 = sorted_AB[kmeans_AB_rows.labels_==0, :][:, kmeans_AB_cols.labels_==0]
                    sorted_AB_cluster2 = sorted_AB[kmeans_AB_rows.labels_==1, :][:, kmeans_AB_cols.labels_==1]
                    sorted_AB_cluster3 = sorted_AB[kmeans_AB_rows.labels_==0, :][:, kmeans_AB_cols.labels_==1]
                    sorted_AB_cluster4 = sorted_AB[kmeans_AB_rows.labels_==1, :][:, kmeans_AB_cols.labels_==0]
                    mean_AB = pd.DataFrame([np.mean(sorted_AB_cluster1.astype(np.float64)), 
                                        np.mean(sorted_AB_cluster2.astype(np.float64)),
                                        np.mean(sorted_AB_cluster3.astype(np.float64)),
                                        np.mean(sorted_AB_cluster4.astype(np.float64))], columns = ['means'])
                    mean_AB['cluster'] = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4']
                    mean_AB['size'] = [sorted_AB_cluster1.size, sorted_AB_cluster2.size, sorted_AB_cluster3.size, sorted_AB_cluster4.size]
                    mean_AB['shape'] = [sorted_AB_cluster1.shape, sorted_AB_cluster2.shape, sorted_AB_cluster3.shape, sorted_AB_cluster4.shape]
                except Exception as e:
                    print(f"Error calculating mean PAE for clustered AB interface at index {i}: {e}")
            else:
                try:
                    mean_AB = np.mean(binder_AB.astype(np.float64))
                    mean_AB['cluster'] = 'KMeans_failed'
                    mean_AB['size'] = binder_AB.size
                    mean_AB['shape'] = binder_AB.shape
                except Exception as e:
                    print(f"Error calculating mean PAE for AB interface at index {i}: {e}")

            if (kmeans_BA_rows is not None and kmeans_BA_cols is not None and
                np.unique(kmeans_BA_rows.labels_).size > 1 and np.unique(kmeans_BA_cols.labels_).size > 1):
                try:
                    sorted_BA = binder_BA[:, kmeans_BA_cols.labels_.argsort()]
                    sorted_BA = binder_BA[kmeans_BA_rows.labels_.argsort(), :]
                    sorted_BA_cluster5 = sorted_BA[kmeans_BA_rows.labels_==0, :][:, kmeans_BA_cols.labels_==0]
                    sorted_BA_cluster6 = sorted_BA[kmeans_BA_rows.labels_==1, :][:, kmeans_BA_cols.labels_==1]
                    sorted_BA_cluster7 = sorted_BA[kmeans_BA_rows.labels_==0, :][:, kmeans_BA_cols.labels_==1]
                    sorted_BA_cluster8 = sorted_BA[kmeans_BA_rows.labels_==1, :][:, kmeans_BA_cols.labels_==0]
                    mean_BA = pd.DataFrame([np.mean(sorted_BA_cluster5.astype(np.float64)), 
                                        np.mean(sorted_BA_cluster6.astype(np.float64)),
                                        np.mean(sorted_BA_cluster7.astype(np.float64)),
                                        np.mean(sorted_BA_cluster8.astype(np.float64))], columns = ['means'])
                    mean_BA['cluster'] = ['Cluster_5', 'Cluster_6', 'Cluster_7', 'Cluster_8']
                    mean_BA['size'] = [sorted_BA_cluster5.size, sorted_BA_cluster6.size, sorted_BA_cluster7.size, sorted_BA_cluster8.size]
                    mean_BA['shape'] = [sorted_BA_cluster5.shape, sorted_BA_cluster6.shape, sorted_BA_cluster7.shape, sorted_BA_cluster8.shape]
                except Exception as e:
                    print(f"Error calculating mean PAE for clustered BA interface at index {i}: {e}")
            else:
                try:
                    mean_BA = np.mean(binder_BA.astype(np.float64))
                    mean_BA['cluster'] = 'KMeans_failed'
                    mean_BA['size'] = binder_BA.size
                    mean_BA['shape'] = binder_BA.shape
                except Exception as e:
                    print(f"Error calculating mean PAE for BA interface at index {i}: {e}")

            pae_score = pd.concat([mean_AB, mean_BA], axis=0).reset_index(drop=True)
            min_pae = np.min(pae_score['means'].astype(np.float64))
            max_pae = np.max(pae_score['means'].astype(np.float64))
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
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")
            else:
                try:
                    mean_hotspot_0 = np.mean(hotspot_matrix_0.astype(np.float64))
                    mean_hotspot_0['cluster'] = 'KMeans_failed'
                    mean_hotspot_0['size'] = hotspot_matrix_0.size
                    mean_hotspot_0['shape'] = hotspot_matrix_0.shape
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")

            if hotspot_cluster3 is not None and hotspot_cluster4 is not None:
                try:
                    mean_hotspot_1 = pd.DataFrame([np.mean(hotspot_cluster3.astype(np.float64)), 
                                        np.mean(hotspot_cluster4.astype(np.float64))], columns = ['means'])
                    mean_hotspot_1['cluster'] = ['Cluster_3', 'Cluster_4']
                    mean_hotspot_1['size'] = [hotspot_cluster3.size, hotspot_cluster4.size]
                    mean_hotspot_1['shape'] = [hotspot_cluster3.shape, hotspot_cluster4.shape]
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")
            else:
                try:
                    mean_hotspot_1 = np.mean(hotspot_matrix_1.astype(np.float64))
                    mean_hotspot_1['cluster'] = 'KMeans_failed'
                    mean_hotspot_1['size'] = hotspot_matrix_1.size
                    mean_hotspot_1['shape'] = hotspot_matrix_1.shape
                except Exception as e:
                    print(f"Error calculating mean PAE for hotspot residues at index {i}: {e}")

            hotspot_pae_score = pd.concat([mean_hotspot_0, mean_hotspot_1], axis=0).reset_index(drop=True)
            hotspot_min_pae = np.min(hotspot_pae_score['means'].astype(np.float64))
            hotspot_max_pae = np.max(hotspot_pae_score['means'].astype(np.float64))
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
            'min_pae_size': min_pae_size, 
            'min_pae_size_fraction': min_pae_size_fraction,
            'min_pae_shape': min_pae_shape, 
            'min_pae_cluster': min_pae_cluster,
            'hotspot_min_pae': round(hotspot_min_pae, 3),
            'hotspot_max_pae': round(hotspot_max_pae, 3),
            'hotspot_min_pae_size': hotspot_min_pae_size,
            'hotspot_min_pae_size_fraction': hotspot_min_pae_size_fraction,
            'hotspot_min_pae_shape': hotspot_min_pae_shape,
            'hotspot_min_pae_cluster': hotspot_min_pae_cluster,
            'pae_sample': pae_sample
        }
    except Exception as e:
        print(f"Error at index {i}: {e}")

def parallel_process_pae(af2scores, af2pae, contigmap, hotspots, num_cores):
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
    
    Note:
    - The process_pae function is called in parallel for each row of af2scores and pae.
    """
    with multiprocessing.Pool(processes=num_cores) as pool:
        pae_results = pool.starmap(process_pae, [(i, af2scores, af2pae, contigmap, hotspots) for i in range(af2scores.shape[0])])
        
        # Extract the relevant parts from the dictionaries
        min_pae_list = [pd.Series(result['min_pae']) if result is not None else pd.Series([None]) for result in pae_results]
        max_pae_list = [pd.Series(result['max_pae']) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_size_list = [pd.Series(result['min_pae_size']) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_size_fraction_list = [pd.Series(result['min_pae_size_fraction']) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_shape_list = [pd.Series([tuple(result['min_pae_shape'])]) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_cluster_list = [pd.Series(result['min_pae_cluster']) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_min_pae_list = [pd.Series(result['hotspot_min_pae']) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_max_pae_list = [pd.Series(result['hotspot_max_pae']) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_min_pae_size_list = [pd.Series(result['hotspot_min_pae_size']) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_min_pae_size_fraction_list = [pd.Series(result['hotspot_min_pae_size_fraction']) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_min_pae_shape_list = [pd.Series([tuple(result['hotspot_min_pae_shape'])]) if result is not None else pd.Series([None]) for result in pae_results]
        hotspot_min_pae_cluster_list = [pd.Series(result['hotspot_min_pae_cluster']) if result is not None else pd.Series([None]) for result in pae_results]
        pae_sample = [pd.Series(result['pae_sample']) if result is not None else pd.Series([None]) for result in pae_results]
        
        # Concatenate the extracted parts if they are pandas objects
        min_pae_df = pd.concat(min_pae_list)
        max_pae_df = pd.concat(max_pae_list)
        min_pae_size_df = pd.concat(min_pae_size_list)
        min_pae_size_fraction_df = pd.concat(min_pae_size_fraction_list)
        min_pae_shape_df = pd.concat(min_pae_shape_list)
        min_pae_cluster_df = pd.concat(min_pae_cluster_list)
        hotspot_min_pae_df = pd.concat(hotspot_min_pae_list)
        hotspot_max_pae_df = pd.concat(hotspot_max_pae_list)
        hotspot_min_pae_size_df = pd.concat(hotspot_min_pae_size_list)
        hotspot_min_pae_size_fraction_df = pd.concat(hotspot_min_pae_size_fraction_list)
        hotspot_min_pae_shape_df = pd.concat(hotspot_min_pae_shape_list)
        hotspot_min_pae_cluster_df = pd.concat(hotspot_min_pae_cluster_list)
        pae_sample = pd.concat(pae_sample)
        
        # Combine the results into a final DataFrame or dictionary
        final_results = {
            'pae_cluster': min_pae_df,
            'size': min_pae_size_df,
            'size_fraction': min_pae_size_fraction_df,
            'shape': min_pae_shape_df,
            'cluster': min_pae_cluster_df,
            'pae_description': pae_sample,
            'max_pae_cluster': max_pae_df,
            'hotspot_pae_cluster': hotspot_min_pae_df,
            'hotspot_max_pae_cluster': hotspot_max_pae_df,
            'hotspot_size': hotspot_min_pae_size_df,
            'hotspot_size_fraction': hotspot_min_pae_size_fraction_df,
            'hotspot_shape': hotspot_min_pae_shape_df,
            'hotspot_cluster': hotspot_min_pae_cluster_df
        }
        
    return final_results

if __name__ == '__main__':
    #################################
    # Parse Arguments
    #################################

    parser = argparse.ArgumentParser()

    # I/O Arguments
    parser.add_argument( "-score", type=str, default=None, help='The path of a file of af2-initial guess scores' )
    parser.add_argument( "-pae", type=str, default=None, help='The path of a file of af2-initial guess pae values' )
    parser.add_argument( "-checkpoint", type=str, default=None, help='The path of the checkpoint file' )
    parser.add_argument( "-contigmap", type=str, default=None, help='The RFdiffusion contigmap parameter' )
    parser.add_argument( "-hotspots", type=str, default=None, help='The RFdiffusion hotspots parameter' )

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

    # Repair the pae
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
    num_cores = max(multiprocessing.cpu_count() - 2, 1)
    print("This may take a while. The number of cores used is :", num_cores, "\n")
    #from af2_initial_guess.pae_clustering import parallel_process_pae, process_pae
    pae_results = parallel_process_pae(af2scores, pae, args.contigmap, args.hotspots, num_cores)
    
    # Add the pae scores to the af2scores dataframe and write a file for missing pae values
    print('Adding the clustered pae scores to the af2.sc file \n')
    merged_df = pd.merge(af2scores, pd.DataFrame(pae_results), left_on=['description'], right_on=['pae_description'], how='left')
    merged_df.to_csv(args.score, index=False, sep="\t")
    
    missing_pae = merged_df[merged_df['pae_description'].isna()]['description']  
    if missing_pae.size > 0:
        print('Writing a file for missing pae values to the af2.sc.missing file')
        missing_pae.to_csv(args.score + '.missing', index=False, header=False)
        print(f'Removing the missing pae values from the checkpoint file {args.checkpoint} \n')
        clean_checkpoint(merged_df['pae_description'], args.checkpoint)