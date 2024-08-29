#!/usr/bin/env python3

import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing
import subprocess


def process_pae(i, af2scores, pae):
    """
    Process the PAE (Protein-Antigen Interface Energy) for a given index and performs KMeans clustering.
    
    Args:
        i (int): The index of the PAE to process.
        af2scores (pandas.DataFrame): The DataFrame containing AF2 scores.
        pae (pandas.DataFrame): The DataFrame containing PAE values.
    
    Returns:
        dict: A dictionary containing the following information:
            - 'min_pae' (float): The minimum mean PAE score from the 8 clusters after KMeans clustering.
            - 'min_pae_size' (int): The cluster size (row x column) for the cluster with the minimum mean PAE.
            - 'min_pae_shape' (tuple): The shape (row x column) for the cluster with the minimum mean PAE.
            - 'min_pae_cluster' (str): The cluster number for the cluster with the minimum mean PAE.
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
            complex = pae[[1]].iloc[j].str.split(r'\s+|,\s*', expand=True)
            pae_sample = pae[[2]].iloc[j]
            if pae_sample.isna().any():
                print(f"PAE sample is None at index {i}. Changing to NaN")
                pae_sample = np.nan
                j += 1
            else:
                pae_sample = pae_sample.iloc[0].replace(' ', '')
            if af2scores_sample != pae_sample:
                i += 1
                continue
            break
        
        complex = complex.dropna(how='all', axis=1)
        complex_list = complex.values.flatten().tolist()
        complex_size = int(np.sqrt(len(complex_list)))

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
                    sorted_AB_cluster1 = sorted_AB[kmeans_AB_rows.labels_==0, :][:, kmeans_AB_cols.labels_==0]
                    sorted_AB_cluster1.shape

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
            min_pae_size = pae_score.loc[pae_score['means'].idxmin(), 'size']
            min_pae_shape = pae_score.loc[pae_score['means'].idxmin(), 'shape']
            min_pae_cluster = pae_score.loc[pae_score['means'].idxmin(), 'cluster']

        except Exception as e:
            pae_score = np.nan
            min_pae = np.nan
            min_pae_size = np.nan
            min_pae_shape = np.nan
            min_pae_cluster = np.nan
            print(f"Error calculating PAE score at index {i}: {e}. Setting to NaN")

        return {
            'min_pae': round(min_pae, 3), 
            'min_pae_size': min_pae_size, 
            'min_pae_shape': min_pae_shape, 
            'min_pae_cluster': min_pae_cluster,
            'pae_sample': pae_sample
        }
    except Exception as e:
        print(f"Error at index {i}: {e}")

def parallel_process_pae(af2scores, pae, num_cores):
    """
    Perform parallel processing of the process_pae function on the given af2scores and pae arrays using multiple cores.
    
    Parameters:
    - af2scores (numpy.ndarray): Array of shape (n, m) representing the af2scores from the '.sc' file when running alphafold initial guess.
    - pae (numpy.ndarray): Array of shape (n,) representing the pae values from the '.pae' file when running alphafold initial guess.
    - num_cores (int): Number of CPU cores to be used for parallel processing.
    
    Returns:
    - final_results (dict): A dictionary containing the following keys:
        - 'pae' (pandas.Series): The minimum mean PAE score from the 8 clusters after KMeans clustering.
        - 'size' (pandas.Series): The cluster size (row x column) for the cluster with the minimum mean PAE.
        - 'shape' (pandas.Series): The shape (row x column) for the cluster with the minimum mean PAE.
        - 'cluster' (pandas.Series): The cluster number for the cluster with the minimum mean PAE.
        - 'pae_description' (pandas.Series): The description for the cluster with the minimum mean PAE.
    
    Note:
    - The process_pae function is called in parallel for each row of af2scores and pae.
    """
    with multiprocessing.Pool(processes=num_cores) as pool:
        pae_results = pool.starmap(process_pae, [(i, af2scores, pae) for i in range(af2scores.shape[0])])
        
        # Extract the relevant parts from the dictionaries
        min_pae_list = [pd.Series(result['min_pae']) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_size_list = [pd.Series(result['min_pae_size']) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_shape_list = [pd.Series([tuple(result['min_pae_shape'])]) if result is not None else pd.Series([None]) for result in pae_results]
        min_pae_cluster_list = [pd.Series(result['min_pae_cluster']) if result is not None else pd.Series([None]) for result in pae_results]
        pae_sample = [pd.Series(result['pae_sample']) if result is not None else pd.Series([None]) for result in pae_results]
        
        # Concatenate the extracted parts if they are pandas objects
        min_pae_df = pd.concat(min_pae_list)
        min_pae_size_df = pd.concat(min_pae_size_list)
        min_pae_shape_df = pd.concat(min_pae_shape_list)
        min_pae_cluster_df = pd.concat(min_pae_cluster_list)
        pae_sample = pd.concat(pae_sample)
        
        # Combine the results into a final DataFrame or dictionary
        final_results = {
            'pae_cluster': min_pae_df,
            'size': min_pae_size_df,
            'shape': min_pae_shape_df,
            'cluster': min_pae_cluster_df,
            'pae_description': pae_sample
        }
        
    return final_results

def repair_pae_script(file_path):
    """
    Repairs the PAE script by running the 'repair_pae' command that calls the 'repair_pae.sh' shell script on the specified file.
    Only repairs the file if a string of the format '(\d+(\.\d+)?)pae:' (e.g. '11.3pae:') is found in the file. If a repair is necessary, a backup file is created.

    Parameters:
    - file_path (str): The path to the PAE script file.

    Returns:
    - A '.pae' repaired file and the original file as '.pae.backup' in the same directory as the original file. 
    """
    try:
        result = subprocess.run(['repair_pae', file_path], check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running repair_pae: {e.stderr}")
        
if __name__ == '__main__':

    #################################
    # Parse Arguments
    #################################

    parser = argparse.ArgumentParser()

    # I/O Arguments
    parser.add_argument( "-score", type=str, default="", help='The path of a file of af2-initial guess scores' )
    parser.add_argument( "-pae", type=str, default="", help='The path of a file of af2-initial guess pae values' )
    parser.add_argument( "-backup", type=str, default="af2sc.backup", help='The path of a backup file of af2-initial guess scores' )

    args = parser.parse_args()

    #################################

    # Read in the AF2 initial guess score file
    print("Read in the AF2 initial guess score file \n")
    af2scores = pd.read_csv(args.score, sep='\s+')
    print("Writing backup file \n")
    af2scores.to_csv(args.backup, index=False, sep='\t')

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
            pae = pd.concat([pae, chunk])
    elif csv.map(lambda cell: any(substring in str(cell) for substring in ['tag:'])).any().any():
        print("New pae file format \n")
        pae = pd.DataFrame()
        for chunk in pd.read_csv(args.pae, sep = r'pae: | tag: ', index_col=False, header=None, engine='python', usecols=[1, 2], chunksize=chunksize):
            pae = pd.concat([pae, chunk])
    else:
        raise ValueError("The PAE file does not contain the expected format.")

    # KMeans clustering of the pae values in parallel
    print("KMeans clustering is performed on the pae matrix in parallel \n")
    num_cores = max(multiprocessing.cpu_count() - 2, 1)
    print("This may take a while. The number of cores used is :", num_cores, "\n")
    
    pae_results = parallel_process_pae(af2scores, pae, num_cores)
    
    # Add the pae scores to the af2scores dataframe and write a file for missing pae values
    print('Adding the clustered pae scores to the af2.sc file')
    merged_df = pd.merge(af2scores, pd.DataFrame(pae_results), left_on=['description'], right_on=['pae_description'], how='left')
    merged_df.to_csv(args.score, index=False, sep="\t")
    print('Writing a file for missing pae values to the af2.sc.missing file')
    missing_pae = merged_df[merged_df['pae_description'].isna()]['description']  
    missing_pae.to_csv(args.score + '.missing', index=False, header=False)