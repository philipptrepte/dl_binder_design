#!/usr/bin/env python3

import pandas as pd
import argparse
import numpy as np
import shutil
import subprocess
import os

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument( "-score", type=str, default="", help='The path of a file of af2-initial guess scores' )
parser.add_argument( "-pae", type=str, default="", help='The path of a file of af2-initial guess pae values' )
parser.add_argument( "-silent", type=str, default="af2.silent", help='The path of the af2-initial guess silent file' )
parser.add_argument( "-checkpoint", type=str, default="", help='The path of af2-initial guess checkpoint file which will be overwritten' )
parser.add_argument( "-reference", type=str, default="pae_description", help='Cleaning checkpoint file referencing to the missing pae (`pae_description`) or missing score file (`description`)' )

args = parser.parse_args()

#################################

def clean_checkpoint(ids, af2checkpoint):
    """
    Remove identifiers from checkpoint file that are not in `ids`.

    Parameters:
    - ids (list): A list of ids to be checked against the checkpoint file.
    - af2checkpoint (str): The file path of the checkpoint file.

    Returns:
    - Writes a backup checkpoint file to the same path as the input checkpoint file.
    - Writes cleaned checkpoint file to the same path as the input checkpoint file.

    Prints:
    - The number of missing ids removed from the checkpoint file.
    - The path of the backup checkpoint file.
    - The path of the cleaned checkpoint file.

    """
    # Read in checkpoint file
    checkpoint = pd.read_csv(af2checkpoint, header=0, names=['file'])
    if ids.str.contains('_af2pred').any():
        ids = ids.str.replace('_af2pred', '')

    # Remove ids from checkpoint file not in score file description
    missing_ids = checkpoint['file'].isin(ids)
    if (~missing_ids).any():
        print(f"Removing {(~missing_ids).sum()} missing ids from checkpoint file \n")
        checkpoint_cleaned = checkpoint[missing_ids]

        # Write cleaned checkpoint file
        print(f"Writing backup checkpoint file to {af2checkpoint}" + ".backup \n")
        shutil.copy(af2checkpoint, af2checkpoint + ".backup")
        print(f"Writing cleaned checkpoint file to {af2checkpoint} \n")
        checkpoint_cleaned.to_csv(af2checkpoint, index=False, header=False)

    elif missing_ids.all():
        print("All ids in checkpoint file are in score file description. \n")

def clean_af2score(af2score, af2checkpoint, af2reference='pae_description'):
    """
    Cleans the AF2 initial guess score file by removing duplicates found in the 'af2reference' column and removes the respective identifiers from the 'af2checkpoint' file.

    Parameters:
    - af2score (str): The path to the AF2 initial guess score file.
    - af2checkpoint (str): The path to the checkpoint file.
    - af2reference (str, optional): The column name referencing to the column 'description' or 'pae_description' in the af2score file. Defaults to 'pae_description'.

    Returns:
    - Writes a backup of the score file to the same path as the input score file.
    - Writes the cleaned score file to the same path as the input score file.
    - scores_no_duplicates (DataFrame): The cleaned score file without duplicates in the 'description' column.
    """
    assert af2reference in ['description', 'pae_description'], "af2reference must be either 'description' or 'pae_description'."
    print(f"Read in the AF2 initial guess score file {af2score} \n")
    scores = pd.read_csv(af2score, sep=r'\s+(?![^()]*\))', engine='python', index_col=False)
    if scores.columns.isin([af2reference]).any() == False:
        print(f"Reference column {af2reference} not found in the score file. Setting af2reference to 'description'. \n")
        af2reference = 'description'
    ids = scores[af2reference].str.replace('_af2pred', '')
    # Check for duplicates in the 'description' column
    duplicates = scores['description'].duplicated(keep='last')
    if duplicates.any():
        print("Duplicates found in the 'description' column of the AF2 initial guess score file. Removing duplicates. \n")
        scores_no_duplicates = scores[~duplicates]
        if scores_no_duplicates['description'].duplicated().any():
            print("Duplicates still found in the 'description' column of the AF2 initial guess score file. \n")
            exit(1)
        else:
            print(f"Saving a backup of the score file to {af2score}" + ".backup \n")
            shutil.copy(af2score, af2score + ".backup")
            print(f"Saving score file without duplicates to {af2score} \n")
            scores_no_duplicates.to_csv(af2score, sep="\t", index=False)
            print(f"Removing duplicates from the checkpoint file {af2checkpoint} \n")
            clean_checkpoint(ids, af2checkpoint)
    else:
        print("No duplicates found in the 'description' column of the AF2 initial guess score file. \n")
        scores_no_duplicates = scores

    return scores_no_duplicates

def clean_pae(af2pae, af2checkpoint):
    """
    Cleans the PAE file by removing duplicates and removes the respective identifiers from the af2checkpoint file.
    
    Parameters:
    - pae_file (str): The path to the PAE file.
    - af2checkpoint (str): The path to the checkpoint file.
    - af2reference (str, optional): The reference column name in the PAE file. Defaults to 'pae_description'.
    
    Returns:
    - Writes a backup of the PAE file to the same path as the input PAE file.
    - Writes the cleaned PAE file to the same path as the input PAE file.
    - pae_no_duplicates (DataFrame): The cleaned PAE file without duplicates.
    
    Raises:
    - ValueError: If the PAE file does not contain the expected format.
    """

    chunksize=10**3
    print(f"Read in the pae file {af2pae} \n")
    csv = pd.read_csv(af2pae, sep = '\s+', header=None, engine='python', nrows=1) 
    if csv.map(lambda cell: any(substring in str(cell) for substring in ['[', ']'])).any().any():
        print("Old pae file format detected \n")
        pae = pd.DataFrame()
        for chunk in pd.read_csv(pae_file, sep = '\[ |\]|\s+\]', index_col=False, header=None, engine='python', chunksize=chunksize):
            pae = pd.concat([pae, chunk])
    elif csv.map(lambda cell: any(substring in str(cell) for substring in ['tag:'])).any().any():
        print("New pae file format detected \n")
        pae = pd.DataFrame()
        for chunk in pd.read_csv(pae_file, sep = r'pae: | tag: ', index_col=False, header=None, engine='python', usecols=[1, 2], chunksize=chunksize):
            pae = pd.concat([pae, chunk])
    else:
        raise ValueError("The PAE file does not contain the expected format.")

    duplicates_pae = pae[2].duplicated(keep='last')
    if duplicates_pae.any():
        print("Duplicates found in the second column 'pae_description' of the PAE file. \n")
        print(f"Saving a backup of the pae file {af2pae + '.backup'} \n")
        shutil.copy(af2pae, af2pae + ".backup")
        pae_no_duplicates = pae[~duplicates_pae]
        if pae_no_duplicates[1].duplicated().any():
            print("Duplicates still found in the 'description' column of the PAE file. Exiting. \n")
            exit(1)
        else:
            print(f"Saving pae file without duplicates to {af2pae} \n")
            pae_no_duplicates.insert(0, 'pae', 'pae:')
            pae_no_duplicates.insert(2, 'tag', 'tag:')
            pae_no_duplicates.to_csv(af2pae, sep=' ', index=False, header=False, quotechar=' ', quoting=3, escapechar=' ')
            print(f"Removing duplicates from the checkpoint file {af2checkpoint} \n")
            clean_checkpoint(pae_no_duplicates[2], af2checkpoint)
    else:
        print("No duplicates found in the PAE file. \n")
        pae_no_duplicates = pae
    
    return pae_no_duplicates

def clean_silent(af2silent, af2scores_no_duplicates, pae_no_duplicates, af2checkpoint):
    """
    Cleans the silent file by removing duplicate sequences. Also removes identifiers from the `af2checkpoint` file that are found in the score and pae files, but not in the silent file and vice versa.

    Parameters:
    - af2silent (str): Path to the input silent file.
    - af2scores_no_duplicates (pd.DataFrame): DataFrame containing the score file with no duplicate sequences.
    - pae_no_duplicates (pd.DataFrame): DataFrame containing the pae file with no duplicate sequences.
    - af2checkpoint (str): Path to the checkpoint file.
    
    Returns:
    - Writes a backup of the silent file to the same path as the input silent file.
    - Writes the cleaned silent file to the same path as the input silent file.
    """
    silent_path = os.path.dirname(af2silent)
    if not silent_path:
        silent_path = os.getcwd()
    silent_file = os.path.basename(af2silent)
    print(f"Read in the af2.silen file {silent_path}/{silent_file} \n")
    result = subprocess.run(["silentls", silent_file], capture_output=True, cwd=silent_path)
    if result.returncode == 0:
        output = result.stdout.decode('utf-8')
        lines = output.split('\n')
        lines_series = pd.Series(lines).str.replace('af2pred_1', 'af2pred')
        lines_series = lines_series[lines_series != '']
        # Check for duplicates in the silent file
        duplicates_silent = lines_series.duplicated(keep='last')
        if duplicates_silent.sum() > 0:
            print("Duplicates found in the silent file. \n")
            print(f"Writing backup silent file to {af2silent} + .backup \n")
            shutil.copy(af2silent, af2silent + '.backup')
            print("Removing duplicates from silent file \n")
            subprocess.run(f"silentdropduplicatesequences {silent_file} > {silent_file + '.unique'}", shell=True, cwd=silent_path)
            subprocess.run(f"mv {silent_file + '.unique'} {silent_file}", shell=True, cwd=silent_path)
            subprocess.run(f"mv {silent_file + '.unique.idx'} {silent_file + '.idx'}", shell=True, cwd=silent_path)
            result_unique = subprocess.run(["silentls", silent_file], capture_output=True, cwd=silent_path)
            if result_unique.returncode == 0:
                output_unique = result_unique.stdout.decode('utf-8')
                lines_unique = output_unique.split('\n')
                lines_unique_series = pd.Series(lines_unique).str.replace('af2pred_1', 'af2pred')
                lines_unique_series = lines_unique_series[lines_unique_series != '']
                duplicates_silent_unique = lines_unique_series.duplicated(keep='last')
                if duplicates_silent_unique.sum() > 0:
                    print("Duplicates still found in the silent file after removing duplicates. Exiting. \n")
                    exit(1)
                else:
                    print("No duplicates found in the silent file after removing duplicates. \n")
            else:
                print("Error executing silentls command:", result_unique.stderr.decode('utf-8'))
            
            silent_missing = pd.Series()
            if lines_unique_series.isin(af2scores_no_duplicates['description']).all():
                #print("All sequences in the silent file are in the score file. \n")
                next
            else:
                silent_missing = pd.concat([silent_missing, lines_unique_series[~lines_unique_series.isin(af2scores_no_duplicates['description'])]])
            if lines_unique_series.isin(pae_no_duplicates[2]).all():
                #print("All sequences in the silent file are in the pae file. \n")
                next
            else:
                silent_missing = pd.concat([silent_missing, lines_unique_series[~lines_unique_series.isin(pae_no_duplicates[2])]])
            if af2scores_no_duplicates['description'].isin(lines_unique_series).all():
                #print("All sequences in the score file are in the silent file. \n")
                next
            else:
                af2scores_missing = af2scores_no_duplicates[~af2scores_no_duplicates['description'].isin(lines_unique_series)]['description']
                silent_missing = pd.concat([silent_missing, af2scores_missing])
            if pae_no_duplicates[2].isin(lines_unique_series).all():
                #print("All sequences in the pae file are in the silent file. \n")
                next
            else:
                pae_missing = pae_no_duplicates[2][~pae_no_duplicates[2].isin(lines_unique_series)]
                silent_missing = pd.concat([silent_missing, pae_missing])

            if silent_missing.empty:
                print("All sequences in the silent file are in the score and pae files and vice versa. \n")
            else:
                print("Identifiers found in the AF2 score or PAE files that are not in the silent file. Removing from checkpoint \n")
                silent_ids = pd.Series(lines_unique_series)
                clean_checkpoint(silent_ids, af2checkpoint)

        elif duplicates_silent.sum() == 0:
            print("No duplicates found in the silent file. \n")

    else:
        print("Error executing silentls command:", result.stderr.decode('utf-8'))

if __name__ == '__main__':
    print('\n########################################################')
    print("### Cleaning duplicates from AF2 initial guess files ###")
    print('######################################################## \n')
    
    af2scores_no_duplicates = clean_af2score(args.score, args.checkpoint, args.reference)
    
    print('\n#########################################')
    print("### Cleaning duplicates from PAE file ###")
    print('######################################### \n')

    pae_no_duplicates = clean_pae(args.pae, args.checkpoint)
    
    print('\n############################################')
    print("### Cleaning duplicates from silent file ###")
    print('############################################ \n')
    
    clean_silent(args.silent, af2scores_no_duplicates, pae_no_duplicates, args.checkpoint)


    