import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import warnings
from sklearn.cluster import KMeans
import os
import re
import py3Dmol
from Bio.PDB import PDBParser
#os.chdir(os.path.dirname(os.path.realpath('__file__')))
warnings.filterwarnings('ignore', category=UserWarning)


def pae_heatmap(selected_binder, pae, af2scores):
    """
    Generate a heatmap visualization of the PAE (Positional Amino Acid Enrichment) for a selected binder.
    
    Args:
        selected_binder (str): The description of the binder to visualize.
        pae (DataFrame): The PAE data containing the binder information.
        af2scores (DataFrame): The af2scores data containing the binder length information.

    Returns:
    matplotlib figure: A figure containing the heatmap visualization of the PAE for the selected binder.
    
    """

    binderlen = af2scores[af2scores['description'] == selected_binder]['binderlen']
    binderlen = int(binderlen.iloc[0])

    pae = pae[pae[2].str.strip()==selected_binder]

    complex = pae[[1]].iloc[0].str.split(',\s+|\s+', expand=True)
    complex = complex.dropna(how='all', axis=1)
    complex_list = complex.values.flatten().tolist()
    complex_size = int(np.sqrt(len(complex_list)))
    complex_matrix = np.reshape(complex_list, (complex_size, complex_size))
    targetlen = complex_size - binderlen

    #extract pae of binder
    binder_AB = complex_matrix[0:binderlen, binderlen:]
    binder_AB_transpose = np.transpose(binder_AB)
    binder_BA = complex_matrix[binderlen:, 0:binderlen]
    binder_BA_transpose = np.transpose(binder_BA)
    #perform kmeans clustering
    kmeans_AB_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_AB)
    interAB_hline = sum(kmeans_AB_rows.labels_ == 0)
    kmeans_AB_cols = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_AB_transpose)
    interAB_vline = sum(kmeans_AB_cols.labels_ == 0)
    kmeans_BA_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_BA)
    interBA_hline = sum(kmeans_BA_rows.labels_ == 0)
    kmeans_BA_cols = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(binder_BA_transpose)
    interBA_vline = sum(kmeans_BA_cols.labels_ == 0)

    # KMeans clustering gave a result
    sorted_AB = binder_AB[kmeans_AB_rows.labels_.argsort(), :]
    sorted_AB = binder_AB[:, kmeans_AB_cols.labels_.argsort()]
    
    sorted_BA = binder_BA[:, kmeans_BA_cols.labels_.argsort()]
    sorted_BA = binder_BA[kmeans_BA_rows.labels_.argsort(), :]
    # Define the heatmap colors and create the colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    fig.suptitle(f'PAE heatmap and KMeans clustering of inter-protein distances for {selected_binder}')
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    plt.subplots_adjust(wspace=1, hspace=1)

    sns.heatmap(complex_matrix.astype(np.float64), cmap=cmap, vmin=0, vmax=35, ax=ax1)
    ax1.set_xlabel('amino acid')
    ax1.set_ylabel('amino acid')
    ax1.text(x = (binderlen / 2), y = (binderlen / 2), s = selected_binder, ha = 'center', va = 'center', fontsize = 12, color='white')
    ax1.text(x = binderlen + (targetlen / 2), y = (binderlen / 2), s = 'inter AB', ha = 'center', va = 'center', fontsize = 12, color='white')
    ax1.text(x = binderlen + (targetlen / 2), y = binderlen + (targetlen / 2), s = 'target', ha = 'center', va = 'center', fontsize = 12, color='white')
    ax1.text(x = (binderlen / 2), y = binderlen + (targetlen / 2), s = 'inter BA', ha = 'center', va = 'center', fontsize = 12, color='white')
    ax1.axvline(x=binderlen, color='black', linestyle='-')
    ax1.axhline(y=binderlen, color='black', linestyle='-')
    ax1_ticks = np.arange(0, complex_size + 1, 25)
    ax1.set_xticks(ax1_ticks)
    ax1.set_xticklabels(ax1_ticks)
    ax1.set_yticks(ax1_ticks)
    ax1.set_yticklabels(ax1_ticks)
    

    sns.heatmap(binder_AB.astype(np.float64), cmap=cmap, ax=ax2, vmin=0, vmax=35)
    ax2.set_title('inter_AB')
    ax2.set_xlabel('amino acid')
    ax2.set_ylabel('amino acid')

    sns.heatmap(sorted_AB.astype(np.float64), cmap=cmap, ax=ax3, vmin=0, vmax=35)
    ax3.set_title('clustered inter_AB')
    ax3.axvline(x=interAB_vline, color='black', linestyle='-')
    ax3.axhline(y=interAB_hline, color='black', linestyle='-')
    ax3.set_xlabel('amino acid')
    ax3.set_ylabel('amino acid')

    sns.heatmap(binder_BA.astype(np.float64), cmap=cmap, ax=ax4, vmin=0, vmax=35)
    ax4.set_title('inter_BA')
    ax4.set_xlabel('amino acid')
    ax4.set_ylabel('amino acid')

    sns.heatmap(sorted_BA.astype(np.float64), cmap=cmap, ax=ax5, vmin=0, vmax=35)
    ax5.set_title('clustered inter_BA')
    ax5.axvline(x=interBA_vline, color='black', linestyle='-')
    ax5.axhline(y=interBA_hline, color='black', linestyle='-')
    ax5.set_xlabel('amino acid')
    ax5.set_ylabel('amino acid')

    x_ticks = np.arange(0, targetlen + 1, 25)
    y_ticks = np.arange(0, binderlen + 1, 25)
    for ax in [ax2, ax3]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
    for ax in [ax4, ax5]:
        ax.set_xticks(y_ticks)
        ax.set_xticklabels(y_ticks)
        ax.set_yticks(x_ticks)
        ax.set_yticklabels(x_ticks)

    plt.show()

def pae_heatmap_hotspots(selected_binder, pae, af2scores, contigmap, hotspots):
    """
    Generate a heatmap visualization of the PAE (Positional Amino Acid Enrichment) for a selected binder.
    
    Args:
        selected_binder (str): The description of the binder to visualize.
        pae (DataFrame): The PAE data containing the binder information.
        af2scores (DataFrame): The af2scores data containing the binder length information.

    Returns:
    matplotlib figure: A figure containing the heatmap visualization of the PAE for the selected binder.
    
    """

    binderlen = af2scores[af2scores['description'] == selected_binder]['binderlen']
    binderlen = int(binderlen.iloc[0])

    # Define the contigmap string
    ranges = re.findall(r'[AB](\d+)-(\d+)/0', contigmap)
    # Convert ranges to individual numbers and combine them into one list
    numbers = []
    for start, end in ranges:
            numbers.extend(range(int(start), int(end) + 1))
    # Define the hotspot string
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

    selected_pae = pae[pae[2].str.strip()==selected_binder]
    complex = selected_pae[[1]].iloc[0].str.split(',\s+|\s+', expand=True)
    complex = complex.dropna(how='all', axis=1)
    complex_list = complex.values.flatten().tolist()
    complex_size = int(np.sqrt(len(complex_list)))
    complex_matrix = np.reshape(complex_list, (complex_size, complex_size))
    targetlen = complex_size - binderlen

    #extract pae of binder
    hotspot_matrix_0 = complex_matrix[residues, 0:binderlen]
    hotspot_matrix_1 = complex_matrix[0:binderlen, residues]

    #perform kmeans clustering
    try:
        hotspot_cluster_0_rows = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(np.transpose(hotspot_matrix_0))
        hotspot_cluster_1_cols = KMeans(n_clusters=2, random_state=0, n_init='auto').fit((hotspot_matrix_1))
    except:
        print('KMeans clustering failed')

    # KMeans clustering gave a result
    if hotspot_cluster_0_rows is not None and hotspot_cluster_1_cols is not None:
        hotspot_matrix_0 = hotspot_matrix_0[:, hotspot_cluster_0_rows.labels_.argsort()]
        hotspot_matrix_1 = hotspot_matrix_1[hotspot_cluster_1_cols.labels_.argsort(), :]
    else:
        hotspot_matrix_0 = hotspot_matrix_0
        hotspot_matrix_1 = hotspot_matrix_1

    # Define the heatmap colors and create the colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    fig.suptitle(f'Hotspots distances for {selected_binder}')
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.heatmap(hotspot_matrix_0.astype(np.float64), cmap=cmap, vmin=0, vmax=35, ax=ax1)
    ax1.set_xlabel('amino acid')
    ax1.set_ylabel('amino acid')
    ax1.set_yticks(np.array(range(len(hotspots))) + 0.5)
    ax1.set_yticklabels(hotspots)
    if hotspot_cluster_0_rows is not None:
        ax1.axvline(x=sum(hotspot_cluster_0_rows.labels_ == 0), color='black', linestyle='-')



    sns.heatmap(hotspot_matrix_1.astype(np.float64), cmap=cmap, vmin=0, vmax=35, ax=ax2)
    ax2.set_xlabel('amino acid')
    ax2.set_ylabel('amino acid')
    ax2.set_xticks(np.array(range(len(hotspots))) + 0.5)
    ax2.set_xticklabels(hotspots)
    if hotspot_cluster_1_cols is not None:
        ax2.axhline(y=sum(hotspot_cluster_1_cols.labels_ == 0), color='black', linestyle='-')

    plt.show()

def plot_binder(file, selected_binder, contigmap, hotspots):
    # Create a 3D viewer
    print(file)
    
    # Parse the PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(file=file, id=selected_binder)
    # Get the length of chain A
    chain_A = structure[0]['A']
    length_of_chain_A = len(chain_A)

    # Define the contigmap string
    contigmap = contigmap
    ranges = re.findall(r'[AB](\d+)-(\d+)/0', contigmap)

    # Convert ranges to individual numbers and combine them into one list
    numbers = []
    for start, end in ranges:
            numbers.extend(range(int(start), int(end) + 1))

    # Define the hotspot string
    hotspots = hotspots
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

    residues = [hotspot + length_of_chain_A + 1 for hotspot in hotspot]

    p = py3Dmol.view(query=file, height=400, width=400)

    # Set the color of chain A to orange
    p.setStyle({'chain': 'A'}, {'cartoon': {'color': 'orange'}})
    
    # Set the color of chain B to cyan
    p.setStyle({'chain': 'B'}, {'cartoon': {'color': 'cyan'}})
    # Set the color of specific amino acid numbers
    p.addStyle({'resi': residues}, {'stick': {'color': 'red'}})

    # Show the 3D viewer
    p.show()

def barplot_scores(af2scores, selected_binder):
    if af2scores.columns.isin(['pae_description']).any():
        selected_binder_data = af2scores[af2scores['pae_description'] == selected_binder]
    elif af2scores.columns.isin(['description']).any():
        selected_binder_data = af2scores[af2scores['description'] == selected_binder]
    selected_binder_data = selected_binder_data.select_dtypes(include=[np.number])

    # Melt the dataframe to convert it into long format
    melted_data = selected_binder_data.melt(var_name='Column', value_name='Value')
    melted_data['Value'] = melted_data['Value'].replace(np.nan, 0)

    # Create the barplot
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Scores for {selected_binder}')
    sns.barplot(data=melted_data, x='Value', y='Column', hue='Value')

    # Display the plot
    plt.show()

def pae_monomer_heatmap(selected_monomer, af2pae):
    
    if af2pae.columns.isin(['description']).any():
        pae = af2pae[af2pae['description'].str.strip()==selected_monomer]
        complex = pae[['PAE']].iloc[0].str.split(',\s+|\s+', expand=True)
    else:
        pae = af2pae[af2pae[2].str.strip()==selected_monomer]
        complex = pae[[1]].iloc[0].str.split(',\s+|\s+', expand=True)
    complex = complex.dropna(how='all', axis=1)
    complex_list = complex.values.flatten().tolist()
    complex_size = int(np.sqrt(len(complex_list)))
    complex_matrix = np.reshape(complex_list, (complex_size, complex_size))
    
    # Define the heatmap colors and create the colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    plt_ticks = np.arange(0, complex_size + 1, 25)
    
    sns.heatmap(complex_matrix.astype(np.float64), cmap=cmap, vmin=0, vmax=35)
    plt.xlabel('amino acid')
    plt.ylabel('amino acid')
    plt.text(x = (complex_size / 2), y = (complex_size / 2), s = selected_monomer, ha = 'center', va = 'center', fontsize = 12, color='white')
    plt.xticks(ticks = plt_ticks, labels = plt_ticks)
    plt.yticks(ticks = plt_ticks, labels = plt_ticks)
    
    plt.show()

if __name__ == '__main__':
    pae_heatmap(selected_binder, pae, af2scores)