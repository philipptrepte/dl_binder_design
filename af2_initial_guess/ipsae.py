# ipsae.py modified from https://github.com/DunbrackLab/IPSAE for use with AlphaFold2 with the "initial guess" modifications from https://github.com/nrbennet/dl_binder_design
# script for calculating the ipSAE score for scoring pairwise protein-protein interactions in AlphaFold2 and AlphaFold3 models
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

# Also calculates:
#    pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
#    pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
#    LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

# Roland Dunbrack
# Fox Chase Cancer Center
# version 3
# April 6, 2025
# MIT license: script can be modified and redistributed for non-commercial and commercial use, as long as this information is reproduced.

# Modified by Philipp Trepte
# Qanatpharma AG
# Alter Postplatz 2
# 6370 Stans, Switzerland

# It may be necessary to install numpy with the following command:
#      pip install numpy

# Usage:

#  python ipsae.py <path_to_af2_pae_file>     <path_to_af2_pdb_file>     <pae_cutoff> <dist_cutoff>     [<tag1> <tag2> ...]
#
# All output files will be in same path/folder as cif or pdb file

import sys, os
import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm
import datetime
np.set_printoptions(threshold=np.inf)  # for printing out full numpy arrays for debugging


# Input and output files and parameters

# Ensure correct usage
if len(sys.argv) < 5:
    print("Usage for AF2 with initial guess:")
    print("   python ipsae.py <path_to_pae_file> <path_to_silent_file> <pae_cutoff> <dist_cutoff> Optional:[<tag1> <tag2> ...]")
    print("   python ipsae.py out_af2.pae out_af2.silent 10 10")
    sys.exit(1)

pae_file_path =    sys.argv[1]
silent_file_path = sys.argv[2]
pae_cutoff =       float(sys.argv[3])
dist_cutoff =      float(sys.argv[4])
if len(sys.argv) > 5:
    tags = list(sys.argv[5:])
else:
    tags = None
pae_string =       str(int(pae_cutoff))
if pae_cutoff<10:  pae_string="0"+pae_string
dist_string =      str(int(dist_cutoff))
if dist_cutoff<10: dist_string="0"+dist_string

#pae_AURKA_TPX2_model_0.npz

if ".silent" in silent_file_path:
    silent_stem=silent_file_path.replace(".silent","")
    path_stem =     f'{silent_file_path.replace(".silent","")}_pae_{pae_string}_dist_{dist_string}'
else:
    print("Wrong file type ", silent_file_path)
    sys.exit()
    
file_path =        path_stem + "_IPSAE.txt"
#file2_path =       path_stem + "_IPSAE_byres.txt"
pml_path =         path_stem + "_IPSAE.pml"
OUT =              open(file_path,'w')
PML =              open(pml_path,'w')
#OUT2 =             open(file2_path,'w')

# Extract the pdb files from the .silent file using silent tools
cpus = max(os.cpu_count()+4, 4)
silent_path = Path(silent_file_path)
silent_dir = silent_path.parent
silent_name = silent_path.name
# Set locale to C to avoid issues with silentextract for MacOS
env = os.environ.copy()
env["LC_ALL"] = "C"
# or: env["LC_ALL"] = "en_US.UTF-8"

if not os.path.exists(Path(silent_dir).joinpath("AF2")):
    print("Extracting PDB files from silent file...")
    command = f'mkdir -p AF2 && cd AF2 && silentextract -j {cpus} ../"{silent_name}"'
    subprocess.run(command, shell=True, check=True, env=env, cwd=silent_dir)
else:
    print("AF2 folder already exisits. PDB files already extracted.")

# Definte a pae extraction function
def extract_pae_by_tags(file_path, tags_to_extract):
    """
    Extract lines from a .pae file that match specified tags.

    Args:
        file_path (str): The path to the .pae file.
        tags_to_extract (list): A list of tags to filter by.

    Returns:
        tuple: A tuple containing:
            - list of lines that contain the specified tags.
            - list of corresponding PDB file paths.
    """
    matching_lines = []
    pdb_file_path = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains any of the specified tags
            for tag in tags_to_extract:
                if f"tag: {tag}" in line:
                    matching_lines.append(line.strip())
                    pdb_file_path.append("AF2/" + tag + ".pdb")
                    break  # Stop checking other tags for this line

    return matching_lines, pdb_file_path

# Extract the relevant lines from the PAE file
if tags is not None:
    tags_to_extract = tags
    # example: tags_to_extract = ['0_2102_dldesign_0_cycle1_af2pred', '0_20_dldesign_0_cycle1_af2pred', '0_2101_dldesign_0_cycle1_af2pred', '0_2103_dldesign_0_cycle1_af2pred', '0_2100_dldesign_0_cycle1_af2pred']
    pae_matching_lines, pdb_file_paths = extract_pae_by_tags(pae_file_path, tags_to_extract)

# Define the ptm and d0 functions
def ptm_func(x,d0):
    return 1.0/(1+(x/d0)**2.0)  
ptm_func_vec=np.vectorize(ptm_func)  # vector version

# Define the d0 functions for numbers and arrays; minimum value = 1.0; from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702–710 (2004)
def calc_d0(L,pair_type):
    L=float(L)
    if L<27: L=27
    min_value=1.0
    if pair_type=='nucleic_acid': min_value=2.0
    d0=1.24*(L-15)**(1.0/3.0) - 1.8
    return max(min_value, d0)

def calc_d0_array(L,pair_type):
    # Convert L to a NumPy array if it isn't already one (enables flexibility in input types)
    L = np.array(L, dtype=float)
    L = np.maximum(27,L)
    min_value=1.0

    if pair_type=='nucleic_acid': min_value=2.0

    # Calculate d0 using the vectorized operation
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0/3.0) - 1.8)


# Define the parse_atom_line function for PDB lines (by column) and mmCIF lines (split by white_space)
# parsed_line = parse_atom_line(line)
# line = "ATOM    123  CA  ALA A  15     11.111  22.222  33.333  1.00 20.00           C"
def parse_pdb_atom_line(line):
    atom_num = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21].strip()
    residue_seq_num = line[22:26].strip()
    x = line[30:38].strip()
    y = line[38:46].strip()
    z = line[46:54].strip()

    # Convert string numbers to integers or floats as appropriate
    atom_num = int(atom_num)
    residue_seq_num = int(residue_seq_num)
    x = float(x)
    y = float(y)
    z = float(z)

    return {
        'atom_num': atom_num,
        'atom_name': atom_name,
        'residue_name': residue_name,
        'chain_id': chain_id,
        'residue_seq_num': residue_seq_num,
        'x': x,
        'y': y,
        'z': z
    }

# Function for printing out residue numbers in PyMOL scripts
def contiguous_ranges(numbers):
    if not numbers:  # Check if the set is empty
        return
    
    sorted_numbers = sorted(numbers)  # Sort the numbers
    start = sorted_numbers[0]
    end = start
    ranges = []  # List to store ranges

    def format_range(start, end):
        if start == end:
            return f"{start}"
        else:
            return f"{start}-{end}"

    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number
    
    # Append the last range after the loop
    ranges.append(format_range(start, end))

    # Join all ranges with a plus sign and print the result
    string='+'.join(ranges)
    return(string)

# Initializes a nested dictionary with all values set to 0
def init_chainpairdict_zeros(chainlist):
    return {chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with NumPy arrays of zeros of a specified size
def init_chainpairdict_npzeros(chainlist, arraysize):
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with empty sets.
def init_chainpairdict_set(chainlist):
    return {chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

def classify_chains(chains, residue_types):
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    
    # Get unique chains and iterate over them
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        # Find indices where the current chain is located
        indices = np.where(chains == chain)[0]
        # Get the residues for these indices
        chain_residues = residue_types[indices]
        # Count nucleic acid residues
        nuc_count = sum(residue in nuc_residue_set for residue in chain_residues)
        
        # Determine if the chain is a nucleic acid or protein
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'
    
    return chain_types


# Load residues from AlphaFold with Initial Guess PDB into lists; each residue is a dictionary
# Read PDB file to get CA coordinates, chainids, and residue numbers
# Convert to np arrays, and calculate distances
residues = []
cb_residues = []
chains = []
atomsitefield_num=0
atomsitefield_dict={} # contains order of atom_site fields in mmCIF files; handles any mmCIF field order

# For af3 and boltz1: need mask to identify CA atom tokens in plddt vector and pae matrix;
# Skip ligand atom tokens and non-CA-atom tokens in PTMs (those not in residue_set)
token_mask=list()     
residue_set= {"ALA", "ARG", "ASN", "ASP", "CYS",
              "GLN", "GLU", "GLY", "HIS", "ILE",
              "LEU", "LYS", "MET", "PHE", "PRO",
              "SER", "THR", "TRP", "TYR", "VAL",
              "DA", "DC", "DT", "DG", "A", "C", "U", "G"}

nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

#OUT2.write("i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res  ipSAE_d0chn ipSAE_d0dom    ipSAE \n")
OUT.write("\nChn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n")
PML.write("# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  n0res  n0chn  n0dom   d0res   d0chn   d0dom  nres1   nres2   dist1   dist2  Model\n")

try:
    total_lines = sum(1 for _ in open(pae_file_path, 'r'))
except Exception:
    total_lines = None

with open(pae_file_path, 'r') as file:
    for pae_line in tqdm(file, total=total_lines, desc="Loading PAE values and extracting distances from PDB files", unit="line"):
        atomsitefield_dict = {}
        token_mask = []
        residues = []
        cb_residues = []
        chains = []
        atomsitefield_num=0
        
        pae_line = pae_line.strip()
        if "tag:" not in pae_line or "pae:" not in pae_line:
            continue  # Skip lines that do not contain both "tag:" and "pae:"
        tag = pae_line.split("tag: ")[-1]
        if (tags is not None) and (tag not in tags):
            continue  # Skip this line if the tag is not in the specified tags
        file_path = Path(silent_dir).joinpath("AF2").joinpath(f"{tag}.pdb")

        # Extract PAE values from the line
        pae_values_str = pae_line.split("pae: ")[1].split(" tag: ")[0]
        pae_values = np.array([float(x.strip()) for x in pae_values_str.split(",") if x.strip()])

        # Add your processing logic here
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as PDB:
            for pdb_line in PDB:
                if pdb_line.startswith("_atom_site."):
                    pdb_line=pdb_line.strip()
                    (atomsite,fieldname)=pdb_line.split(".")
                    atomsitefield_dict[fieldname]=atomsitefield_num
                    atomsitefield_num += 1
                    
                if pdb_line.startswith("ATOM") or pdb_line.startswith("HETATM"):
                    atom=parse_pdb_atom_line(pdb_line)
                    if atom is None:  # ligand atom
                        token_mask.append(0)
                        continue

                    if atom['atom_name'] == "CA" or "C1" in atom['atom_name']:
                        token_mask.append(1)
                        residues.append({
                            'atom_num': atom['atom_num'],
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'residue': f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                        })
                        chains.append(atom['chain_id'])

                    if atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or (atom['residue_name']=="GLY" and atom['atom_name']=="CA"):
                        cb_residues.append({
                            'atom_num': atom['atom_num'],
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'residue': f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                        })

                    # add nucleic acids and non-CA atoms in PTM residues to tokens (as 0), whether labeled as "HETATM" (af3) or as "ATOM" (boltz1)
                    if atom['atom_name'] != "CA" and "C1" not in atom['atom_name'] and atom['residue_name'] not in residue_set:
                        token_mask.append(0)

        # Convert structure information to numpy arrays
        numres = len(residues)
        CA_atom_num=  np.array([res['atom_num']-1 for res in residues])  # for AF3 atom indexing from 0
        CB_atom_num=  np.array([res['atom_num']-1 for res in cb_residues])  # for AF3 atom indexing from 0
        coordinates = np.array([res['coor']       for res in cb_residues])
        chains = np.array(chains)
        unique_chains = np.unique(chains)
        token_array=np.array(token_mask)
        ntokens=np.sum(token_array)
        residue_types=np.array([res['res'] for res in residues])

        # chain types (nucleic acid (NA) or protein) and chain_pair_types ('nucleic_acid' if either chain is NA) for d0 calculation
        # arbitrarily setting d0 to 2.0 for NA/protein or NA/NA chain pairs (approximately 21 base pairs)
        chain_dict = classify_chains(chains, residue_types)
        chain_pair_type = init_chainpairdict_zeros(unique_chains)
        chain_pair_type = init_chainpairdict_zeros(unique_chains)
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1==chain2: continue
                if chain_dict[chain1] == 'nucleic_acid' or chain_dict[chain2] == 'nucleic_acid':
                    chain_pair_type[chain1][chain2]='nucleic_acid'
                else:
                    chain_pair_type[chain1][chain2]='protein'
            
        # Calculate distance matrix using NumPy broadcasting
        distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))

        # If you want a matrix, reshape as needed (e.g., square matrix)
        # Example: pae_matrix = pae_values.reshape((numres, numres))
        pae_matrix = pae_values.reshape((numres, numres))
        
        # Compute chain-pair-specific interchain PTM and PAE, count valid pairs, and count unique residues
        # First, create dictionaries of appropriate size: top keys are chain1 and chain2 where chain1 != chain2
        # Nomenclature:
        # iptm_d0chn =  calculate iptm  from PAEs with no PAE cutoff; d0 = numres in chain pair = len(chain1) + len(chain2)
        # ipsae_d0chn = calculate ipsae from PAEs with PAE cutoff;    d0 = numres in chain pair = len(chain1) + len(chain2)
        # ipsae_d0dom = calculate ipsae from PAEs with PAE cutoff;    d0 from number of residues in chain1 and chain2 that have interchain PAE<cutoff
        # ipsae_d0res = calculate ipsae from PAEs with PAE cutoff;    d0 from number of residues in chain2 that have interchain PAE<cutoff given residue in chain1
        # 
        # for each chain_pair iptm/ipsae, there is (for example)
        # ipsae_d0res_byres = by-residue array;
        # ipsae_d0res_asym  = asymmetric pair value (A->B is different from B->A)
        # ipsae_d0res_max   = maximum of A->B and B->A value
        # ipsae_d0res_asymres = identify of residue that provides each asym maximum
        # ipsae_d0res_maxres =  identify of residue that provides each maximum over both chains
        #
        # n0num = number of residues in whole complex provided by AF2 model
        # n0chn = number of residues in chain pair = len(chain1) + len(chain2)
        # n0dom = number of residues in chain pair that have good PAE values (<cutoff)
        # n0res = number of residues in chain2 that have good PAE residues for each residue of chain1

        iptm_d0chn_byres  = init_chainpairdict_npzeros(unique_chains, numres)
        ipsae_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
        ipsae_d0dom_byres = init_chainpairdict_npzeros(unique_chains, numres)
        ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

        iptm_d0chn_asym   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0chn_asym  = init_chainpairdict_zeros(unique_chains)
        ipsae_d0dom_asym  = init_chainpairdict_zeros(unique_chains)
        ipsae_d0res_asym  = init_chainpairdict_zeros(unique_chains)

        iptm_d0chn_max    = init_chainpairdict_zeros(unique_chains)
        ipsae_d0chn_max   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0dom_max   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0res_max   = init_chainpairdict_zeros(unique_chains)

        iptm_d0chn_asymres   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0chn_asymres  = init_chainpairdict_zeros(unique_chains)
        ipsae_d0dom_asymres  = init_chainpairdict_zeros(unique_chains)
        ipsae_d0res_asymres  = init_chainpairdict_zeros(unique_chains)

        iptm_d0chn_maxres    = init_chainpairdict_zeros(unique_chains)
        ipsae_d0chn_maxres   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0dom_maxres   = init_chainpairdict_zeros(unique_chains)
        ipsae_d0res_maxres   = init_chainpairdict_zeros(unique_chains)

        n0chn       = init_chainpairdict_zeros(unique_chains)
        n0dom       = init_chainpairdict_zeros(unique_chains)
        n0dom_max   = init_chainpairdict_zeros(unique_chains)
        n0res       = init_chainpairdict_zeros(unique_chains)
        n0res_max   = init_chainpairdict_zeros(unique_chains)
        n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

        d0chn       = init_chainpairdict_zeros(unique_chains)
        d0dom       = init_chainpairdict_zeros(unique_chains)
        d0dom_max   = init_chainpairdict_zeros(unique_chains)
        d0res       = init_chainpairdict_zeros(unique_chains)
        d0res_max   = init_chainpairdict_zeros(unique_chains)
        d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

        valid_pair_counts           = init_chainpairdict_zeros(unique_chains)
        dist_valid_pair_counts      = init_chainpairdict_zeros(unique_chains)
        unique_residues_chain1      = init_chainpairdict_set(unique_chains)
        unique_residues_chain2      = init_chainpairdict_set(unique_chains)
        dist_unique_residues_chain1 = init_chainpairdict_set(unique_chains)
        dist_unique_residues_chain2 = init_chainpairdict_set(unique_chains)

        # calculate ipTM/ipSAE with and without PAE cutoff

        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue

                n0chn[chain1][chain2]=np.sum( chains==chain1) + np.sum(chains==chain2) # total number of residues in chain1 and chain2
                d0chn[chain1][chain2]=calc_d0(n0chn[chain1][chain2], chain_pair_type[chain1][chain2])
                ptm_matrix_d0chn=np.zeros((numres,numres))
                ptm_matrix_d0chn=ptm_func_vec(pae_matrix,d0chn[chain1][chain2])

                valid_pairs_iptm = (chains == chain2)
                valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

                for i in range(numres):


                    if chains[i] != chain1:
                        continue

                    valid_pairs_ipsae = valid_pairs_matrix[i]  # row for residue i of chain1
                    iptm_d0chn_byres[chain1][chain2][i] =  ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                    ipsae_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_ipsae].mean() if valid_pairs_ipsae.any() else 0.0

                    # Track unique residues contributing to the IPSAE for chain1,chain2
                    valid_pair_counts[chain1][chain2] += np.sum(valid_pairs_ipsae)
                    if valid_pairs_ipsae.any():
                        iresnum=residues[i]['resnum']
                        unique_residues_chain1[chain1][chain2].add(iresnum)
                        for j in np.where(valid_pairs_ipsae)[0]:
                            jresnum=residues[j]['resnum']
                            unique_residues_chain2[chain1][chain2].add(jresnum)
                            
                    # Track unique residues contributing to iptm in interface
                    valid_pairs = (chains == chain2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                    dist_valid_pair_counts[chain1][chain2] += np.sum(valid_pairs)

                    # Track unique residues contributing to the IPTM
                    if valid_pairs.any():
                        iresnum=residues[i]['resnum']
                        dist_unique_residues_chain1[chain1][chain2].add(iresnum)
                        for j in np.where(valid_pairs)[0]:
                            jresnum=residues[j]['resnum']
                            dist_unique_residues_chain2[chain1][chain2].add(jresnum)

        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                residues_1 = len(unique_residues_chain1[chain1][chain2])
                residues_2 = len(unique_residues_chain2[chain1][chain2])
                n0dom[chain1][chain2] = residues_1+residues_2
                d0dom[chain1][chain2] = calc_d0(n0dom[chain1][chain2], chain_pair_type[chain1][chain2])

                ptm_matrix_d0dom = np.zeros((numres,numres))
                ptm_matrix_d0dom = ptm_func_vec(pae_matrix,d0dom[chain1][chain2])

                valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

                # Assuming valid_pairs_matrix is already defined
                n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
                d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])

                n0res_byres[chain1][chain2] = n0res_byres_all
                d0res_byres[chain1][chain2] = d0res_byres_all
                
                for i in range(numres):
                    if chains[i] != chain1:
                        continue
                    valid_pairs = valid_pairs_matrix[i]
                    ipsae_d0dom_byres[chain1][chain2][i] = ptm_matrix_d0dom[i, valid_pairs].mean() if valid_pairs.any() else 0.0

                    ptm_row_d0res=np.zeros((numres))
                    ptm_row_d0res=ptm_func_vec(pae_matrix[i], d0res_byres[chain1][chain2][i])
                    ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0

                    outstring = f'{i+1:<4d}    ' + (
                        f'{chain1:4}      '
                        f'{chain2:4}      '
                        f'{residues[i]["resnum"]:4d}           '
                        f'{residues[i]["res"]:3}        '        
                        f'{int(n0chn[chain1][chain2]):5d}  '
                        f'{int(n0dom[chain1][chain2]):5d}  '
                        f'{int(n0res_byres[chain1][chain2][i]):5d}  '
                        f'{d0chn[chain1][chain2]:8.3f}  '
                        f'{d0dom[chain1][chain2]:8.3f}  '
                        f'{d0res_byres[chain1][chain2][i]:8.3f}   '
                        f'{iptm_d0chn_byres[chain1][chain2][i]:8.4f}    '
                        f'{ipsae_d0chn_byres[chain1][chain2][i]:8.4f}    '
                        f'{ipsae_d0dom_byres[chain1][chain2][i]:8.4f}    '
                        f'{ipsae_d0res_byres[chain1][chain2][i]:8.4f}\n'
                    )
                    #OUT2.write(outstring)
                    
        # Compute interchain ipTM and ipSAE for each chain pair
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue

                interchain_values = iptm_d0chn_byres[chain1][chain2]
                max_index = np.argmax(interchain_values)
                iptm_d0chn_asym[chain1][chain2] = interchain_values[max_index]
                iptm_d0chn_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

                interchain_values = ipsae_d0chn_byres[chain1][chain2]
                max_index = np.argmax(interchain_values)
                ipsae_d0chn_asym[chain1][chain2] = interchain_values[max_index]
                ipsae_d0chn_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

                interchain_values = ipsae_d0dom_byres[chain1][chain2]
                max_index = np.argmax(interchain_values)
                ipsae_d0dom_asym[chain1][chain2] = interchain_values[max_index]
                ipsae_d0dom_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

                interchain_values = ipsae_d0res_byres[chain1][chain2]
                max_index = np.argmax(interchain_values)
                ipsae_d0res_asym[chain1][chain2] = interchain_values[max_index]
                ipsae_d0res_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"
                n0res[chain1][chain2]=n0res_byres[chain1][chain2][max_index]
                d0res[chain1][chain2]=d0res_byres[chain1][chain2][max_index]

                # pick maximum value for each chain pair for each iptm/ipsae type
                if chain1 > chain2:
                    maxvalue=max(iptm_d0chn_asym[chain1][chain2], iptm_d0chn_asym[chain2][chain1])
                    if maxvalue==iptm_d0chn_asym[chain1][chain2]: maxres=iptm_d0chn_asymres[chain1][chain2]
                    else: maxres=iptm_d0chn_asymres[chain2][chain1]
                    iptm_d0chn_max[chain1][chain2]=maxvalue
                    iptm_d0chn_maxres[chain1][chain2]=maxres
                    iptm_d0chn_max[chain2][chain1]=maxvalue
                    iptm_d0chn_maxres[chain2][chain1]=maxres

                    maxvalue=max(ipsae_d0chn_asym[chain1][chain2], ipsae_d0chn_asym[chain2][chain1])
                    if maxvalue==ipsae_d0chn_asym[chain1][chain2]: maxres=ipsae_d0chn_asymres[chain1][chain2]
                    else: maxres=ipsae_d0chn_asymres[chain2][chain1]
                    ipsae_d0chn_max[chain1][chain2]=maxvalue
                    ipsae_d0chn_maxres[chain1][chain2]=maxres
                    ipsae_d0chn_max[chain2][chain1]=maxvalue
                    ipsae_d0chn_maxres[chain2][chain1]=maxres

                    maxvalue=max(ipsae_d0dom_asym[chain1][chain2], ipsae_d0dom_asym[chain2][chain1])
                    if maxvalue==ipsae_d0dom_asym[chain1][chain2]:
                        maxres=ipsae_d0dom_asymres[chain1][chain2]
                        maxn0=n0dom[chain1][chain2]
                        maxd0=d0dom[chain1][chain2]
                    else:
                        maxres=ipsae_d0dom_asymres[chain2][chain1]
                        maxn0=n0dom[chain2][chain1]
                        maxd0=d0dom[chain2][chain1]
                    ipsae_d0dom_max[chain1][chain2]=maxvalue
                    ipsae_d0dom_maxres[chain1][chain2]=maxres
                    ipsae_d0dom_max[chain2][chain1]=maxvalue
                    ipsae_d0dom_maxres[chain2][chain1]=maxres
                    n0dom_max[chain1][chain2]=maxn0
                    n0dom_max[chain2][chain1]=maxn0
                    d0dom_max[chain1][chain2]=maxd0
                    d0dom_max[chain2][chain1]=maxd0

                    maxvalue=max(ipsae_d0res_asym[chain1][chain2], ipsae_d0res_asym[chain2][chain1])
                    if maxvalue==ipsae_d0res_asym[chain1][chain2]:
                        maxres=ipsae_d0res_asymres[chain1][chain2]
                        maxn0=n0res[chain1][chain2]
                        maxd0=d0res[chain1][chain2]
                    else:
                        maxres=ipsae_d0res_asymres[chain2][chain1]
                        maxn0=n0res[chain2][chain1]
                        maxd0=d0res[chain2][chain1]
                    ipsae_d0res_max[chain1][chain2]=maxvalue
                    ipsae_d0res_maxres[chain1][chain2]=maxres
                    ipsae_d0res_max[chain2][chain1]=maxvalue
                    ipsae_d0res_maxres[chain2][chain1]=maxres
                    n0res_max[chain1][chain2]=maxn0
                    n0res_max[chain2][chain1]=maxn0
                    d0res_max[chain1][chain2]=maxd0
                    d0res_max[chain2][chain1]=maxd0

                        
        chaincolor={'A':'magenta',   'B':'marine',   'C':'lime',        'D':'orange',
                    'E':'yellow',    'F':'cyan',     'G':'lightorange', 'H':'pink',
                    'I':'deepteal',  'J':'forest',   'K':'lightblue',   'L':'slate',
                    'M':'violet',    'N':'arsenic',  'O':'iodine',      'P':'silver',
                    'Q':'red',       'R':'sulfur',   'S':'purple',      'T':'olive',
                    'U':'palegreen', 'V':'green',    'W':'blue',        'X':'palecyan',
                    'Y':'limon',     'Z':'chocolate'}

        chainpairs=set()
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 >= chain2: continue
                chainpairs.add(chain1 + "-" + chain2)

        for pair in sorted(chainpairs):
            (chain_a, chain_b) = pair.split("-")
            pair1 = (chain_a, chain_b)
            pair2 = (chain_b, chain_a)
            for pair in (pair1, pair2):
                chain1=pair[0]
                chain2=pair[1]

                if chain1 in chaincolor:
                    color1=chaincolor[chain1]
                else:
                    color1='magenta'

                if chain2 in chaincolor:
                    color2=chaincolor[chain2]
                else:
                    color2='marine'

                residues_1 = len(unique_residues_chain1[chain1][chain2])
                residues_2 = len(unique_residues_chain2[chain1][chain2])
                dist_residues_1 = len(dist_unique_residues_chain1[chain1][chain2])
                dist_residues_2 = len(dist_unique_residues_chain2[chain1][chain2])
                pairs = valid_pair_counts[chain1][chain2]
                dist_pairs = dist_valid_pair_counts[chain1][chain2]
                
                outstring=f'{chain1}    {chain2}     {pae_string:3}  {dist_string:3}  {"asym":5} ' + (
                    f'{ipsae_d0res_asym[chain1][chain2]:8.6f}    '
                    f'{ipsae_d0chn_asym[chain1][chain2]:8.6f}    '
                    f'{ipsae_d0dom_asym[chain1][chain2]:8.6f}    '
                    f'{int(n0res[chain1][chain2]):5d}  '
                    f'{int(n0chn[chain1][chain2]):5d}  '
                    f'{int(n0dom[chain1][chain2]):5d}  '
                    f'{d0res[chain1][chain2]:6.2f}  '
                    f'{d0chn[chain1][chain2]:6.2f}  '
                    f'{d0dom[chain1][chain2]:6.2f}  '
                    f'{residues_1:5d}   '
                    f'{residues_2:5d}   '
                    f'{dist_residues_1:5d}   '
                    f'{dist_residues_2:5d}   '
                    f'{tag}\n')
                OUT.write(outstring)
                PML.write("# " + outstring)
                if chain1 > chain2:
                    residues_1 = max(len(unique_residues_chain2[chain1][chain2]), len(unique_residues_chain1[chain2][chain1]))
                    residues_2 = max(len(unique_residues_chain1[chain1][chain2]), len(unique_residues_chain2[chain2][chain1]))
                    dist_residues_1 = max(len(dist_unique_residues_chain2[chain1][chain2]), len(dist_unique_residues_chain1[chain2][chain1]))
                    dist_residues_2 = max(len(dist_unique_residues_chain1[chain1][chain2]), len(dist_unique_residues_chain2[chain2][chain1]))

                    outstring=f'{chain2}    {chain1}     {pae_string:3}  {dist_string:3}  {"max":5} ' + (
                        f'{ipsae_d0res_max[chain1][chain2]:8.6f}    '
                        f'{ipsae_d0chn_max[chain1][chain2]:8.6f}    '
                        f'{ipsae_d0dom_max[chain1][chain2]:8.6f}    '
                        f'{int(n0res_max[chain1][chain2]):5d}  '
                        f'{int(n0chn[chain1][chain2]):5d}  '
                        f'{int(n0dom_max[chain1][chain2]):5d}  '
                        f'{d0res_max[chain1][chain2]:6.2f}  '
                        f'{d0chn[chain1][chain2]:6.2f}  '
                        f'{d0dom_max[chain1][chain2]:6.2f}  '
                        f'{residues_1:5d}   '
                        f'{residues_2:5d}   '
                        f'{dist_residues_1:5d}   '
                        f'{dist_residues_2:5d}   '
                        f'{tag}\n')
                    OUT.write(outstring)
                    PML.write("# " + outstring)
                        
                chain_pair= f'color_{chain1}_{chain2}'
                chain1_residues = f'chain  {chain1} and resi {contiguous_ranges(unique_residues_chain1[chain1][chain2])}'
                chain2_residues = f'chain  {chain2} and resi {contiguous_ranges(unique_residues_chain2[chain1][chain2])}'
                PML.write(f'alias {chain_pair}, color gray80, all; color {color1}, {chain1_residues}; color {color2}, {chain2_residues}\n\n')
            OUT.write("\n")