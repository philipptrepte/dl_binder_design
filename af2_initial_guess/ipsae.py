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
from concurrent.futures import ProcessPoolExecutor, as_completed
np.set_printoptions(threshold=np.inf)  # for printing out full numpy arrays for debugging

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

def parse_pae_line(pae_line):
    pae_line = pae_line.strip()
    if "tag:" not in pae_line or "pae:" not in pae_line:
        return None

    tag = pae_line.split("tag: ")[-1]
    pae_values_str = pae_line.split("pae: ")[1].split(" tag: ")[0]
    pae_values = np.array([float(x.strip())
                           for x in pae_values_str.split(",") if x.strip()])
    return tag, pae_values

def process_one_model(args):
    """
    args: (tag, pae_values, silent_dir, pae_cutoff, dist_cutoff, residue_set, pae_string, dist_string)
    returns: list of strings to be written to OUT
    """
    (tag, pae_values, silent_dir, pae_cutoff, dist_cutoff,
     residue_set, pae_string, dist_string) = args

    lines = []
    file_path = Path(silent_dir).joinpath("AF2").joinpath(f"{tag}.pdb")
    if not file_path.exists():
        return lines  # nothing for this tag

    atomsitefield_dict = {}
    token_mask = []
    residues = []
    cb_residues = []
    chains_list = []
    atomsitefield_num = 0

    with open(file_path, 'r') as PDB:
        for pdb_line in PDB:
            if pdb_line.startswith("_atom_site."):
                pdb_line = pdb_line.strip()
                (_, fieldname) = pdb_line.split(".")
                atomsitefield_dict[fieldname] = atomsitefield_num
                atomsitefield_num += 1

            if not (pdb_line.startswith("ATOM") or pdb_line.startswith("HETATM")):
                continue

            atom = parse_pdb_atom_line(pdb_line)

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
                chains_list.append(atom['chain_id'])

            if atom['atom_name'] == "CB" or "C3" in atom['atom_name'] or (atom['residue_name'] == "GLY" and atom['atom_name'] == "CA"):
                cb_residues.append({
                    'atom_num': atom['atom_num'],
                    'coor': np.array([atom['x'], atom['y'], atom['z']]),
                    'res': atom['residue_name'],
                    'chainid': atom['chain_id'],
                    'resnum': atom['residue_seq_num'],
                    'residue': f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                })

            if atom['atom_name'] != "CA" and "C1" not in atom['atom_name'] and atom['residue_name'] not in residue_set:
                token_mask.append(0)

    numres = len(residues)
    if numres == 0:
        return lines

    try:
        pae_matrix = pae_values.reshape((numres, numres))
    except ValueError:
        return lines

    chains = np.array(chains_list)
    unique_chains = np.unique(chains)
    residue_types = np.array([r['res'] for r in residues])

    # classify chains
    chain_dict = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            chain_pair_type[c1][c2] = 'nucleic_acid' if (chain_dict[c1] == 'nucleic_acid' or chain_dict[c2] == 'nucleic_acid') else 'protein'

    coordinates = np.array([r['coor'] for r in cb_residues])
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2))

    # init dicts (unchanged)
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

    # main loops (same as before, using chains)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            n0chn[c1][c2] = np.sum(chains == c1) + np.sum(chains == c2)
            d0chn[c1][c2] = calc_d0(n0chn[c1][c2], chain_pair_type[c1][c2])
            ptm_matrix_d0chn = ptm_func_vec(pae_matrix, d0chn[c1][c2])
            valid_pairs_iptm = (chains == c2)
            valid_pairs_matrix = (chains == c2) & (pae_matrix < pae_cutoff)
            for i in range(numres):
                if chains[i] != c1:
                    continue
                v_ipsae = valid_pairs_matrix[i]
                iptm_d0chn_byres[c1][c2][i] = ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                ipsae_d0chn_byres[c1][c2][i] = ptm_matrix_d0chn[i, v_ipsae].mean() if v_ipsae.any() else 0.0
                valid_pair_counts[c1][c2] += np.sum(v_ipsae)
                if v_ipsae.any():
                    unique_residues_chain1[c1][c2].add(residues[i]['resnum'])
                    for j in np.where(v_ipsae)[0]:
                        unique_residues_chain2[c1][c2].add(residues[j]['resnum'])
                v_dist = (chains == c2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                dist_valid_pair_counts[c1][c2] += np.sum(v_dist)
                if v_dist.any():
                    dist_unique_residues_chain1[c1][c2].add(residues[i]['resnum'])
                    for j in np.where(v_dist)[0]:
                        dist_unique_residues_chain2[c1][c2].add(residues[j]['resnum'])

    # domain/res loops
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            residues_1 = len(unique_residues_chain1[c1][c2])
            residues_2 = len(unique_residues_chain2[c1][c2])
            n0dom[c1][c2] = residues_1 + residues_2
            d0dom[c1][c2] = calc_d0(n0dom[c1][c2], chain_pair_type[c1][c2])
            ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom[c1][c2])
            valid_pairs_matrix = (chains == c2) & (pae_matrix < pae_cutoff)
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[c1][c2])
            n0res_byres[c1][c2] = n0res_byres_all
            d0res_byres[c1][c2] = d0res_byres_all
            for i in range(numres):
                if chains[i] != c1:
                    continue
                v = valid_pairs_matrix[i]
                ipsae_d0dom_byres[c1][c2][i] = ptm_matrix_d0dom[i, v].mean() if v.any() else 0.0
                ptm_row_d0res = ptm_func_vec(pae_matrix[i], d0res_byres[c1][c2][i])
                ipsae_d0res_byres[c1][c2][i] = ptm_row_d0res[v].mean() if v.any() else 0.0

    # asym + max summaries
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            idx = np.argmax(iptm_d0chn_byres[c1][c2]); iptm_d0chn_asym[c1][c2] = iptm_d0chn_byres[c1][c2][idx]; iptm_d0chn_asymres[c1][c2] = residues[idx]['residue']
            idx = np.argmax(ipsae_d0chn_byres[c1][c2]); ipsae_d0chn_asym[c1][c2] = ipsae_d0chn_byres[c1][c2][idx]; ipsae_d0chn_asymres[c1][c2] = residues[idx]['residue']
            idx = np.argmax(ipsae_d0dom_byres[c1][c2]); ipsae_d0dom_asym[c1][c2] = ipsae_d0dom_byres[c1][c2][idx]; ipsae_d0dom_asymres[c1][c2] = residues[idx]['residue']
            idx = np.argmax(ipsae_d0res_byres[c1][c2]); ipsae_d0res_asym[c1][c2] = ipsae_d0res_byres[c1][c2][idx]; ipsae_d0res_asymres[c1][c2] = residues[idx]['residue']
            n0res[c1][c2] = n0res_byres[c1][c2][idx]; d0res[c1][c2] = d0res_byres[c1][c2][idx]

    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 <= c2:
                continue
            def pick(val_a, val_b, res_a, res_b):
                return (val_a, res_a) if val_a >= val_b else (val_b, res_b)
            iptm_d0chn_max[c1][c2], iptm_d0chn_maxres[c1][c2] = pick(iptm_d0chn_asym[c1][c2], iptm_d0chn_asym[c2][c1], iptm_d0chn_asymres[c1][c2], iptm_d0chn_asymres[c2][c1])
            ipsae_d0chn_max[c1][c2], ipsae_d0chn_maxres[c1][c2] = pick(ipsae_d0chn_asym[c1][c2], ipsae_d0chn_asym[c2][c1], ipsae_d0chn_asymres[c1][c2], ipsae_d0chn_asymres[c2][c1])
            ipsae_d0dom_max[c1][c2], ipsae_d0dom_maxres[c1][c2] = pick(ipsae_d0dom_asym[c1][c2], ipsae_d0dom_asym[c2][c1], ipsae_d0dom_asymres[c1][c2], ipsae_d0dom_asymres[c2][c1])
            ipsae_d0res_max[c1][c2], ipsae_d0res_maxres[c1][c2] = pick(ipsae_d0res_asym[c1][c2], ipsae_d0res_asym[c2][c1], ipsae_d0res_asymres[c1][c2], ipsae_d0res_asymres[c2][c1])
            # mirror
            iptm_d0chn_max[c2][c1] = iptm_d0chn_max[c1][c2]; iptm_d0chn_maxres[c2][c1] = iptm_d0chn_maxres[c1][c2]
            ipsae_d0chn_max[c2][c1] = ipsae_d0chn_max[c1][c2]; ipsae_d0chn_maxres[c2][c1] = ipsae_d0chn_maxres[c1][c2]
            ipsae_d0dom_max[c2][c1] = ipsae_d0dom_max[c1][c2]; ipsae_d0dom_maxres[c2][c1] = ipsae_d0dom_maxres[c1][c2]
            ipsae_d0res_max[c2][c1] = ipsae_d0res_max[c1][c2]; ipsae_d0res_maxres[c2][c1] = ipsae_d0res_maxres[c1][c2]

    # output (pair-level summary)
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if c1 < c2:
                #lines.append(f'{c1}    {c2}     {pae_string:3}  {dist_string:3}  asym  {ipsae_d0dom_asym[c1][c2]:8.6f} {tag}')
                #lines.append(f'{c2}    {c1}     {pae_string:3}  {dist_string:3}  asym  {ipsae_d0dom_asym[c2][c1]:8.6f} {tag}')
                lines.append(f'{c1}    {c2}     {pae_string:3}  {dist_string:3}  max   {ipsae_d0dom_max[c1][c2]:8.6f} {tag}')
    return lines


if __name__ == "__main__":
    # Input and output files and parameters

    # Ensure correct usage
    if len(sys.argv) < 5:
        print("Usage for AF2 with initial guess:")
        print("   python ipsae.py <path_to_pae_file> <path_to_silent_file> <pae_cutoff> <dist_cutoff>")
        print("   python ipsae.py out_af2.pae out_af2.silent 10 10")
        sys.exit(1)

    pae_file_path =    sys.argv[1]
    silent_file_path = sys.argv[2]
    pae_cutoff =       float(sys.argv[3])
    dist_cutoff =      float(sys.argv[4])
    pae_string =       str(int(pae_cutoff))
    if pae_cutoff<10:  pae_string="0"+pae_string
    dist_string =      str(int(dist_cutoff))
    if dist_cutoff<10: dist_string="0"+dist_string

    if ".silent" in silent_file_path:
        silent_stem=silent_file_path.replace(".silent","")
        path_stem =     f'{silent_file_path.replace(".silent","")}_pae_{pae_string}_dist_{dist_string}'
    else:
        print("Wrong file type ", silent_file_path)
        sys.exit()
        
    file_path =        path_stem + "_IPSAE.txt"
    OUT =              open(file_path,'w')

    # Extract the pdb files from the .silent file using silent tools
    max_workers = os.cpu_count() or 1
    silent_path = Path(silent_file_path)
    silent_dir = silent_path.parent
    silent_name = silent_path.name
    # Set locale to C to avoid issues with silentextract for MacOS
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    # or: env["LC_ALL"] = "en_US.UTF-8"

    # For af3 and boltz1: need mask to identify CA atom tokens in plddt vector and pae matrix;
    # Skip ligand atom tokens and non-CA-atom tokens in PTMs (those not in residue_set)
    token_mask=list()     
    residue_set= {"ALA", "ARG", "ASN", "ASP", "CYS",
                "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO",
                "SER", "THR", "TRP", "TYR", "VAL",
                "DA", "DC", "DT", "DG", "A", "C", "U", "G"}


    if not os.path.exists(Path(silent_dir).joinpath("AF2")):
        print("Extracting PDB files from silent file...")
        command = f'mkdir -p AF2 && cd AF2 && silentextract -j {max_workers} ../"{silent_name}"'
        subprocess.run(command, shell=True, check=True, env=env, cwd=silent_dir)
    else:
        print("AF2 folder already exisits. PDB files already extracted.")
    # read and parse all PAE lines once
    tasks = []
    try:
        total_lines = sum(1 for _ in open(pae_file_path, 'r'))
    except Exception:
        total_lines = None

    print("\n")
    with open(pae_file_path, 'r') as f:
        for pae_line in tqdm(f, total=total_lines,
                             desc="Reading PAE file", unit=" line"):
            parsed = parse_pae_line(pae_line)
            if parsed is None:
                continue
            tasks.append(parsed)   # (tag, pae_values)

    # parallel processing
    print("\n")
    print(f"Processing {len(tasks)} models in parallel using {max_workers} workers...")
    # build argument tuples with context
    worker_tasks = [(tag, pae_values, silent_dir, pae_cutoff, dist_cutoff,
                     residue_set, pae_string, dist_string) for (tag, pae_values) in tasks]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_model, wt) for wt in worker_tasks]

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Processing models in parallel",
                        unit="model"):
            lines = fut.result()
            for line in lines:
                OUT.write(line + "\n")

    print(f"\nIPSAE results written to {file_path}\n")
    OUT.close()