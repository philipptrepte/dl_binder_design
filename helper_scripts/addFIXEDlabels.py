#!/usr/bin/env python

import numpy as np
import os
import argparse

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pdbdir", type=str, required=True)
parser.add_argument("--trbdir", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
args = parser.parse_args()

pdb_list = os.listdir(args.pdbdir)

for pdb in pdb_list:
    
    if not pdb.endswith(".pdb"):
        if args.verbose:
            print(f"Skipping {pdb} as it is not a PDB file")
        continue

    pdb_path = os.path.join(args.pdbdir, pdb)

    trb_path = os.path.join(args.trbdir, f'{pdb.split(".")[0]}.trb')

    if not os.path.exists(trb_path):
        print(f"Error: TRB file not found for PDB {pdb}")
        continue

    data = np.load(trb_path, allow_pickle=True)
    
    if 'receptor_con_hal_pdb_idx' in data:
        #Identify the last residue number in A chain
        last_res_id = int(data['receptor_con_hal_pdb_idx'][0][1]) - 1

        #Identify where inpaint seq is True, ie, kept fixed
        indices = np.where(data['inpaint_seq'][:last_res_id])[0]
    else:
        #Identify where inpaint seq is True in the only chain
        indices = np.where(data['inpaint_seq'])[0]

    if args.verbose:
        print(f"Adding FIXED labels to {pdb} at positions {indices}")

    
    remarks = []
    for position in indices:
        remark = f"REMARK PDBinfo-LABEL:{position+1: >5} FIXED"
        remarks.append(remark)

    remarks_str = '\n'.join(remarks)
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()
    
    with open(pdb_path, 'w') as f:
        for line in pdb_lines:
            number = int(line[22:26])  # Extract the number at position 22 to 26
            if number in (indices+1):
                if line[20:23] == " A ":
                    line = line[:20] + " B " + line[23:]  # Replace the value at position 20 to 22
            f.write(line)
        f.write('\n')
        f.write(remarks_str)
