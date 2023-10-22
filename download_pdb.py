from Bio.PDB import PDBParser
import requests
import csv
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO



def calc_dist(amino_acids_coor, pro, dist_path):
    dict_distances = {}
    for resi, resi_coord1 in amino_acids_coor:
        for resi2, resi_coord2 in amino_acids_coor:
            if resi != resi2:
                dist = 0
                for i in range(3):
                    dist += (resi_coord1[i] - resi_coord2[i]) ** 2
                dist = dist ** 0.5
                if (resi, resi2) not in dict_distances:
                    dict_distances[((resi, resi2))] = dist
                else:
                    dict_distances[((resi, resi2))] = min(dist, dict_distances[((resi, resi2))])

    f_out = open(f"{dist_path}/{pro}.csv", "w")
    for resi, dist in dict_distances.items():
        f_out.write(f"{resi[0]},{resi[1]},{dist}\n")
    f_out.close()

def extract_sequence_from_pdb(pdb_file_path, pdb_id,  output_dir, dist_path, chain_pdb = -1):

    amino_acid_dict = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "AP5": ""
    }

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)

    dict_model_chain_seq = {}
    for model in structure:
        for chain in model:
            if chain_pdb == -1 or chain._id == chain_pdb:
                sequence = ""
                amino_acids_coor = []
                amino_acid = 0
                for residue in chain:
                    if residue.get_resname() != "HOH" and residue._id[0] == " ":  # Exclude water molecules
                        sequence += amino_acid_dict[residue.get_resname()]
                        amino_acid += 1
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                coordinates = atom.get_coord()
                                amino_acids_coor.append((amino_acid, coordinates))

                dict_model_chain_seq[f"{chain._id}_{model.id}"] = sequence

                #calc_dist
                name_for_dist_file = f"{pdb_id}_{chain._id}_{model.id}" if chain_pdb == -1 else  f"{pdb_id}_{chain._id}"
                calc_dist(amino_acids_coor, name_for_dist_file, dist_path )

                io = PDBIO()
                io.set_structure(chain)
                io.save(f"{output_dir}/{name_for_dist_file}.pdb")


    return dict_model_chain_seq




def download_pdb_and_save(pdb_id, output_dir):

    try:
        pdb_id, chain = pdb_id.split("_")
        pdb_file_path = f"{output_dir}/{pdb_id}.pdb"
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)

        if response.status_code == 200:
            with open(pdb_file_path, 'w') as pdb_file:
                pdb_file.write(response.text)
            print(f"PDB file {pdb_id}.pdb downloaded and saved successfully.")
        else:
            print(f"Failed to download PDB file {pdb_id}.pdb. Error: {response.status_code}")

        return 0

    except:
        print(f"Failed to download {pdb_id} PDB")
        return -1



