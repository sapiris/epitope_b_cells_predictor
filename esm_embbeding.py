import sys
import subprocess
import torch

ESM_SCRIPT_PATH =  "esm_utils/extract.py"

class esm_embbeding():
    def __init__(self, accs, seqs,  esm_encoding_dir):
        self.accs = accs
        self.seqs = seqs
        self.esm_encoding_dir = esm_encoding_dir

    def create_fasta_for_esm_transformer(self):
        """
        Outputs fasta file accesions and sequences into a fasta file format, that can be read by ESM-2 transformer.
        """
        uppercase_entries = list()
        # convert all sequences to uppercase
        entries = list(zip(self.accs, self.seqs))

        for entry in entries:
            acc = entry[0]
            sequence = entry[1]
            upper_case_sequence = sequence.upper()
            uppercase_entries.append((acc, upper_case_sequence))

        with open( f"{self.esm_encoding_dir}/antigens_{self.accs}.fasta", "w") as outfile:
            output = str()
            for entry in uppercase_entries:
                output += f">{entry[0]}\n{entry[1]}\n"

            output = output[:-1]
            outfile.write(output)

    def call_esm_script(self):
        fastaPath = f"{self.esm_encoding_dir}/antigens_{self.accs}.fasta"

        try:  # only using this for biolib implementation
                subprocess.check_call(
                    ['python', ESM_SCRIPT_PATH, "esm2_t33_650M_UR50D", fastaPath, self.esm_encoding_dir, "--include",
                     "per_tok"])
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"ESM model could not be run with following error message: {error}.\nThis is likely a memory issue.")


    def prepare_esm_data(self):

            esm_representations = list()
            for acc in self.accs:
                esm_encoded_acc = torch.load(self.esm_encoding_dir + f"/{acc}.pt")
                esm_representation = esm_encoded_acc["representations"][33]

                esm_representations.append(esm_representation)

            return esm_representations

