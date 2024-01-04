# CALIBER - Conformational And LInear B cell Epitopes pRediction


### Prediction
All the trained models are on the website - https://caliber.math.biu.ac.il/, and can be predicted directly through the website. You can also use a git code for this

The test data can be  in 3 different formats (sequnces in /fasta format, PDB IDs, PDB Files)
main.py --mode predict --init XXX --model XXX --epi XXX --test_seq_input XXX
main.py --mode predict --init XXX --model XXX --epi XXX --test_pdb_path XXX
main.py --mode predict --init XXX --model XXX --epi XXX --test_pdb_list XXX

For example: 
main.py --mode predict --init Random --model BiLSTM --epi Nonlinear --test_seq_input data/nonlinear_test.fasta

### Training
main.py --mode train --init XXX --model XXX --epi XXX --train_path XXX --test_path XXX

For example:
main.py --mode train --init Random --model BiLSTM --epi Linear --train_path data/linear_train.csv --test_path data/linear_test.csv


### Parameters

| Parameter  | Description                                        | Required       | Options                                          |
|------------|----------------------------------------------------|----------------|--------------------------------------------------|
| --mode | pradiction using the pre trained model or training | True           | "predict", "train"                               |
| --init| Choose the protein-encoding                        | True           | "Kidera" ,"Random",  "Kidera+bio", "ESM-2"       |
|--model| Choose the model architecture                      | True           | "BiLSTM" ,"GCN", "Boosting"                      |
|--epi | Choose the epitope sequences                       | True           | "Linear", "Nonlinear", "Both"                    |
|--train_path | Path to the training data                          | only for train | path for the trainig data                        |
|--test_path | Path to the testing data                           | only for train | path for the test data                           |    
|--test_seq_input | Path to predict data in FASTA format               | False          | path for the data in string                      |
|--test_pdb_path | Path to the predict data of PDB files              | False          | path for folder with PDB files                   |
|--test_pdb_list | Path to the predict data of PDB IDs in a list      | False          | path for file including PDB IDs   |

