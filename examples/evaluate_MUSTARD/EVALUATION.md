# Evaluation

Note that we used some models from Multibench. 

Then, we recommend to clone the repository:
```sh
git clone https://github.com/pliang279/MultiBench.git
cd MultiBench
```
Create a virtual environment and install the required packages
```sh
python -m venv <env_name>
./<env_name>/Scripts/activate
pip install -r requirements.txt   
```

Work directory ```evaluate_MUSTARD```
```sh
cd /examples/evaluate_MUSTARD/ 
```
Make sure the following notebooks and scripts are in ```evaluate_MUSTARD```: 
- ```evaluate_LLEasAttVectors_MLP.ipynb```
- ```evaluate_LLEasAttVectors_Transformer.ipynb```
- ```evaluate_with_MLP.py```
- ```evaluate_with_Transformer.py```


Create dir ```LLEasAtt_representations``` and move there the pkl files that were previously generated.
```sh
mkdir LLEasAtt_representations
```

The main additions into the structure:
```sh
├── code
│   └── load_AffectDatasets.py # adapted version                                
├── # default directories
├── examples
│   ├── data   
│   │   └── sarcasm.pkl # make to add this dataset
│   ├── evaluate_MUSTARD                        
│   │   ├── LLEasAtt_representations # pkl files previously obtained (e.g. mustard_bagVectors_UM_N_10_T_V_A.pkl)
│   │   ├── evaluate_LLEasAttVectors_MLP.ipynb # experiments examples
│   │   ├── evaluate_LLEasAttVectors_Transformer.ipynb # experiments examples
│   │   ├── evaluate_with_MLP.py # functions
│   │   └── EVALUATION.md
│   ├── requirements.txt # to create and adapt the virtual environment
└── # default directories
```

To replicate the reported results: 
- Table 1 execute ```evaluate_LLEasAttVectors_MLP.ipynb```
- Table 2 execute ```evaluate_LLEasAttVectors_Transformer.ipynb```
