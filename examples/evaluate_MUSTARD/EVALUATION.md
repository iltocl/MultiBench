# Evaluation

*Note that we used some models from [Multibench](https://github.com/pliang279/MultiBench.git)*. 

Then, we recommend to clone our adapted repository branch (mm-lle-mustard):
```sh
# the original repository
# git clone https://github.com/pliang279/MultiBench.git 

# our mm-lle-mustard branch repository
git clone --branch mm-lle-mustard https://github.com/iltocl/MultiBench.git
cd MultiBench
```
Create a virtual environment and install the required packages
```sh
python -m venv ENV_NAME
./ENV_NAME/Scripts/activate
pip install -r requirements.txt   
```

Work directory ```evaluate_MUSTARD```
```sh
cd /examples/evaluate_MUSTARD/ 
```
Make sure the following notebooks and scripts are in ```evaluate_MUSTARD```: 
- ```evaluate_LLEasAttVectors_MLP_votes.ipynb```
- ```evaluate_LLEasAttVectors_TR_MLP.ipynb```
- ```evaluate_with_MLP.py```
- ```evaluate_with_Transformer.py```

Create dir ```LLEasAtt_representations``` and move the pkl files that were previously generated.
```sh
mkdir LLEasAtt_representations
# add pkl files (e.g. mustard_bagVectors_UM_N_10_T_V_A.pkl)
```

The main adaptations of the repository structure:
```sh
├── code
│   └── load_AffectDatasets.py # adapted version                                
├── # default directories
├── examples
│   ├── data   
│   │   └── sarcasm.pkl # make sure you add the MUSTARD dataset file
│   ├── evaluate_MUSTARD                        
│   │   ├── LLEasAtt_representations # pkl files previously obtained (e.g. mustard_bagVectors_UM_N_10_T_V_A.pkl)
│   │   ├── evaluate_LLEasAttVectors_MLP_votes.ipynb # experiments examples
│   │   ├── evaluate_LLEasAttVectors_TR_MLP.ipynb # experiments examples
│   │   ├── evaluate_with_MLP.py # functions
│   │   ├── evaluate_with_Transformer.py #functions
│   │   └── EVALUATION.md 
│   ├── requirements.txt # to create and adapt the virtual environment
└── # default directories
```

You can find: 
- ```sarcasm.pkl``` in [sarcasm](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7?usp=drive_link)
- LLEasAtt representations (pkl files) in [generated_representations_MUSTARD](https://drive.google.com/drive/folders/1nWBFPyhx4-NmE4_2wIwyiiVcsQN_6QU6?usp=drive_link)


