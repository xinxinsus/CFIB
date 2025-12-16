# CFIB
The code of CFIB model: A cause fusion framework with information bottleneck for conversational  causal emotion entailment 


# Data Processing and Model Training Pipeline

## Data Processing

### Step 1: Raw Format Conversion
Convert the original data format to an intermediate format using:
```bash
python dateprocess_pro.py
```

### Step 2: Dialogue Text Processing
Process conversation texts into the input format required by the model:
```bash
python dataProcess_dual.py
```

## Model Training
To train the CFIB model, run:
```bash
python cfib.py
```
**Note:** Please refer to the original paper for specific hyperparameter configurations.

## Model Evaluation
To test the trained model, run:
```bash
python cfib_eval.py
```

## Parameter Settings
Please note that some parameter configurations in the code may differ from those reported in the original paper due to experimental adjustments during previous testing.  If there are discrepancies between parameter settings in the code and those specified in the original paper, please use the parameter values from the paper.
