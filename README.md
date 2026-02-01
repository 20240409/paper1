# A Multi-Stage Learning Framework Integrating Reconstruction and Classification for Region-Level EEG-Based Fatigue Detection
This repository provides the implementation of our paper on EEG-based fatigue detection. We propose a multi-stage learning framework that integrates joint pre-training and supervised fine-tuning to improve fatigue recognition performance.
<p align="center">
  <img src="figure/main_model.png" width="70%">
</p>
The figure above illustrates our proposed two-stage EEG-based fatigue detection framework. 
The model consists of two main stages:

1. **Joint Training Stage**  
   In this stage, the model is jointly trained with reconstruction and classification tasks, all parameters in the model are trainable.

2. **Supervised Fine-tuning Stage**  
   In this stage, the pretrained encoder is transferred to a classification model consisting of the frozen encoder and a newly initialized classifier.
   All encoder parameters are frozen, and only the classifier is optimized under labeled supervision.

# requirements
Python version is 3.9, run the following code to install requirements:
```bash
pip install -r requirements.txt
```

# Datasets
You can click [here](https://bcmi.sjtu.edu.cn/~seed/seed-vig.html) to download the SEED‑VIG dataset. Then, run the following code to store the DE features into the
the `dataset/data` folder.
```bash
python dataset/DE.py
```
The data will be organized as

dataset/

├─ data/

│  ├─ DE_0

│  ├─ DE_1

│  ├─ ...

│  ├─ DE_22
      
│  ├─ label_0

│  ├─ label_1

│  ├─ ...

│  ├─ label_22

Our self-Made dataset is not planned to be made public at this time.

# Train And Test
You can run the following code to train the model and modify the model configuration parameters in the code to evaluate its performance 
under different experimental settings. The model parameters obtained from both training stages will be saved in the `run/save_dict` folder.
```bash
python run/train.py
```
After training is complete, you can evaluate the model on the test set using your trained model, 
or you can directly run the following code to load our checkpoints and reproduce the results of our experiments.
```bash
python run/test.py
```