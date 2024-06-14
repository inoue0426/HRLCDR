HRLCDR
===============================
Source code and data for "Hypergraph representation learning for cancer drug response prediction"

![Framework of HRLCDR](https://github.com/weiba/HRLCDR/blob/master/HRLCDR.jpg)  
# Requirements
All implementations of HRLCDR are based on PyTorch.HRLCDR requires the following dependencies:
- python==3.10.8
- pytorch==1.13.0
- numpy==1.23.4+mkl
- scipy==1.9.3
- pandas==1.5.1
- scikit-learn=1.1.3
# Data
- Data defines the data used by the model
	- GDSC/Data/
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- gene_feature.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
		- null_mask.csv records the null values in the cell line-drug association matrix.
        - 5-fold_CV.csv records the 5-fold cross-validation set for the classification task.
        - testset.csv records the independent test set for the classification task.
        - 5-fold_CV_ic50.csv records the 5-fold cross-validation set for the regression task.
        - testset_ic50.csv records the independent test set for the regression task.
	- CCLE/Data/
		- cell_drug.csv records the log IC50 association matrix of cell line-drug.
		- cell_drug_binary.csv records the binary cell line-drug association matrix.
		- gene_feature.csv records cell line gene expression features.
		- drug_feature.csv records the fingerprint features of drugs.
        - 5-fold_CV.csv records the 5-fold cross-validation set for the classification task.
        - testset.csv records the independent test set for the classification task.
        - 5-fold_CV_ic50.csv records the 5-fold cross-validation set for the regression task.
        - testset_ic50.csv records the independent test set for the regression task.
- load_data.py defines the data loading of the model.
- model.py defines the complete HRLCDR model.
- myutils.py defines the tool functions needed to run the entire algorithm as well as the evaluation metrics.
- sampler.py defines the sampling method of the model.
## Preprocessing your own data
Explanations on how you can process your own data and prepare it for HRLCDR running.

> In our study, we followed the data preprocessing steps described in MOLI[1] , and the data preprocessing code was derived from [MOLI](https://github.com/hosseinshn/MOLI).
> [1]Sharifi-Noghabi, H., et al. MOLI: multi-omics late integration with deep neural networks for drug response prediction. Bioinformatics 2019;35(14):i501-i509.

> You can download the processing data via the link above, or use the data provided in the GDSC/Data/ and CCLE/Data folders
# Usage
Once you have configured the environment, you can simply run HRLCDR by running command:
>We provide the division of the dataset in the ***sampler.py*** file, together with the ***main.py*** file, which will randomly select one-tenth of the positive samples and the same number of negative samples as the independent test set, and the rest of the data as the cross-validation set.

For classification tasks, you can use the following command:
```
python main.py
```
For regression tasks, you can use the following command:
```
python main_ic50.py
```
For novel cell line responses tasks, you can use the following command:
```
python main_new_cell.py
```
For novel drug responses tasks, you can use the following command:
```
python main_new_drug.py
```
Once the program has finished running, you are able to see the evaluation results of the independent test set directly on the screen! At the same time the program saves the test set's predictions and true labels in . /result_data/, and saves the test set and cross-validation set in ./Data/

All of the above commands complete a single experiment. The output of the main file consists of the true and predicted values of the test set as well as the divided test set and cross-validation set. In the subsequent statistical analysis, we will analyze the output of the master file. myutils.py file contains the performance of the whole experiment and all the tools needed for the analysis, such as the computation of AUC, AUPRC, PCC, SCC and RMSE. All functions are developed using PyTorch with CUDA support.

If you want to evaluate the test set results later, you can use the following command:
```
python evauate.py
```
# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
