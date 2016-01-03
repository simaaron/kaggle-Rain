How Much Did It Rain? II
=================
Kaggle competition winning solution
-----------------------------
This document describes how to generate the winning solution to the Kaggle competition [*How Much Did It Rain? II*](https://www.kaggle.com/c/how-much-did-it-rain-ii).

Further documentation on the method can be found in this [blog post](http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/).

## Generating the solution  

### Install the dependencies
The models are written in Python 2.7 and makes use of the NumPy, scikit-learn, and pandas packages. These can be installed individually via `pip` or all together in a free Python distribution such as [Anaconda](https://www.continuum.io/downloads).

Theano can be installed and configured to use any available NVIDIA GPUs by following the instructions [here](http://deeplearning.net/software/theano/install.html) and [here](http://deeplearning.net/software/theano/tutorial/using_gpu.html). The Lasagne package often requires the latest version of Theano; a simple `pip install Theano` may give a version that is out-of-date (see Lasagne documentation for details).  

Lasagne can be installed by following the instructions [here](http://lasagne.readthedocs.org/en/latest/user/installation.html).


### Download the code
To download the code run:

```
git clone git://github.com/simaaron/kaggle-Rain.git
```
This includes an empty `data` folder.


### Download the training and test data
The training and test data can be downloaded from the Kaggle competition webpage at this [link](https://www.kaggle.com/c/how-much-did-it-rain-ii/data). The two extracted files `train.csv` and `test.csv` should be placed in the `data` folder (see above). 

Note: the benchmark sample solution and code provided by Kaggle are not required.

### Preprocess the data
Replace the `NaN` entries with zeros (training and test data) and remove the outliers (training data only) by running:

```
python data_preprocessing.py
```
This will also create three additional `train`, `valid`, and `test` folders. The size of the validation holdout subset and the outlier threshold expected rainfall value can be changed in the above Python script.

### Augment the data sets with *dropin* copies
Create random augmentation copies of the datasets by running:

```
python data_augmentation_train.py
python data_augmentation_valid.py
python data_augmentation_test.py
```
This creates 61 randomly augmented copies of the preprocessed training and test data sets and one of the validation holdout set. Note that each copy is > 2GB in size. If there is an issue with insufficient hard disk space, one should modify the training script `NNregression_*.py` and test script `NNprediction_*.py` to perform these augmentations dynamically.

The number of copies can be changed in the above scripts.

### Train the networks
The two best models can be trained by running:

```
python NNregression_v1.py -v=1
python NNregression_v2.py -v=2
```
The list of functions corresponding to the different models can be found in the Python script `NN_architectures.py`. The remaining models can be trained by simply modifying the corresponding function import and call within either script above and then saving and running a new script:

```
python NNregression_v*.py -v=*
``` 
The outputs from different models are continually saved into separate output folders. These include the files `training_scores.txt` and `validation_scores.txt` which, for monitoring purposes, give the evolution of the training and validation errors respectively. The file `model.npz` is the current best fitting set of model parameters (w.r.t. the validation holdout set), and the `last_learn_rate.txt` records the current (decayed) learning rate.

### Generate predictions from augmented test sets
The set of 61 augmented test set predictions from the model 'v1' can be obtained by running:

```
for j in `seq 0 60`;
do
	python NNpredictor_v1.py -rd=$j
done
```
The predictions from the pre-trained model included in the code download can be obtained by running:

```
for j in `seq 0 60`;
do
	python NNpredictor_v1.py -rd=$j -i pretrained_model_v1.npz
done
```



### Average the augmented predictions
The predictions from different augmented copies can be combined by running:

```
python ensembling.py -v=1 -nr=61
```

This averages the 61 predictions of the model 'v1' and saves it to the file `ens_submission_v1_61ave_mean.csv`.

The individual predictions from the models 'v1' and 'v2' would place one 2nd/3rd in the competition. A straight average of the two solutions would be sufficient for 1st place. 










