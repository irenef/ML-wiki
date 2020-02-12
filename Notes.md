## CNN vs RNN
* CNN:
  * CNN take a fixed size input and generate fixed-size outputs.
  * CNN is a type of feed-forward artificial neural network - are variations of multilayer perceptrons which are designed to use minimal amounts of preprocessing.
  * CNNs use connectivity pattern between its neurons is inspired by the organization of the animal visual cortex, whose individual neurons are arranged in such a way that they respond to overlapping regions tiling the visual field.
  * CNNs are ideal for images and videos processing.
* RNN:
  * RNN can handle arbitrary input/output lengths.
  * RNN, unlike feedforward neural networks, can use their internal memory to process arbitrary sequences of inputs.
  * Recurrent neural networks use time-series information (i.e. what I spoke last will impact what I will speak next.)
  * RNNs are ideal for text and speech analysis.

## DATA VISUALISATION
In general, it is a good idea to use PCA/ICA/Random projection to generate visualisations of the data. This may or may not help build intuition for why particular models work over others. If you have 2 or 3-dimensional data (this is the number of features), you have no excuse for not doing data visualisation.

## MODEL DIFFERENCES
Difference between logistic regression and perceptron on linearly separable data: logistic regression is more robust to outliers.
 
Difference between linear SVM and perceptron on linearly separable data: SVM finds the separating hyperplane of maximum margin while the perceptron just finds any separating hyperplane.
 
Perceptrons do not work on non-linearly separable data (you cannot train them and get them to converge on non-linearly separable data) while soft-margin SVMs and logistic regression do. This is exactly why nobody uses perceptrons in machine learning work.
 
Difference between neural networks and linear models: neural networks learn a hierarchy of nonlinear features while discriminative linear models learn linear hyperplanes (or linear hyperplanes in the kernel space for kernel SVM). 

## GENERATIVE MODELS VERSUS DISCRIMINATIVE MODELS
Recall that each labelled example is an instance/feature-vector and label pair (x, y). A generative model learns the joint probability distribution of the instances and the labels P(x, y). A discriminative model learns the posterior probability distribution P(y | x). In general, the features learned by a generative model are associated with a particular class (i.e. correlated with that class) and the features learned by a discriminative model are used to distinguish between one class and the rest in a one versus rest multiclass classification scenario.

Examples of discriminative models: SVM, logistic regression, perceptron and KNN.
Examples of generative models: HMM, GMM and Naive Bayes.

## Bayes Rule 
P(y=1|x) = P(x|y=1) * P(y=1) / P(x)

P(x) = sum_y{P(x,y)} = P(x|y=1) * P(y=1) + P(x|y=0) * P(y=0)


## INTERPRETING MODELS
The learned weights for any classification linear model are not more informative than the learned weights for any other classification linear model. When classifying examples, the magnitude of the learned weights tells you the importance placed upon a feature and the sign of the learned weight tells you whether the feature is used to discriminate in favour of the "positive" class or against the "positive" class (think about one vs rest multiclass classifier).
 
For a decision tree or each decision tree in a random forest, you should directly inspect the hierarchy of rules in the tree to interpret the learned model.
 
For a convolutional neural network in computer vision, you should directly inspect the first layer (convolutional) of the neural net to understand the low-level features learned.

### Accuracy, Precision, Recall, F1
Accuracy  = (TP+TN) / (TP+TN+FP+FN)

Precision = (TP) / (TP+FP) --> when the cost of false positive is high 

Recall    = (TP) / (TP+FN) --> when the cost of false negative is high 

F1        = 2* (precision * recall) / (precision + recall)

## TUNING HYPERPARAMETERS
Always perform cross-validation to tune your model on some validation set or just use k-fold cross-validation.

Machine learning models are hyperparameter sensitive and require tuning. You cannot just run machine learning models as a black box. You need to tune the models, inspect the learned rules/weights and reason about the output of the models.

## DETECTING OVERFITTING
It is a good idea to plot how training error and validation error change with the number of updates for gradient-based machine learning models. This will help you detect overfitting and see if regularisation is appropriate.

## REGULARISATION
The purpose of regularisation is to prevent overfitting by constraining the weights and/or applying a penalty in the objective function to the weights of a model. There are other types of regularisation too such as data augmentation for training convolutional neural networks.

## COMPARISON TO BASELINE
Always compare the performance of your machine learning models to some kind of baseline on the test set. This could be a trivial baseline such as predicting the most common class in the test set for all test set examples or a standard machine learning model.

## FEATURE ENGINEERING AND FEATURE SELECTION
Typically, you need to try out multiple types of features to see if you can get better performance out of a particular machine learning model for your machine learning task. Feature selection tends to improve performance but not always. You never use feature selection in computer vision as far as I know.

## FEATURE STANDARDISATION
For certain machine learning models such as SVMs and logistic regression, it is a good idea to standardise the features to have a mean of 0 and a standard deviation of 1. You calculate the feature means and standard deviations using the training set and you apply these means/standard deviations to normalise the features in the train/valid/test sets. This is done to make sure that the models learned are not sensitive to scaling.

## GRADIENT LEARNING
When we wish to perform gradient learning to fit a model to data, we typically use so-called gradient following.
 
Suppose we have an objective function that is a cost function. We want optimisation to find the hypothesis h in the hypothesis space H that is some kind of minimum (local minimum or global minimum) of the objective function. We simply take the gradient wrt the model parameters of the objective function and take a step of size eta in the direction of the negative gradient in the parameter space to fit the model in each parameter update iteration. The direction of the negative gradient is the direction of steepest descent of the objective function at the current parameter setting.
 
Suppose we have an objective function that is a reward function. We want optimisation to find the hypothesis h in the hypothesis space H that is some kind of maximum (local maximum or global maximum) of the objective function. We simply take the gradient wrt the model parameters of the objective function and take a step of size eta in the direction of the positive gradient in the parameter space to fit the model in each parameter update iteration. The direction of the gradient is the direction of steepest ascent of the objective function at the current parameter setting.
 
Gradient learning typically finds local minima or local maxima of the objective function. It is not guaranteed to find the global minimum or the global maximum.

## PCA
[Principal Component Analysis (PCA) clearly explained by StatQuest](https://www.youtube.com/watch?v=_UVHneBUBW0 )

## Autoencoder
[Autoencoder: neural networks for unsupervised learning] (https://sefiks.com/2018/03/21/autoencoder-neural-networks-for-unsupervised-learning/)
