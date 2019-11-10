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

