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
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(x)=\sum_{\forall y}{P(x,y)}=P(x|y=1) * P(y=1) + P(x|y=0) * P(y=0)" title="P(x)=\sum_{\forall y}{P(x,y)}=P(x|y=1) * P(y=1) + P(x|y=0) * P(y=0)" />

$$x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}.$$

## INTERPRETING MODELS
The learned weights for any classification linear model are not more informative than the learned weights for any other classification linear model. When classifying examples, the magnitude of the learned weights tells you the importance placed upon a feature and the sign of the learned weight tells you whether the feature is used to discriminate in favour of the "positive" class or against the "positive" class (think about one vs rest multiclass classifier).
 
For a decision tree or each decision tree in a random forest, you should directly inspect the hierarchy of rules in the tree to interpret the learned model.
 
For a convolutional neural network in computer vision, you should directly inspect the first layer (convolutional) of the neural net to understand the low-level features learned.

## TUNING HYPERPARAMETERS
Always perform cross-validation to tune your model on some validation set or just use k-fold cross-validation.

Machine learning models are hyperparameter sensitive and require tuning. You cannot just run machine learning models as a black box. You need to tune the models, inspect the learned rules/weights and reason about the output of the models.

## Accuracy, Precision, Recall, F1
Accuracy = 

