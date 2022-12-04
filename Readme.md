## Sentiment Analysis
In this project our main aim is to develop an efficient approach for analysing
depressive tweets using various Machine Learning Algorithms like vector space
model, tf-idf, naive Bayes etc.

### Methodology
1. Data Collection
2. Data Cleaning
3. Data Pre-processing
4. Training Dataset Formation
5. Model Building
6. Prediction on testing dataset

### Architecture
<img src="MIsc\image.png" ></img>

### NetworkX 
<img src="MIsc\network_x.png" ></img>


## Concepts
### IR Model used in Project
Vectorization our data set using TF-IDF
Machine learning algorithms often use numerical data, so when dealing with textual
data or any natural language processing (NLP) task, a sub-field of ML/AI dealing with
text, that data first needs to be converted to a vector of numerical data by a process
known as vectorization. TF-IDF vectorization involves calculating the TF-IDF score
for every word in your corpus relative to that document and then putting that
information into a vector (see image below using example documents “A” and “B”).
Thus each document in your corpus would have its own vector, and the vector would
have a TF-IDF score for every single word in the entire collection of documents.
Once you have these vectors you can apply them to various use cases such as
seeing if two documents are similar by comparing their TF-IDF vector using cosine
similarity.
### Random Forest Model
A random forest classifier. A random forest is a meta estimator that fits a number of
decision tree classifiers on various sub-samples of the dataset and uses averaging
to improve the predictive accuracy and control over-fitting.
Naive Bayes Model
In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers"
based on applying Bayes' theorem with strong (naive) independence assumptions
between the features . They are among the simplest Bayesian network models, but
coupled with kernel density estimation, they can achieve high accuracy levels.
Naive Bayes classifiers are highly scalable, requiring a number of parameters linear
in the number of variables (features/predictors) in a learning problem.
Maximum-likelihood training can be done by evaluating a closed-form expression,
which takes linear time, rather than by expensive iterative approximation as used for
many other types of classifiers.
### Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions
An approach to semi-supervised learning is proposed that is based on a Gaussian random
field model. Labelled and unlabeled data are represented as vertices in a weighted graph,
with edge weights encoding the similarity between instances. A new approach to
semi-supervised learning that is based on a random field model defined on a weighted graph
over the unlabeled and labelled data, where the weights are given in terms of a similarity
function between instances.

