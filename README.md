# PredLig

Python framework for Link Prediction based on supervised machine learning algorithms.
PredLig was designed to support the experimentation of diverse feature selection strategies.

The framework should allow us to:

1. Select an arbitrary set of data that represents a graph on which one desires to apply link predction technics.
2. Transform the dataset in a binary classification problem according to the supervisioned learning paradigm on link prediction.
3. Enhance the topological and agregated attributes in order to caracterize the vertex pairs for classification.
4. Experiment and choose of the classification algorithms used on the enhanced dataset.
5. Choose and configure of one feature selection strategies.
6. Apply of the feature selection strategy in a wrapper approach for each classification algorithm using K-fold crossed validation.


# Requirements

To run PredLig, beside the applications listed in the `requirements.txt`, you must install the bleeding edge version of Scikit Learn. In order to do that just follow the instructions listed [here](http://scikit-learn.org/dev/developers/advanced_installation.html#bleeding-edge).
