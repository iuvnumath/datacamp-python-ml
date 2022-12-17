## Applying logistic regression and SVM

<p class="chapter__description">
In this chapter you will learn the basics of applying logistic
regression and support vector machines (SVMs) to classification
problems. You’ll use the <code>scikit-learn</code> library to fit
classification models to real data.
</p>

### scikit-learn refresher

#### KNN classification

<p>
In this exercise you’ll explore a subset of the
<a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie
Review Dataset</a>. The variables <code>X_train</code>,
<code>X_test</code>, <code>y_train</code>, and <code>y_test</code> are
already loaded into the environment. The <code>X</code> variables
contain features based on the words in the movie reviews, and the
<code>y</code> variables contain labels for whether the review sentiment
is positive (+1) or negative (-1).
</p>
<p>
<em>This course touches on a lot of concepts you may have forgotten, so
if you ever need a quick refresher, download the
<a href="http://datacamp-community-prod.s3.amazonaws.com/eb807da5-dce5-4b97-a54d-74e89f14266b">scikit-learn
Cheat Sheet</a> and keep it handy!</em>
</p>

<li>
Create a KNN model with default hyperparameters.
</li>
<li>
Fit the model.
</li>
<li>
Print out the prediction for the test example 0.
</li>

``` python
# edited/added
import numpy as np
from sklearn.datasets import load_svmlight_file
X_train, y_train = load_svmlight_file('archive/Linear-Classifiers-in-Python/datasets/train_labeledBow.feat')
X_test, y_test = load_svmlight_file('archive/Linear-Classifiers-in-Python/datasets/test_labeledBow.feat')
X_train = X_train[11000:13000,:2500]
y_train = y_train[11000:13000]
y_train[y_train < 5] = -1.0
y_train[y_train >= 5] = 1.0
X_test = X_test[11000:13000,:2500]
y_test = y_test[11000:13000]
y_test[y_train < 5] = -1.0
y_test[y_train >= 5] = 1.0

from sklearn.neighbors import KNeighborsClassifier

# Create and fit the model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# Predict on the test features, print the results
```

    ## KNeighborsClassifier()

``` python
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)
```

    ## Prediction for test example 0: 1.0

<p class>
Nice work! Looks like you remember how to use <code>scikit-learn</code>
for supervised learning.
</p>

#### Comparing models

<p>
Compare k nearest neighbors classifiers with k=1 and k=5 on the
handwritten digits data set, which is already loaded into the variables
<code>X_train</code>, <code>y_train</code>, <code>X_test</code>, and
<code>y_test</code>. You can set k with the <code>n_neighbors</code>
parameter when creating the <code>KNeighborsClassifier</code> object,
which is also already imported into the environment.
</p>
<p>
Which model has a higher test accuracy?
</p>

``` python
# Create and fit the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
```

    ## KNeighborsClassifier(n_neighbors=1)

``` python
knn.score(X_test, y_test)

# Predict on the test features, print the results
```

    ## 0.1645

``` python
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)

# Create and fit the model
```

    ## Prediction for test example 0: 1.0

``` python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

    ## KNeighborsClassifier()

``` python
knn.score(X_test, y_test)

# Predict on the test features, print the results
```

    ## 0.056

``` python
pred = knn.predict(X_test)[0]
print("Prediction for test example 0:", pred)
```

    ## Prediction for test example 0: 1.0

-   [ ] k=1
-   [x] k=5

<p class>
Great! You’ve just done a bit of model selection!
</p>

#### Overfitting

<p>
Which of the following situations looks like an example of overfitting?
</p>

-   [ ] Training accuracy 50%, testing accuracy 50%.
-   [ ] Training accuracy 95%, testing accuracy 95%.
-   [x] Training accuracy 95%, testing accuracy 50%.
-   [ ] Training accuracy 50%, testing accuracy 95%.

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Great job! Looks like you understand overfitting.
</p>

### Applying logistic regression and SVM

#### Running LogisticRegression and SVC

<p>
In this exercise, you’ll apply logistic regression and a support vector
machine to classify images of handwritten digits.
</p>

<li>
Apply logistic regression and SVM (using <code>SVC()</code>) to the
handwritten digits data set using the provided train/validation split.
</li>
<li>
For each classifier, print out the training and validation accuracy.
</li>

``` python
# edited/added
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

    ## LogisticRegression()
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(

``` python
print(lr.score(X_train, y_train))
```

    ## 1.0

``` python
print(lr.score(X_test, y_test))

# Apply SVM and print scores
```

    ## 0.9488888888888889

``` python
svm = SVC()
svm.fit(X_train, y_train)
```

    ## SVC()

``` python
print(svm.score(X_train, y_train))
```

    ## 0.994060876020787

``` python
print(svm.score(X_test, y_test))
```

    ## 0.9844444444444445

<p class>
Nicely done! Later in the course we’ll look at the similarities and
differences of logistic regression vs. SVMs.
</p>

#### Sentiment analysis for movie reviews

<p>
In this exercise you’ll explore the probabilities outputted by logistic
regression on a subset of the
<a href="https://ai.stanford.edu/~amaas/data/sentiment/">Large Movie
Review Dataset</a>.
</p>
<p>
The variables <code>X</code> and <code>y</code> are already loaded into
the environment. <code>X</code> contains features based on the number of
times words appear in the movie reviews, and <code>y</code> contains
labels for whether the review sentiment is positive (+1) or negative
(-1).
</p>

<li>
Train a logistic regression model on the movie review data.
</li>
<li>
Predict the probabilities of negative vs. positive for the two given
reviews.
</li>
<li>
Feel free to write your own reviews and get probabilities for those too!
</li>

``` python
# edited/added
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file('archive/Linear-Classifiers-in-Python/datasets/train_labeledBow.feat')
X = X[11000:13000,:2500]
y = y[11000:13000]
y[y < 5] = -1.0
y[y >= 5] = 1.0
vocab = pd.read_csv('archive/Linear-Classifiers-in-Python/datasets/vocab.csv')['0'].values.tolist()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary = vocab)
def get_features(review):
    return vectorizer.transform([review])
  
# Instantiate logistic regression and train
lr = LogisticRegression()
lr.fit(X, y)

# Predict sentiment for a glowing review
```

    ## LogisticRegression()
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(

``` python
review1 = "LOVED IT! This movie was amazing. Top 10 this year."
review1_features = get_features(review1)
print("Review:", review1)
```

    ## Review: LOVED IT! This movie was amazing. Top 10 this year.

``` python
print("Probability of positive review:", lr.predict_proba(review1_features)[0,1])

# Predict sentiment for a poor review
```

    ## Probability of positive review: 0.8807769884058808

``` python
review2 = "Total junk! I'll never watch a film by that director again, no matter how good the reviews."
review2_features = get_features(review2)
print("Review:", review2)
```

    ## Review: Total junk! I'll never watch a film by that director again, no matter how good the reviews.

``` python
print("Probability of positive review:", lr.predict_proba(review2_features)[0,1])
```

    ## Probability of positive review: 0.9086001000263592

<p class>
Fantastic! The second probability would have been even lower, but the
word “good” trips it up a bit, since that’s considered a “positive”
word.
</p>

### Linear classifiers

#### Which decision boundary is linear?

<p>
Which of the following is a linear decision boundary?
</p>

<img src="archive/Linear-Classifiers-in-Python/datasets/decision_boundary.png">

-   [x] (1)
-   [ ] (2)
-   [ ] (3)
-   [ ] (4)

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Good job! You correctly identified the linear decision boundary.
</p>

#### Visualizing decision boundaries

<p>
In this exercise, you’ll visualize the decision boundaries of various
classifier types.
</p>
<p>
A subset of <code>scikit-learn</code>’s built-in <code>wine</code>
dataset is already loaded into <code>X</code>, along with binary labels
in <code>y</code>.
</p>

<li>
Create the following classifier objects with default hyperparameters:
<code>LogisticRegression</code>, <code>LinearSVC</code>,
<code>SVC</code>, <code>KNeighborsClassifier</code>.
</li>
<li>
Fit each of the classifiers on the provided data using a
<code>for</code> loop.
</li>
<li>
Call the <code>plot_4\_classifers()</code> function (similar to the code
<a href="https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html">here</a>),
passing in <code>X</code>, <code>y</code>, and a list containing the
four classifiers.
</li>

``` python
# edited/added
import matplotlib.pyplot as plt
X = np.array([[11.45,  2.4 ],
       [13.62,  4.95],
       [13.88,  1.89],
       [12.42,  2.55],
       [12.81,  2.31],
       [12.58,  1.29],
       [13.83,  1.57],
       [13.07,  1.5 ],
       [12.7 ,  3.55],
       [13.77,  1.9 ],
       [12.84,  2.96],
       [12.37,  1.63],
       [13.51,  1.8 ],
       [13.87,  1.9 ],
       [12.08,  1.39],
       [13.58,  1.66],
       [13.08,  3.9 ],
       [11.79,  2.13],
       [12.45,  3.03],
       [13.68,  1.83],
       [13.52,  3.17],
       [13.5 ,  3.12],
       [12.87,  4.61],
       [14.02,  1.68],
       [12.29,  3.17],
       [12.08,  1.13],
       [12.7 ,  3.87],
       [11.03,  1.51],
       [13.32,  3.24],
       [14.13,  4.1 ],
       [13.49,  1.66],
       [11.84,  2.89],
       [13.05,  2.05],
       [12.72,  1.81],
       [12.82,  3.37],
       [13.4 ,  4.6 ],
       [14.22,  3.99],
       [13.72,  1.43],
       [12.93,  2.81],
       [11.64,  2.06],
       [12.29,  1.61],
       [11.65,  1.67],
       [13.28,  1.64],
       [12.93,  3.8 ],
       [13.86,  1.35],
       [11.82,  1.72],
       [12.37,  1.17],
       [12.42,  1.61],
       [13.9 ,  1.68],
       [14.16,  2.51]])
y = np.array([ True,  True, False,  True,  True,  True, False, False,  True,
       False,  True,  True, False, False,  True, False,  True,  True,
        True, False,  True,  True,  True, False,  True,  True,  True,
        True,  True,  True,  True,  True, False,  True,  True,  True,
       False, False,  True,  True,  True,  True, False, False, False,
        True,  True,  True, False,  True])
        
def make_meshgrid(x, y, h=.02, lims=None):
    """Create a mesh of points to plot in
    
    Parameters
    ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional
        
    Returns
    -------
        xx, yy : ndarray
    """
    
    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
  
def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """Plot the decision boundaries for a classifier.
    
    Parameters
    ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,-1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(Z,extent=(np.min(xx), np.max(xx), np.min(yy), np.max(yy)), 
                        origin='lower', vmin=0, vmax=1, **params)
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out
  
def plot_classifier(X, y, clf, ax=None, ticks=False, proba=False, lims=None): 
    # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)
    
    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False
        
    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8, proba=proba)
    if proba:
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel('probability of red $\Delta$ class', fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
        #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors=\'k\', linewidth=1)
    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y==labels[0]], X1[y==labels[0]], cmap=plt.cm.coolwarm, 
                   s=60, c='b', marker='o', edgecolors='k')
        ax.scatter(X0[y==labels[1]], X1[y==labels[1]], cmap=plt.cm.coolwarm, 
                   s=60, c='r', marker='^', edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #     ax.set_xlabel(data.feature_names[0])
    #     ax.set_ylabel(data.feature_names[1])
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())
        #     ax.set_title(title)
    if show:
        plt.show()
    else:
        return ax
      
def plot_4_classifiers(X, y, clfs):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    for clf, ax, title in zip(clfs, sub.flatten(), ("(1)", "(2)", "(3)", "(4)")):
        # clf.fit(X, y)
        plot_classifier(X, y, clf, ax, ticks=True)
        ax.set_title(title)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define the classifiers
classifiers = [LogisticRegression(), LinearSVC(),
               SVC(), KNeighborsClassifier()]
               
# Fit the classifiers
for c in classifiers:
    c.fit(X, y)
    
# Plot the classifiers
```

    ## LogisticRegression()
    ## LinearSVC()
    ## SVC()
    ## KNeighborsClassifier()
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/svm/_base.py:1199: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
    ##   warnings.warn(

``` python
plot_4_classifiers(X, y, classifiers)
plt.show()
```

<img src="Linear-Classifiers-in-Python_files/figure-markdown_github/unnamed-chunk-5-1.png" width="672" />

<p class>
Nice! As you can see, logistic regression and linear SVM are linear
classifiers whereas KNN is not. The default SVM is also non-linear, but
this is hard to see in the plot because it performs poorly with default
hyperparameters. With better hyperparameters, it performs well.
</p>