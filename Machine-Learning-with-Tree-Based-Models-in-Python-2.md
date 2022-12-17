## The Bias-Variance Tradeoff

<p class="chapter__description">
The bias-variance tradeoff is one of the fundamental concepts in
supervised machine learning. In this chapter, you’ll understand how to
diagnose the problems of overfitting and underfitting. You’ll also be
introduced to the concept of ensembling where the predictions of several
models are aggregated to produce predictions that are more robust.
</p>

### Generalization Error

#### Complexity, bias and variance

<p>
In the video, you saw how the complexity of a model labeled
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="0" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>
influences the bias and variance terms of its generalization error.<br>
Which of the following correctly describes the relationship between
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="1" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>’s
complexity and
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="2" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>’s
bias and variance terms?
</p>

-   [ ] As the complexity of
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="3" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>
    decreases, the bias term decreases while the variance term
    increases.
-   [ ] As the complexity of
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="4" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>
    decreases, both the bias and the variance terms increase.
-   [ ] As the complexity of
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="5" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>
    increases, the bias term increases while the variance term
    decreases.
-   [x] As the complexity of
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="6" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-texatom texclass="ORD"><mjx-mover><mjx-over style="padding-bottom: 0.06em; padding-left: 0.237em; margin-bottom: -0.531em;"><mjx-mo class="mjx-n"><mjx-c class="mjx-c5E"></mjx-c></mjx-mo></mjx-over><mjx-base><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D453 TEX-I"></mjx-c></mjx-mi></mjx-base></mjx-mover></mjx-texatom></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>f</mi><mo stretchy="false">^</mo></mover></mrow></math></mjx-assistive-mml></mjx-container>
    increases, the bias term decreases while the variance term
    increases.

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Great work! You’re now able to relate model complexity to bias and
variance!
</p>

#### Overfitting and underfitting

<p>
In this exercise, you’ll visually diagnose whether a model is
overfitting or underfitting the training set.
</p>
<p>
For this purpose, we have trained two different models
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="8" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container>
and
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="9" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi></math></mjx-assistive-mml></mjx-container>
on the auto dataset to predict the <code>mpg</code> consumption of a car
using only the car’s displacement (<code>displ</code>) as a feature.
</p>
<p>
The following figure shows you scatterplots of <code>mpg</code> versus
<code>displ</code> along with lines corresponding to the training set
predictions of models
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="10" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container>
and
<mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="11" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi></math></mjx-assistive-mml></mjx-container>
in red.
</p>
<p>
<img src="https://assets.datacamp.com/production/repositories/1796/datasets/f905399bc06da86c2a3af27b20717de5a777e6e1/diagnose-problems.jpg" alt="diagnose">
</p>
<p>
Which of the following statements is true?
</p>

-   [ ]
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="12" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container>
    suffers from high bias and overfits the training set.
-   [ ]
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="13" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D434 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>A</mi></math></mjx-assistive-mml></mjx-container>
    suffers from high variance and underfits the training set.
-   [x]
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="14" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi></math></mjx-assistive-mml></mjx-container>
    suffers from high bias and underfits the training set.
-   [ ]
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="15" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mi class="mjx-i"><mjx-c class="mjx-c1D435 TEX-I"></mjx-c></mjx-mi></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mi>B</mi></math></mjx-assistive-mml></mjx-container>
    suffers from high variance and underfits the training set.

<p class="dc-completion-pane__message dc-u-maxw-100pc">
Absolutely! Model B is not able to capture the nonlinear dependence of
<code>mpg</code> on <code>displ</code>.
</p>

### Diagnose bias and variance problems

#### Instantiate the model

<p>
In the following set of exercises, you’ll diagnose the bias and variance
problems of a regression tree. The regression tree you’ll define in this
exercise will be used to predict the mpg consumption of cars from the
auto dataset using all available features.
</p>
<p>
We have already processed the data and loaded the features matrix
<code>X</code> and the array <code>y</code> in your workspace. In
addition, the <code>DecisionTreeRegressor</code> class was imported from
<code>sklearn.tree</code>.
</p>

<li>
Import <code>train_test_split</code> from
<code>sklearn.model_selection</code>.
</li>
<li>
Split the data into 70% train and 30% test.
</li>
<li>
Instantiate a <code>DecisionTreeRegressor</code> with max depth 4 and
<code>min_samples_leaf</code> set to 0.26.
</li>

``` python
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=SEED)
```

<p class>
Great work! In the next exercise, you’ll evaluate <code>dt</code>’s CV
error.
</p>

#### Evaluate the 10-fold CV error

<p>
In this exercise, you’ll evaluate the 10-fold CV Root Mean Squared Error
(RMSE) achieved by the regression tree <code>dt</code> that you
instantiated in the previous exercise.
</p>
<p>
In addition to <code>dt</code>, the training data including
<code>X_train</code> and <code>y_train</code> are available in your
workspace. We also imported <code>cross_val_score</code> from
<code>sklearn.model_selection</code>.
</p>
<p>
Note that since <code>cross_val_score</code> has only the option of
evaluating the negative MSEs, its output should be multiplied by
negative one to obtain the MSEs. The CV RMSE can then be obtained by
computing the square root of the average MSE.
</p>

<li>
Compute <code>dt</code>‘s 10-fold cross-validated MSE by setting the
<code>scoring</code> argument to <code>’neg_mean_squared_error’</code>.
</li>
<li>
Compute RMSE from the obtained MSE scores.
</li>

``` python
# edited/added
from sklearn.model_selection import cross_val_score

# Compute the array containing the 10-folds CV MSEs
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, 
                                  scoring='neg_mean_squared_error', 
                                  n_jobs=-1) 
                                  
# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))
```

    ## CV RMSE: 5.14

<p class>
Great work! A very good practice is to keep the test set untouched until
you are confident about your model’s performance. CV is a great
technique to get an estimate of a model’s performance without affecting
the test set.
</p>

#### Evaluate the training error

<p>
You’ll now evaluate the training set RMSE achieved by the regression
tree <code>dt</code> that you instantiated in a previous exercise.
</p>
<p>
In addition to <code>dt</code>, <code>X_train</code> and
<code>y_train</code> are available in your workspace.
</p>
<p>
Note that in scikit-learn, the MSE of a model can be computed as
follows:
</p>
<pre><code>MSE_model = mean_squared_error(y_true, y_predicted)
</code></pre>
<p>
where we use the function <code>mean_squared_error</code> from the
<code>metrics</code> module and pass it the true labels
<code>y_true</code> as a first argument, and the predicted labels from
the model <code>y_predicted</code> as a second argument.
</p>

<li>
Import <code>mean_squared_error</code> as <code>MSE</code> from
<code>sklearn.metrics</code>.
</li>
<li>
Fit <code>dt</code> to the training set.
</li>
<li>
Predict <code>dt</code>’s training set labels and assign the result to
<code>y_pred_train</code>.
</li>
<li>
Evaluate <code>dt</code>’s training set RMSE and assign it to
<code>RMSE_train</code>.
</li>

``` python
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
```

    ## DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=1)

``` python
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))
```

    ## Train RMSE: 5.15

<p class>
Awesome! Notice how the training error is roughly equal to the 10-folds
CV error you obtained in the previous exercise.
</p>

#### High bias or high variance?

<p>
In this exercise you’ll diagnose whether the regression tree
<code>dt</code> you trained in the previous exercise suffers from a bias
or a variance problem.
</p>
<p>
The training set RMSE (<code>RMSE_train</code>) and the CV RMSE
(<code>RMSE_CV</code>) achieved by <code>dt</code> are available in your
workspace. In addition, we have also loaded a variable called
<code>baseline_RMSE</code> which corresponds to the root mean-squared
error achieved by the regression-tree trained with the <code>disp</code>
feature only (it is the RMSE achieved by the regression tree trained in
chapter 1, lesson 3). Here <code>baseline_RMSE</code> serves as the
baseline RMSE above which a model is considered to be underfitting and
below which the model is considered ‘good enough’.
</p>
<p>
Does <code>dt</code> suffer from a high bias or a high variance problem?
</p>

-   [ ] <code>dt</code> suffers from high variance because
    <code>RMSE_CV</code> is far less than <code>RMSE_train</code>.
-   [x] <code>dt</code> suffers from high bias because
    <code>RMSE_CV</code>
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="16" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2248"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≈</mo></math></mjx-assistive-mml></mjx-container>
    <code>RMSE_train</code> and both scores are greater than
    <code>baseline_RMSE</code>.
-   [ ] <code>dt</code> is a good fit because <code>RMSE_CV</code>
    <mjx-container class="MathJax CtxtMenu_Attached_0" jax="CHTML" role="presentation" tabindex="0" ctxtmenu_counter="17" style="font-size: 116.7%; position: relative;"><mjx-math class="MJX-TEX" aria-hidden="true"><mjx-mo class="mjx-n"><mjx-c class="mjx-c2248"></mjx-c></mjx-mo></mjx-math><mjx-assistive-mml role="presentation" unselectable="on" display="inline"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>≈</mo></math></mjx-assistive-mml></mjx-container>
    <code>RMSE_train</code> and both scores are smaller than
    <code>baseline_RMSE</code>.

<p class>
Correct! <code>dt</code> is indeed underfitting the training set as the
model is too constrained to capture the nonlinear dependencies between
features and labels.
</p>

### Ensemble Learning

#### Define the ensemble

<p>
In the following set of exercises, you’ll work with the
<a href="https://www.kaggle.com/jeevannagaraj/indian-liver-patient-dataset">Indian
Liver Patient Dataset</a> from the UCI Machine learning repository.
</p>
<p>
In this exercise, you’ll instantiate three classifiers to predict
whether a patient suffers from a liver disease using all the features
present in the dataset.
</p>
<p>
The classes <code>LogisticRegression</code>,
<code>DecisionTreeClassifier</code>, and
<code>KNeighborsClassifier</code> under the alias <code>KNN</code> are
available in your workspace.
</p>

<li>
Instantiate a Logistic Regression classifier and assign it to
<code>lr</code>.
</li>
<li>
Instantiate a KNN classifier that considers 27 nearest neighbors and
assign it to <code>knn</code>.
</li>
<li>
Instantiate a Decision Tree Classifier with the parameter
<code>min_samples_leaf</code> set to 0.13 and assign it to
<code>dt</code>.
</li>

``` python
# edited/added
df = pd.read_csv("archive/Machine-Learning-with-Tree-Based-Models-in-Python/datasets/indian_liver_patient_preprocessed.csv")
X = df.drop(columns = ['Liver_disease'])
y = df['Liver_disease']
X_train, X_test,  y_train, y_test = sklearn.model_selection.train_test_split(X,y)
from sklearn.neighbors import KNeighborsClassifier

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNeighborsClassifier(n_neighbors=27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
```

<p class>
Great! In the next exercise, you will train these classifiers and
evaluate their test set accuracy.
</p>

#### Evaluate individual classifiers

<p>
In this exercise you’ll evaluate the performance of the models in the
list <code>classifiers</code> that we defined in the previous exercise.
You’ll do so by fitting each classifier on the training set and
evaluating its test set accuracy.
</p>
<p>
The dataset is already loaded and preprocessed for you (numerical
features are standardized) and it is split into 70% train and 30% test.
The features matrices <code>X_train</code> and <code>X_test</code>, as
well as the arrays of labels <code>y_train</code> and
<code>y_test</code> are available in your workspace. In addition, we
have loaded the list <code>classifiers</code> from the previous
exercise, as well as the function <code>accuracy_score()</code> from
<code>sklearn.metrics</code>.
</p>

<li>
Iterate over the tuples in <code>classifiers</code>. Use
<code>clf_name</code> and <code>clf</code> as the <code>for</code> loop
variables:
<li>
Fit <code>clf</code> to the training set.
</li>
<li>
Predict <code>clf</code>’s test set labels and assign the results to
<code>y_pred</code>.
</li>
<li>
Evaluate the test set accuracy of <code>clf</code> and print the result.
</li>
</li>

``` python
from sklearn.metrics import accuracy_score

# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
 
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
   
    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) 
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
```

    ## LogisticRegression(random_state=1)
    ## Logistic Regression : 0.697
    ## KNeighborsClassifier(n_neighbors=27)
    ## K Nearest Neighbours : 0.676
    ## DecisionTreeClassifier(min_samples_leaf=0.13, random_state=1)
    ## Classification Tree : 0.703
    ## 
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    ## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    ## 
    ## Increase the number of iterations (max_iter) or scale the data as shown in:
    ##     https://scikit-learn.org/stable/modules/preprocessing.html
    ## Please also refer to the documentation for alternative solver options:
    ##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ##   n_iter_i = _check_optimize_result(
    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/base.py:441: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
    ##   warnings.warn(

<p class>
Great work! Notice how Logistic Regression achieved the highest accuracy
of 74.1%.
</p>

#### Better performance with a Voting Classifier

<p>
Finally, you’ll evaluate the performance of a voting classifier that
takes the outputs of the models defined in the list
<code>classifiers</code> and assigns labels by majority voting.
</p>
<p>
<code>X_train</code>, <code>X_test</code>,<code>y_train</code>,
<code>y_test</code>, the list <code>classifiers</code> defined in a
previous exercise, as well as the function <code>accuracy_score</code>
from <code>sklearn.metrics</code> are available in your workspace.
</p>

<li>
Import <code>VotingClassifier</code> from <code>sklearn.ensemble</code>.
</li>
<li>
Instantiate a <code>VotingClassifier</code> by setting the parameter
<code>estimators</code> to <code>classifiers</code> and assign it to
<code>vc</code>.
</li>
<li>
Fit <code>vc</code> to the training set.
</li>
<li>
Evaluate <code>vc</code>’s test set accuracy using the test set
predictions <code>y_pred</code>.
</li>

``` python
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

# Fit vc to the training set
vc.fit(X_train, y_train)

# Evaluate the test set predictions
```

    ## VotingClassifier(estimators=[('Logistic Regression',
    ##                               LogisticRegression(random_state=1)),
    ##                              ('K Nearest Neighbours',
    ##                               KNeighborsClassifier(n_neighbors=27)),
    ##                              ('Classification Tree',
    ##                               DecisionTreeClassifier(min_samples_leaf=0.13,
    ##                                                      random_state=1))])
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
y_pred = vc.predict(X_test)

# Calculate accuracy score
```

    ## /Users/macos/Library/r-miniconda/envs/r-reticulate/lib/python3.8/site-packages/sklearn/base.py:441: UserWarning: X does not have valid feature names, but KNeighborsClassifier was fitted with feature names
    ##   warnings.warn(

``` python
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))
```

    ## Voting Classifier: 0.690

<p class>
Great work! Notice how the voting classifier achieves a test set
accuracy of 76.4%. This value is greater than that achieved by
<code>LogisticRegression</code>.
</p>