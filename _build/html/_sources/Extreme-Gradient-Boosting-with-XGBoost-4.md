## Using XGBoost in pipelines

<p class="chapter__description">
Take your XGBoost skills to the next level by incorporating your models
into two end-to-end machine learning pipelines. You’ll learn how to tune
the most important XGBoost hyperparameters efficiently within a
pipeline, and get an introduction to some more advanced preprocessing
techniques.
</p>

### Review of pipelines using sklearn

#### Exploratory data analysis

<p>
Before diving into the nitty gritty of pipelines and preprocessing,
let’s do some exploratory analysis of the original, unprocessed
<a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">Ames
housing dataset</a>. When you worked with this data in previous
chapters, we preprocessed it for you so you could focus on the core
XGBoost concepts. In this chapter, you’ll do the preprocessing yourself!
</p>
<p>
A smaller version of this original, unprocessed dataset has been
pre-loaded into a <code>pandas</code> DataFrame called <code>df</code>.
Your task is to explore <code>df</code> in the Shell and pick the option
that is <strong>incorrect</strong>. The larger purpose of this exercise
is to understand the kinds of transformations you will need to perform
in order to be able to use XGBoost.
</p>

<li>
The DataFrame has 21 columns and 1460 rows.
</li>
<li>
The mean of the <code>LotArea</code> column is
<code>10516.828082</code>.
</li>
<li>
The DataFrame has missing values.
</li>
<strong>
<li>
The <code>LotFrontage</code> column has no missing values and its
entries are of type <code>float64</code>.
</li>
</strong>
<li>
The standard deviation of <code>SalePrice</code> is
<code>79442.502883</code>.
</li>
<p class>
Well done! The <code>LotFrontage</code> column actually does have
missing values: 259, to be precise. Additionally, notice how columns
such as <code>MSZoning</code>, <code>PavedDrive</code>, and
<code>HouseStyle</code> are categorical. These need to be encoded
numerically before you can use XGBoost. This is what you’ll do in the
coming exercises.
</p>

#### Encoding categorical columns I: LabelEncoder

<p>
Now that you’ve seen what will need to be done to get the housing data
ready for XGBoost, let’s go through the process step-by-step.
</p>
<p>
First, you will need to fill in missing values - as you saw previously,
the column <code>LotFrontage</code> has many missing values. Then, you
will need to encode any categorical columns in the dataset using one-hot
encoding so that they are encoded numerically. You can watch
<a href="https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines?ex=1">this
video</a> from
<a href="https://www.datacamp.com/courses/supervised-learning-with-scikit-learn">Supervised
Learning with scikit-learn</a> for a refresher on the idea.
</p>
<p>
The data has five categorical columns: <code>MSZoning</code>,
<code>PavedDrive</code>, <code>Neighborhood</code>,
<code>BldgType</code>, and <code>HouseStyle</code>. Scikit-learn has a
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html">LabelEncoder</a>
function that converts the values in each categorical column into
integers. You’ll practice using this here.
</p>

<li>
Import <code>LabelEncoder</code> from
<code>sklearn.preprocessing</code>.
</li>
<li>
Fill in missing values in the <code>LotFrontage</code> column with
<code>0</code> using <code>.fillna()</code>.
</li>
<li>
Create a boolean mask for categorical columns. You can do this by
checking for whether <code>df.dtypes</code> equals <code>object</code>.
</li>
<li>
Create a <code>LabelEncoder</code> object. You can do this in the same
way you instantiate any scikit-learn estimator.
</li>
<li>
Encode all of the categorical columns into integers using
<code>LabelEncoder()</code>. To do this, use the
<code>.fit_transform()</code> method of <code>le</code> in the provided
lambda function.
</li>

``` python
# edited/added
df = pd.read_csv("archive/Extreme-Gradient-Boosting-with-XGBoost/datasets/ames_unprocessed_data.csv")

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
```

    ##   MSZoning Neighborhood BldgType HouseStyle PavedDrive
    ## 0       RL      CollgCr     1Fam     2Story          Y
    ## 1       RL      Veenker     1Fam     1Story          Y
    ## 2       RL      CollgCr     1Fam     2Story          Y
    ## 3       RL      Crawfor     1Fam     2Story          Y
    ## 4       RL      NoRidge     1Fam     2Story          Y

``` python
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())
```

    ##    MSZoning  Neighborhood  BldgType  HouseStyle  PavedDrive
    ## 0         3             5         0           5           2
    ## 1         3            24         0           2           2
    ## 2         3             5         0           5           2
    ## 3         3             6         0           5           2
    ## 4         3            15         0           5           2

<p class>
Well done! Notice how the entries in each categorical column are now
encoded numerically. A <code>BldgTpe</code> of <code>1Fam</code> is
encoded as <code>0</code>, while a <code>HouseStyle</code> of
<code>2Story</code> is encoded as <code>5</code>.
</p>

#### Encoding categorical columns II: OneHotEncoder

<p>
Okay - so you have your categorical columns encoded numerically. Can you
now move onto using pipelines and XGBoost? Not yet! In the categorical
columns of this dataset, there is no natural ordering between the
entries. As an example: Using <code>LabelEncoder</code>, the
<code>CollgCr</code> <code>Neighborhood</code> was encoded as
<code>5</code>, while the <code>Veenker</code> <code>Neighborhood</code>
was encoded as <code>24</code>, and <code>Crawfor</code> as
<code>6</code>. Is <code>Veenker</code> “greater” than
<code>Crawfor</code> and <code>CollgCr</code>? No - and allowing the
model to assume this natural ordering may result in poor performance.
</p>
<p>
As a result, there is another step needed: You have to apply a one-hot
encoding to create binary, or “dummy” variables. You can do this using
scikit-learn’s
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">OneHotEncoder</a>.
</p>

<li>
Import <code>OneHotEncoder</code> from
<code>sklearn.preprocessing</code>.
</li>
<li>
Instantiate a <code>OneHotEncoder</code> object called <code>ohe</code>.
Specify the keyword arguments
<code>categorical_features=categorical_mask</code> and
<code>sparse=False</code>.
</li>
<li>
Using its <code>.fit_transform()</code> method, apply the
<code>OneHotEncoder</code> to <code>df</code> and save the result as
<code>df_encoded</code>. The output will be a NumPy array.
</li>
<li>
Print the first 5 rows of <code>df_encoded</code>, and then the shape of
<code>df</code> as well as <code>df_encoded</code> to compare the
difference.
</li>

``` python
# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categories="auto", sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
```

    ## [[0. 0. 0. ... 0. 0. 0.]
    ##  [1. 0. 0. ... 0. 0. 0.]
    ##  [0. 0. 0. ... 0. 0. 0.]
    ##  [0. 0. 0. ... 0. 0. 0.]
    ##  [0. 0. 0. ... 0. 0. 0.]]

``` python
print(df.shape)

# Print the shape of the transformed array
```

    ## (1460, 21)

``` python
print(df_encoded.shape)
```

    ## (1460, 3369)

<p class>
Superb! As you can see, after one hot encoding, which creates binary
variables out of the categorical variables, there are now 62 columns.
</p>

#### Encoding categorical columns III: DictVectorizer

<p>
Alright, one final trick before you dive into pipelines. The two step
process you just went through - <code>LabelEncoder</code> followed by
<code>OneHotEncoder</code> - can be simplified by using a
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html">DictVectorizer</a>.
</p>
<p>
Using a <code>DictVectorizer</code> on a DataFrame that has been
converted to a dictionary allows you to get label encoding as well as
one-hot encoding in one go.
</p>
<p>
Your task is to work through this strategy in this exercise!
</p>

<li>
Import <code>DictVectorizer</code> from
<code>sklearn.feature_extraction</code>.
</li>
<li>
Convert <code>df</code> into a dictionary called <code>df_dict</code>
using its <code>.to_dict()</code> method with <code>“records”</code> as
the argument.
</li>
<li>
Instantiate a <code>DictVectorizer</code> object called <code>dv</code>
with the keyword argument <code>sparse=False</code>.
</li>
<li>
Apply the <code>DictVectorizer</code> on <code>df_dict</code> by using
its <code>.fit_transform()</code> method.
</li>
<li>
Hit ‘Submit Answer’ to print the resulting first five rows and the
vocabulary.
</li>

``` python
# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
```

    ## [[3.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00 2.000e+00 5.480e+02
    ##   1.710e+03 1.000e+00 5.000e+00 8.450e+03 6.500e+01 6.000e+01 3.000e+00
    ##   5.000e+00 5.000e+00 7.000e+00 2.000e+00 0.000e+00 2.085e+05 2.003e+03]
    ##  [3.000e+00 0.000e+00 0.000e+00 1.000e+00 1.000e+00 2.000e+00 4.600e+02
    ##   1.262e+03 0.000e+00 2.000e+00 9.600e+03 8.000e+01 2.000e+01 3.000e+00
    ##   2.400e+01 8.000e+00 6.000e+00 2.000e+00 0.000e+00 1.815e+05 1.976e+03]
    ##  [3.000e+00 0.000e+00 1.000e+00 0.000e+00 1.000e+00 2.000e+00 6.080e+02
    ##   1.786e+03 1.000e+00 5.000e+00 1.125e+04 6.800e+01 6.000e+01 3.000e+00
    ##   5.000e+00 5.000e+00 7.000e+00 2.000e+00 1.000e+00 2.235e+05 2.001e+03]
    ##  [3.000e+00 0.000e+00 1.000e+00 0.000e+00 1.000e+00 1.000e+00 6.420e+02
    ##   1.717e+03 0.000e+00 5.000e+00 9.550e+03 6.000e+01 7.000e+01 3.000e+00
    ##   6.000e+00 5.000e+00 7.000e+00 2.000e+00 1.000e+00 1.400e+05 1.915e+03]
    ##  [4.000e+00 0.000e+00 1.000e+00 0.000e+00 1.000e+00 2.000e+00 8.360e+02
    ##   2.198e+03 1.000e+00 5.000e+00 1.426e+04 8.400e+01 6.000e+01 3.000e+00
    ##   1.500e+01 5.000e+00 8.000e+00 2.000e+00 0.000e+00 2.500e+05 2.000e+03]]

``` python
print(dv.vocabulary_)
```

    ## {'MSSubClass': 12, 'MSZoning': 13, 'LotFrontage': 11, 'LotArea': 10, 'Neighborhood': 14, 'BldgType': 1, 'HouseStyle': 9, 'OverallQual': 16, 'OverallCond': 15, 'YearBuilt': 20, 'Remodeled': 18, 'GrLivArea': 7, 'BsmtFullBath': 2, 'BsmtHalfBath': 3, 'FullBath': 5, 'HalfBath': 8, 'BedroomAbvGr': 0, 'Fireplaces': 4, 'GarageArea': 6, 'PavedDrive': 17, 'SalePrice': 19}

<p class>
Fantastic! Besides simplifying the process into one step,
<code>DictVectorizer</code> has useful attributes such as
<code>vocabulary\_</code> which maps the names of the features to their
indices. With the data preprocessed, it’s time to move onto pipelines!
</p>

#### Preprocessing within a pipeline

<p>
Now that you’ve seen what steps need to be taken individually to
properly process the Ames housing data, let’s use the much cleaner and
more succinct <code>DictVectorizer</code> approach and put it alongside
an <code>XGBoostRegressor</code> inside of a scikit-learn pipeline.
</p>

<li>
Import <code>DictVectorizer</code> from
<code>sklearn.feature_extraction</code> and <code>Pipeline</code> from
<code>sklearn.pipeline</code>.
</li>
<li>
Fill in any missing values in the <code>LotFrontage</code> column of
<code>X</code> with <code>0</code>.
</li>
<li>
Complete the steps of the pipeline with
<code>DictVectorizer(sparse=False)</code> for <code>“ohe_onestep”</code>
and <code>xgb.XGBRegressor()</code> for <code>“xgb_model”</code>.
</li>
<li>
Create the pipeline using <code>Pipeline()</code> and
<code>steps</code>.
</li>
<li>
Fit the <code>Pipeline</code>. Don’t forget to convert <code>X</code>
into a format that <code>DictVectorizer</code> understands by calling
the <code>to_dict(“records”)</code> method on <code>X</code>.
</li>

``` python
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]
         
# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)
```

    ## [15:33:00] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## Pipeline(steps=[('ohe_onestep', DictVectorizer(sparse=False)),
    ##                 ('xgb_model', XGBRegressor())])

<p class>
Well done! It’s now time to see what it takes to use XGBoost within
pipelines.
</p>

### Incorporating XGBoost into pipelines

#### Cross-validating your XGBoost model

<p>
In this exercise, you’ll go one step further by using the pipeline
you’ve created to preprocess <strong>and</strong> cross-validate your
model.
</p>

<li>
Create a pipeline called <code>xgb_pipeline</code> using
<code>steps</code>.
</li>
<li>
Perform 10-fold cross-validation using
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html"><code>cross_val_score()</code></a>.
You’ll have to pass in the pipeline, <code>X</code> (as a dictionary,
using <code>.to_dict(“records”)</code>), <code>y</code>, the number of
folds you want to use, and <code>scoring</code>
(<code>“neg_mean_squared_error”</code>).
</li>
<li>
Print the 10-fold RMSE.
</li>

``` python
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
```

    ## [15:33:03] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:03] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:04] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:04] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:04] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:05] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:05] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:05] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:06] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    ## [15:33:06] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.

``` python
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))
```

    ## 10-fold RMSE:  29903.48369050373

<p class>
Great work!
</p>

#### Kidney disease case study I: Categorical Imputer

<p>
You’ll now continue your exploration of using pipelines with a dataset
that requires significantly more wrangling. The
<a href="https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease">chronic
kidney disease dataset</a> contains both categorical and numeric
features, but contains lots of missing values. The goal here is to
predict who has chronic kidney disease given various blood indicators as
features.
</p>
<p>
As Sergey mentioned in the video, you’ll be introduced to a new library,
<a href="https://github.com/pandas-dev/sklearn-pandas"><code>sklearn_pandas</code></a>,
that allows you to chain many more processing steps inside of a pipeline
than are currently supported in scikit-learn. Specifically, you’ll be
able to impute missing categorical values directly using the
<code>Categorical_Imputer()</code> class in <code>sklearn_pandas</code>,
and the <code>DataFrameMapper()</code> class to apply any arbitrary
sklearn-compatible transformer on DataFrame columns, where the resulting
output can be either a NumPy array or DataFrame.
</p>
<p>
We’ve also created a transformer called a <code>Dictifier</code> that
encapsulates converting a DataFrame using
<code>.to_dict(“records”)</code> without you having to do it explicitly
(and so that it works in a pipeline). Finally, we’ve also provided the
list of feature names in <code>kidney_feature_names</code>, the target
name in <code>kidney_target_name</code>, the features in <code>X</code>,
and the target in <code>y</code>.
</p>
<p>
In this exercise, your task is to apply the
<code>CategoricalImputer</code> to impute all of the categorical columns
in the dataset. You can refer to how the numeric imputation mapper was
created as a template. Notice the keyword arguments
<code>input_df=True</code> and <code>df_out=True</code>? This is so that
you can work with DataFrames instead of arrays. By default, the
transformers are passed a <code>numpy</code> array of the selected
columns as input, and as a result, the output of the DataFrame mapper is
also an array. Scikit-learn transformers have historically been designed
to work with <code>numpy</code> arrays, not <code>pandas</code>
DataFrames, even though their basic indexing interfaces are similar.
</p>

<li>
Apply the categorical imputer using <code>DataFrameMapper()</code> and
<code>SimpleImputer()</code>. <code>SimpleImputer()</code> does not need
any arguments to be passed in. The columns are contained in
<code>categorical_columns</code>. Be sure to specify
<code>input_df=True</code> and <code>df_out=True</code>, and use
<code>category_feature</code> as your iterator variable in the list
comprehension.
</li>

``` python
# edited/added
import pandas as pd
X = pd.read_csv('archive/Extreme-Gradient-Boosting-with-XGBoost/datasets/chronic_kidney_X.csv')
y = pd.read_csv('archive/Extreme-Gradient-Boosting-with-XGBoost/datasets/chronic_kidney_y.csv').to_numpy().ravel()

# Import necessary modules
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.impute import SimpleImputer

# Check number of nulls in each feature columns
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
```

    ## age        9
    ## bp        12
    ## sg        47
    ## al        46
    ## su        49
    ## bgr       44
    ## bu        19
    ## sc        17
    ## sod       87
    ## pot       88
    ## hemo      52
    ## pcv       71
    ## wc       106
    ## rc       131
    ## rbc      152
    ## pc        65
    ## pcc        4
    ## ba         4
    ## htn        2
    ## dm         2
    ## cad        2
    ## appet      1
    ## pe         1
    ## ane        1
    ## dtype: int64

``` python
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy='median')) 
     for numeric_feature in non_categorical_columns],
    input_df=True,
    df_out=True
)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [(category_feature, CategoricalImputer()) 
     for category_feature in categorical_columns],
    input_df=True,
    df_out=True
)
```

<p class>
Great work!
</p>

#### Kidney disease case study II: Feature Union

<p>
Having separately imputed numeric as well as categorical columns, your
task is now to use scikit-learn’s
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html">FeatureUnion</a>
to concatenate their results, which are contained in two separate
transformer objects - <code>numeric_imputation_mapper</code>, and
<code>categorical_imputation_mapper</code>, respectively.
</p>
<p>
You may have already encountered <code>FeatureUnion</code> in
<a href="https://campus.datacamp.com/courses/machine-learning-with-the-experts-school-budgets/improving-your-model?ex=7">Machine
Learning with the Experts: School Budgets</a>. Just like with pipelines,
you have to pass it a list of <code>(string, transformer)</code> tuples,
where the first half of each tuple is the name of the transformer.
</p>

<li>
Import <code>FeatureUnion</code> from <code>sklearn.pipeline</code>.
</li>
<li>
Combine the results of <code>numeric_imputation_mapper</code> and
<code>categorical_imputation_mapper</code> using
<code>FeatureUnion()</code>, with the names <code>“num_mapper”</code>
and <code>“cat_mapper”</code> respectively.
</li>

``` python
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])
```

<p class>
Great work!
</p>

#### Kidney disease case study III: Full pipeline

<p>
It’s time to piece together all of the transforms along with an
<code>XGBClassifier</code> to build the full pipeline!
</p>
<p>
Besides the <code>numeric_categorical_union</code> that you created in
the previous exercise, there are two other transforms needed: the
<code>Dictifier()</code> transform which we created for you, and the
<code>DictVectorizer()</code>.
</p>
<p>
After creating the pipeline, your task is to cross-validate it to see
how well it performs.
</p>

<li>
Create the pipeline using the <code>numeric_categorical_union</code>,
<code>Dictifier()</code>, and <code>DictVectorizer(sort=False)</code>
transforms, and <code>xgb.XGBClassifier()</code> estimator with
<code>max_depth=3</code>. Name the transforms
<code>“featureunion”</code>, <code>“dictifier”</code>
<code>“vectorizer”</code>, and the estimator <code>“clf”</code>.
</li>
<li>
Perform 3-fold cross-validation on the <code>pipeline</code> using
<code>cross_val_score()</code>. Pass it the pipeline,
<code>pipeline</code>, the features, <code>kidney_data</code>, the
outcomes, <code>y</code>. Also set <code>scoring</code> to
<code>“roc_auc”</code> and <code>cv</code> to <code>3</code>.
</li>

``` python
# edited/added
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

# Define Dictifier class to turn df into dictionary as part of pipeline
class Dictifier(BaseEstimator, TransformerMixin):       
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            return X.to_dict("records")
        else:
            return pd.DataFrame(X).to_dict("records")
          
# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])
                    
# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))
```

    ## 3-fold AUC:  0.998637406769937

<p class>
Great work!
</p>

### Tuning XGBoost hyperparameters

#### Bringing it all together

<p>
Alright, it’s time to bring together everything you’ve learned so far!
In this final exercise of the course, you will combine your work from
the previous exercises into one end-to-end XGBoost pipeline to really
cement your understanding of preprocessing and pipelines in XGBoost.
</p>
<p>
Your work from the previous 3 exercises, where you preprocessed the data
and set up your pipeline, has been pre-loaded. Your job is to perform a
randomized search and identify the best hyperparameters.
</p>

<li>
Set up the parameter grid to tune <code>’clf\_\_learning_rate’</code>
(from <code>0.05</code> to <code>1</code> in increments of
<code>0.05</code>), <code>’clf\_\_max_depth’</code> (from <code>3</code>
to <code>10</code> in increments of <code>1</code>), and
<code>’clf\_\_n_estimators’</code> (from <code>50</code> to
<code>200</code> in increments of <code>50</code>).
</li>
<li>
Using your <code>pipeline</code> as the estimator, perform 2-fold
<code>RandomizedSearchCV</code> with an <code>n_iter</code> of
<code>2</code>. Use <code>“roc_auc”</code> as the metric, and set
<code>verbose</code> to <code>1</code> so the output is more detailed.
Store the result in <code>randomized_roc_auc</code>.
</li>
<li>
Fit <code>randomized_roc_auc</code> to <code>X</code> and
<code>y</code>.
</li>
<li>
Compute the best score and best estimator of
<code>randomized_roc_auc</code>.
</li>

``` python
# edited/added
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(.05, 1, .05),
    'clf__max_depth': np.arange(3,10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline,
                                        param_distributions=gbm_param_grid,
                                        n_iter=2, scoring='roc_auc', cv=2, verbose=1)
                                        
# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
```

    ## Fitting 2 folds for each of 2 candidates, totalling 4 fits
    ## RandomizedSearchCV(cv=2,
    ##                    estimator=Pipeline(steps=[('featureunion',
    ##                                               FeatureUnion(transformer_list=[('num_mapper',
    ##                                                                               DataFrameMapper(df_out=True,
    ##                                                                                               features=[(['age'],
    ##                                                                                                          SimpleImputer(strategy='median')),
    ##                                                                                                         (['bp'],
    ##                                                                                                          SimpleImputer(strategy='median')),
    ##                                                                                                         (['sg'],
    ##                                                                                                          SimpleImputer(strategy='median')),
    ##                                                                                                         (['al'],
    ##                                                                                                          SimpleImputer(strategy='median')),
    ##                                                                                                         (['su'],
    ##                                                                                                          SimpleImputer(strategy='...
    ##                                                                                               input_df=True))])),
    ##                                              ('dictifier', Dictifier()),
    ##                                              ('vectorizer',
    ##                                               DictVectorizer(sort=False)),
    ##                                              ('clf', XGBClassifier())]),
    ##                    n_iter=2,
    ##                    param_distributions={'clf__learning_rate': array([0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
    ##        0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]),
    ##                                         'clf__max_depth': array([3, 4, 5, 6, 7, 8, 9]),
    ##                                         'clf__n_estimators': array([ 50, 100, 150])},
    ##                    scoring='roc_auc', verbose=1)

``` python
print(randomized_roc_auc.best_score_)
```

    ## 0.9969066666666666

``` python
print(randomized_roc_auc.best_estimator_)
```

    ## Pipeline(steps=[('featureunion',
    ##                  FeatureUnion(transformer_list=[('num_mapper',
    ##                                                  DataFrameMapper(df_out=True,
    ##                                                                  features=[(['age'],
    ##                                                                             SimpleImputer(strategy='median')),
    ##                                                                            (['bp'],
    ##                                                                             SimpleImputer(strategy='median')),
    ##                                                                            (['sg'],
    ##                                                                             SimpleImputer(strategy='median')),
    ##                                                                            (['al'],
    ##                                                                             SimpleImputer(strategy='median')),
    ##                                                                            (['su'],
    ##                                                                             SimpleImputer(strategy='median')),
    ##                                                                            (['bgr'],
    ##                                                                             SimpleImputer(s...
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('htn',
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('dm',
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('cad',
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('appet',
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('pe',
    ##                                                                             CategoricalImputer()),
    ##                                                                            ('ane',
    ##                                                                             CategoricalImputer())],
    ##                                                                  input_df=True))])),
    ##                 ('dictifier', Dictifier()),
    ##                 ('vectorizer', DictVectorizer(sort=False)),
    ##                 ('clf',
    ##                  XGBClassifier(learning_rate=0.4, max_depth=7,
    ##                                n_estimators=150))])

<p class>
Amazing work! This type of pipelining is very common in real-world data
science and you’re well on your way towards mastering it.
</p>

### Final Thoughts

#### Final Thoughts

Congratulations on completing this course. Let’s go over everything
we’ve covered in this course, as well as where you can go from here with
learning other topics related to XGBoost that we didn’t have a chance to
cover.

#### What We Have Covered And You Have Learned

So, what have we been able to cover in this course? Well, we’ve learned
how to use XGBoost for both classification and regression tasks. We’ve
also covered all the most important hyperparameters that you should tune
when creating XGBoost models, so that they are as performant as
possible. And we just finished up how to incorporate XGBoost into
pipelines, and used some more advanced functions that allow us to
seamlessly work with Pandas DataFrames and scikit-learn. That’s quite a
lot of ground we’ve covered and you should be proud of what you’ve been
able to accomplish.

#### What We Have Not Covered (And How You Can Proceed)

However, although we’ve covered quite a lot, we didn’t cover some other
topics that would advance your mastery of XGBoost. Specifically, we
never looked into how to use XGBoost for ranking or recommendation
problems, which can be done by modifying the loss function you use when
constructing your model. We also didn’t look into more advanced
hyperparameter selection strategies. The most powerful strategy, called
Bayesian optimization, has been used with lots of success, and entire
companies have been created just for specifically using this method in
tuning models (for example, the company sigopt does exactly this). It’s
a powerful method, but would take an entire other DataCamp course to
teach properly! Finally, we haven’t talked about ensembling XGBoost with
other models. Although XGBoost is itself an ensemble method, nothing
stops you from combining the predictions you get from an XGBoost model
with other models, as this is usually a very powerful additional way to
squeeze the last bit of juice from your data. Learning about all of
these additional topics will help you become an even more powerful user
of XGBoost. Now that you know your way around the package, there’s no
reason for you to stop learning how to get even more benefits out of it.

#### Congratulations!

I hope you’ve enjoyed taking this course on XGBoost as I have teaching
it. Please let us know if you’ve enjoyed the course and definitely let
me know how I can improve it. It’s been a pleasure, and I hope you
continue your data science journey from here!