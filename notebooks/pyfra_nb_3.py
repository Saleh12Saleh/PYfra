# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -LanguageId
#     formats: ipynb,py:light
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# Notebook 3
# ==============
# Modelling

# # Importing Packages and Data

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, make_scorer
from imblearn import under_sampling
from imblearn.under_sampling import RandomUnderSampler
from time import sleep
import pyfra

df = pd.read_pickle('../data/df.p')
n_rows_complete = len(df)

# Check whether or not the data is up-to-date (file can't be tracked on github because of it's file size)
pyfra.df_compare_to_description(df=df, description_filepath='../data/df_check_info.csv')

rus = RandomUnderSampler(random_state=23)

# Create a sample of the data, because the whole dataset is too big for us to work with
relative_sample_size = 0.1
df = df.sample(frac=relative_sample_size, random_state=23)

data = df.drop(columns='Severity',axis=1).select_dtypes(include=np.number).dropna(axis=1)
target = df['Severity']
data, target = rus.fit_resample(X=data, y=target)

target.value_counts()

print(f'We are working on {len(target)} data points, which represent {len(target)/n_rows_complete*100:.04f}% of the original data,')

X_train, X_test, y_train, y_test  = train_test_split(data, target, test_size=0.2 ,random_state=23)

# # Scaling the Data and Selecting Features

from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0.01).fit(X_train)
X_train = constant_filter.fit_transform(X_train)
X_test = constant_filter.transform(X_test)

std_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

k_features = 60
kbest_selector = SelectKBest(k=k_features)
kbest_selector.fit(X_train_scaled,y_train);
X_train_scaled_selection = kbest_selector.transform(X_train_scaled)
X_test_scaled_selection = kbest_selector.transform(X_test_scaled)
print(f'We use {k_features} of the original {df.shape[1]} features')

k_best_feature_names = data.columns[kbest_selector.get_support(indices=True)]

# ## Setup of the Cross-Validator
# We will use a repeated stratified cross-validation to make sure to pick the best parameters.
# The stratification will be used to ensure an equal distribution of the different categories in every bin.
# The repetition will be used in order ensure that the result is not an outlier. We will set a lower the number of repetitions, however, to save execution time (default would be 10 repetitions).

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=23)

# ## Support Vector Machine (SVM)
#
# A support vector machine classifier will be used with parameter optimization via grid search.
#
# ### Setup of the SVM and the Grid Search

# +
# Instantiation of the SVM Classifier
# We set the cache size to 1600 MB (default: 200 MB) to reduce the computing time.
# The other parameters will be set via grid search.
svc = svm.SVC(cache_size=4000)

# Choosing the parameters for the grid search
svc_params = {
    'kernel': ['rbf'],
    'gamma': ['scale'],
    'C': [0.5]
}

# Setup of the scoring. 
# We have to define the parameter 'average', because we are not dealing with a binary classification.
# Our sample is balanced, hence we can use a simple approach, using 'micro', which uses the global values of 
# true positives, false negatives and false positives.
f1_scoring = make_scorer(score_func=f1_score, average='micro')

# Instantiation of the GridSearchCv
# verbose is set to 1000 to get as much output as possible, because computation
# can take a long time
# n_jobs is set to -1 to use all available threads for computation.
svc_grid = GridSearchCV(svc, 
                        param_grid=svc_params, 
                        scoring=f1_scoring, 
                        cv=cv,
                        verbose=1000, 
                        n_jobs=-1)
# -

# ### SVM Parameter Optimization, Training and Prediction

# +
# Fitting the grid search to find the best parameter combination
svc_grid.fit(X_train_scaled_selection, y_train)

# Print result of parameter optimization
print('Best parameter combination: ',svc_grid.best_params_)

# Predict target variable for the test set
svc = svc_grid.best_estimator_
y_svc = svc.predict(X_test_scaled_selection)

# -

# ### Metrics of SVM

# Calculate the metrics for the optimal svm model and store them in the result_metrics DataFrame 
# The model will be stored as well in the DataFrame
result_metrics = pyfra.store_metrics(model=svc, model_name='Support Vector Machine',
                               y_test=y_test, y_pred=y_svc)
# Show the interim result                               
result_metrics

pyfra.print_confusion_matrix(y_test, y_svc, 
                            model_name='Support Vector Classifier',
                            filename='svc_conf')

# ## Random Forest
# ### Setup and GridSearch

# +
params = {
    'criterion': ['gini'],
    'max_depth': [5,10],
    'min_samples_leaf':[3,7],
    'n_estimators': [50,100]
    }

RFCLF = GridSearchCV(RandomForestClassifier(),param_grid = params, cv = cv)
RFCLF.fit(X_train_scaled_selection,y_train)

print('Best Params are:',RFCLF.best_params_)
print('Best Score is:',RFCLF.best_score_)
# -

# ### Optimized Model and Metrics

# +
rf = RFCLF.best_estimator_
y_rf = rf.predict(X_test_scaled_selection)

cm = pd.crosstab(y_test,y_rf, rownames=['Real'], colnames=['Prediction'])
print(cm)

result_metrics = pyfra.store_metrics(model=rf, model_name='Random Forest',
                               y_test=y_test, y_pred=y_rf,
                               result_df=result_metrics)
                              
result_metrics
# -

pyfra.print_confusion_matrix(y_test, y_rf, 
                            model_name='Random Fores Classifier',
                            filename='rf_conf')

# # Logistic Regression

# +
#We use and define logistic Regression with n_jobs=-1 to use all cores
LR = LogisticRegression(max_iter=1000)
#for parameters we use 3 type of solver and 6 for C
LR_params = {
    'solver': ['liblinear', 'lbfgs', 'saga'], 
    'C': [10**(i) for i in range(-5, 5)]
}

f1_scoring = make_scorer(score_func=f1_score, average='micro')

# Instantiation of the GridSearchCv
LR_grid = GridSearchCV(LR, param_grid=LR_params, scoring=f1_scoring, cv=cv, n_jobs=-1)

# -

# # LR Parameter Optimization, Training and Prediction

# +
# Fitting the grid search to find the best parameter combination
LR_grid.fit(X_train_scaled_selection, y_train)

# Print result of parameter optimization
print('Best parameter combination: ',LR_grid.best_params_)

# Predict target variable for the test set
LR = LR_grid.best_estimator_
y_LR = LR.predict(X_test_scaled_selection)
# -

# ## Metrics of LR

# Calculate the metrics for the optimal LR model and store them in the result_metrics DataFrame 
# The model will be stored as well in the DataFrame
result_metrics = pyfra.store_metrics(model=LR, model_name='Logistic Regression',
                               y_test=y_test, y_pred=y_LR,
                               result_df=result_metrics)
# Show the interim result                               
result_metrics

pyfra.print_confusion_matrix(y_test, y_LR, 
                            model_name='Logistic Regression Classifier',
                            filename='log_reg_conf')

# # Decision Tree

# ## Setup of the DT and the Grid Search

# +
from sklearn import tree
from sklearn.pipeline import Pipeline

# Grid
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]
parameters = dict(criterion=criterion, max_depth=max_depth)

DT = GridSearchCV(DecisionTreeClassifier(),param_grid = parameters, cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=23))
# 
DT.fit(X_train_scaled_selection,y_train)

# 
print('Best Criterion:', DT.best_estimator_.get_params())
print('Best max_depth:', DT.best_estimator_.get_params())
print(); print(DT.best_estimator_.get_params())
# -

# ## Metrics of Decision Tree

# +
dt = DT.best_estimator_
y_dt = dt.predict(X_test_scaled_selection)
cm = pd.crosstab(y_test,y_dt, rownames=['Real'], colnames=['Prediction'])
print(cm)
result_metrics = pyfra.store_metrics(model=dt, model_name='Decision Tree',
                               y_test=y_test, y_pred=y_dt,
                               result_df=result_metrics)
                              
result_metrics

# +
# ## Interpretation of the Decision Tree
# Decision trees are known to have a high interpretability compared to other machine learning models. The performance of the applied model is worse than the ones of the other models, but we can easily plot the tree and gain insights.
from sklearn.tree import plot_tree
fig = plt.figure(figsize=(12,6));
plot_tree(dt,max_depth=2, fontsize=8, feature_names=k_best_feature_names);

# The plot shows that the most important feature (according to the decision tree) is built-up_area. This binary variable cointains the information, whether the accident happened in a built-up area. We already showed in the first notebook that there seems to be a positive relation between the density of an area and it's **number** of accident. The decision tree here suggests that the **severity** is also affected by a dense population.
# -

pyfra.print_confusion_matrix(y_test, y_dt, 
                            model_name='Decision Tree',
                            filename='dt_conf')

# # Application of Advanced Models


# ## Stacking Classifier

# +
estimators = [('lr', LR), ('svc', svc), ('rf', rf)]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=svc, cv='prefit', n_jobs=-1)

stacking_clf.fit(X_train_scaled_selection, y_train)
y_stacking = stacking_clf.predict(X_test_scaled_selection)
result_metrics = pyfra.store_metrics(model=stacking_clf, model_name='Stacking',
                               y_test=y_test, y_pred=y_stacking,
                               result_df=result_metrics)
result_metrics
# -
# The confusion matrix of the stacking classifier will be analyzed in detail below.

# # ADA Boosting

#Trying ADA boosting on LogisticRegresiion
ADA_Boost = AdaBoostClassifier(estimator = LR , n_estimators = 1000)
ADA_Boost.fit(X_train_scaled_selection, y_train)
y_ada = ADA_Boost.predict(X_test_scaled_selection)

result_metrics = pyfra.store_metrics(model=ADA_Boost, model_name='ADA Boost',
                               y_test=y_test, y_pred=y_ada,
                               result_df=result_metrics)
# Show the interim result                               
result_metrics

pyfra.print_confusion_matrix(y_test, y_ada, 
                            model_name='AdaBoost',
                            filename='ada_conf')

#
# # Results and Conclusion

# ## Comparison of the Performances

plt.barh(y=result_metrics.index.values, width=result_metrics['f1']);
plt.title('$F_1$ Score of different ML models');

# The results show a comparable performance for all machine learning models, with the advanced stacking classifier giving the best score (f_1) and the decision tree giving the worst score. 
#
# There are a few things to consider when analyzing these results.
# 1. We worked on a very small partition of the dataset in order to achieve acceptable execution times. We expect to reach higher performances when working with more data.
# 2. The classifiers based on logistic regression and decision trees have low scores, are not necessarily unfit for the dataset, as they offer more interpretability than the advanced models. This interpretability could help e.g. policy makers to take measures in order to reduce the severity of road accidents.
# 3. A strong correlation between severity and safety measurements (e.g. safety belt) is expected. Unfortunately, this feature could not be used because useful is only available for the last years (2018--).

# ## Analysis of the Confusion Matrix

pyfra.print_confusion_matrix(y_test, y_stacking, 
                            model_name='Stacking Classifier',
                            filename='stacking_conf')

# The correlation matrix of the stacking classifier shows that some categories are more difficult to predict than others. The category "Hospitalized wounded" seems to be the most difficult to predict, as the predictions seem to be quite evenly distributed between the different classes. We can quantify these difficulties by looking at the scores for accuracy and recall for each category.

from sklearn.metrics import classification_report
severity_categories = ("Unscathed","Killed", "Hospitalized wounded", "Light injury")
print(classification_report(y_true=y_test, y_pred=y_stacking, target_names=severity_categories))

# The classification report reflects our observations from the correlation matrix. It is satisfying that the categorie "Killed" is predicted with the highest accuracy; we consider this category as particularly important.

# Export the results
result_metrics.to_pickle('../data/nb_3_results.p')

# +
# Saving the models for further use and investigation
from joblib import dump, load

dump(LR, '../models/log_reg_clf.joblib')
dump(svc, '../models/svc.joblib')
dump(stacking_clf, '../models/stacking_clf.joblib')
