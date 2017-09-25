#!/usr/bin/python
###############################################################################
###############################################################################
##### Identify Fraud from Enron Email (PYTHON 2.7) ############################
###############################################################################
###############################################################################

###############################################################################
## LOAD LIBRARIES 
###############################################################################

import sys
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from mlxtend.classifier import StackingClassifier

###############################################################################
## GET DATASET 
###############################################################################

def get_data_as_df(filename):
    """Load dic and transform to df
    """
    # get dic
    with open(filename, "r") as data_file:
        data_dict = pickle.load(data_file)
    # transform to df    
    df = pd.DataFrame.from_records(list(data_dict.values()))
    employees = pd.Series(list(data_dict.keys()))
    df = df.set_index(employees)
    # adjust variable types 
    df = df.apply(lambda x: pd.to_numeric(x, errors = "coerce"))
    # drop email address column
    df.drop("email_address", 1, inplace = True)
    # replace nas with zeros
    df.fillna(0,inplace = True)
    # df.replace('NaN', np.nan, inplace = True)
    # transform to factor
    df["poi"] = df["poi"].astype("category")
    # poi to first location
    poi = df["poi"]
    df.drop(labels=["poi"], axis=1,inplace = True)
    df.insert(0, "poi", poi)
    return df 

###############################################################################
## EXPLORE DATA 
###############################################################################

def is_extreme(value, p25, p75):
    """Check if value is an outlier
    """
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return value <= lower or value >= upper

def get_outliers(values):
    """Get outlier values and indices 
    """
    # drop NAs   
    values = values.dropna()
    names =  values.index
    # calculate percentiles
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    # initiate empty lists   
    indices_of_outliers = []
    values_of_outliers  = []
    # for each value 
    for value, name in zip(values, names):
        # apply is_extreme
        if is_extreme(value, p25, p75):
            # get index and value
            indices_of_outliers.append(name)
            values_of_outliers.append(value)
            outliers = zip(indices_of_outliers, values_of_outliers)
    return outliers

def remove_colums(df):
    """Remove some useless columns 
    """
    df.drop("TOTAL", inplace = True)
    df.drop("LOCKHART EUGENE E", inplace = True)
    df.drop("THE TRAVEL AGENCY IN THE PARK", inplace = True)
    return df

def quick_and_dirty_summary(df):
    """Get some basic statistics
    """
    # get concise summary of df
    print(df.info())
    # get descriptive statistics
    print(df.describe(exclude = [np.number]))
    print(df.describe(include = [np.number]))
    # get percentage of missing values
    print(df.isnull().sum()/len(df)*100)
    # get correlations
    # print(df.corr()["numerical dependent variable"])

def quick_and_dirty_univariate(df):
    """Get a histogram for each variable (zeros are excluded) 
    """
    df = pd.melt(df)
    df = df[(df!=0)]
    # df = df.dropna()
    g = sns.FacetGrid(df, col = "variable", sharex = False, sharey = False)
    g.set(yticklabels=[])
    g.set(xticklabels=[])
    g.map(sns.distplot, "value");
    
def quick_and_dirty_bivariate(df):
    """Get violin plots for each feature (zeros are excluded) and
    values are min-max normalized
    """
    #df.drop("poi", 1).replace(0, np.nan)
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df["poi"] = df["poi"].astype("category")
    g = sns.PairGrid(df,
                 x_vars=["bonus", 
                         "deferral_payments", 
                         "deferred_income",
                         "director_fees",
                         "exercised_stock_options",
                         "expenses",
                         "from_messages",
                         "from_poi_to_this_person",
                         "from_this_person_to_poi",
                         "loan_advances",
                         "long_term_incentive",
                         "restricted_stock",
                         "restricted_stock_deferred",
                         "salary",
                         "shared_receipt_with_poi",
                         "to_messages",
                         "total_payments",
                         "total_stock_value"],
                 y_vars=["poi"],
                 aspect=.75, size=3.5)
    g.set(xticklabels=[])
    g.map(sns.violinplot, palette="pastel");

def plot_heatmap(df):
    """Get heatmap to check correlation between features
    """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
###############################################################################
## ENGINEER FEATURES AND PREPARE DATA TO BE APPROPIATE FOR SKLEARN 
###############################################################################

def build_features(df):
    """Create some new features  
    """
    df["poi_to_person_mess"]  = df["from_poi_to_this_person"]/df["to_messages"]
    df["person_to_poi_mess"]  = df["from_this_person_to_poi"]/df["to_messages"]
    df["expenses_bonus"]      = df["expenses"]/df["bonus"]
    df["expenses_salery"]     = df["expenses"]/df["salary"]
    df["total_wealth"]        = df["bonus"] + df["salary"]
    df["stock_rel"]           = df["exercised_stock_options"] / df["total_stock_value"]

    
    df.fillna(0,inplace = True)
    df.replace(np.inf, 0, inplace=True)
    df.drop(df.columns[[4,10,13]], axis=1, inplace = True)    
  
def to_dic(df):
    """transfer to dic  
    """
    df = df.to_dict("index")
    return df

def to_lists_label_feature_scheme(dic, df):
    """get features and labels as list  
    """
    features_list = list(df.columns.values)
    data = featureFormat(dic, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return (labels, features)

###############################################################################
## SPOT-CHECKING STANDARD ALGORITHMS -> NB, ADA and GB LOOK PROMISING
###############################################################################

def spot_check_models_default(features, labels, score, norm):
    """ check a variety of algorithms with/without transformation of
    features
    """
    if norm == "none":
        features = features 
    if norm == "stand":
        features = StandardScaler().fit_transform(features)
    if norm == "scale":
        features = MinMaxScaler().fit_transform(features)
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('GB', GradientBoostingClassifier()))
    results = []
    names = []
    scoring = score
    for name, model in models:
        kfold = StratifiedShuffleSplit(labels, 100, random_state = 42)
        cv_results = model_selection.cross_val_score(model,
                                                     features, 
                                                     labels, 
                                                     cv=kfold, 
                                                     scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
     
###############################################################################
## GRID SEARCH FOR ADA 
###############################################################################
    """ perform grid search for ADA 
    """
def tune_ADA(features, labels):
    kfold = StratifiedShuffleSplit(labels, 100, random_state = 42)
    clf = AdaBoostClassifier()
    param_grid = {'learning_rate': [0.1, 0.05, 0.02],
                  'n_estimators': [50, 100, 150]
                  }

    grid_search = GridSearchCV(clf, 
                               param_grid=param_grid, 
                               scoring = "f1",
                               cv=kfold)

    grid_search.fit(features, labels)
    print(grid_search.best_params_, grid_search.best_score_)

###############################################################################
## GRID SEARCH FOR GRADIENT BOOSTING 
###############################################################################

def tune_GB(features, labels):
    """ perform grid search for GB
    """
    # features = StandardScaler().fit_transform(features)
    kfold = StratifiedShuffleSplit(labels, 100, random_state = 42)
    clf = GradientBoostingClassifier()
    param_grid = {'learning_rate': [0.1, 0.05, 0.02],
                  'n_estimators': [50, 100, 150, 200],
                  'max_depth': [2, 3, 4],
                  'min_samples_split': [2, 4, 6],
                  'min_samples_leaf': [1, 3, 5],
                  'min_weight_fraction_leaf': [0, 0.1, 0.2]
                  }

    grid_search = GridSearchCV(clf, 
                               param_grid=param_grid, 
                               scoring = "f1",
                               cv=kfold)

    grid_search.fit(features, labels)
    print(grid_search.best_params_, grid_search.best_score_)

###############################################################################
## STACK TOP GB AND NB
###############################################################################

def stack_models(features, labels):
    """ stack gb and nb
    """
    kfold = StratifiedShuffleSplit(labels, 100, random_state = 42)
    clf1  = GradientBoostingClassifier(learning_rate = 0.1, 
                                       min_samples_leaf = 1, 
                                       n_estimators = 200, 
                                       min_weight_fraction_leaf = 0, 
                                       min_samples_split = 4, 
                                       max_depth = 2)
    #clf2  = AdaBoostClassifier()
    clf3  = GaussianNB()
    lr    = LogisticRegression()
    sclf  = StackingClassifier(classifiers=[clf1, clf3], meta_classifier=lr)
    for clf, label in zip([clf1, clf3, sclf],
                          ['GB', 
                           'Naive Bayes',
                           'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, features, labels,
                                                 cv=kfold, scoring='f1')    
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
              % (scores.mean(), scores.std(), label))

###############################################################################
## USING PIPES TO TEST IF REDUCING DIMENSIONALITY MAKES SENSE -
############################################################################### 

def red_dim(features, labels, red):
    """ run GB with best parameter settings pure, with pca, or SelectKBest 
    beforehand
    """
    estimators = []
    if red == "none":
        estimators.append(('GB', 
                           GradientBoostingClassifier(learning_rate = 0.1, 
                                                      min_samples_leaf = 1, 
                                                      n_estimators = 200, 
                                                      min_weight_fraction_leaf = 0, 
                                                      min_samples_split = 4, 
                                                      max_depth = 2)))
    if red == "pca":
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('pca', decomposition.PCA()))
        estimators.append(('GB', 
                           GradientBoostingClassifier(learning_rate = 0.1, 
                                                      min_samples_leaf = 1, 
                                                      n_estimators = 200, 
                                                      min_weight_fraction_leaf = 0, 
                                                      min_samples_split = 4, 
                                                      max_depth = 2)))
    if red == "kbest":
        estimators.append(('select_best', SelectKBest(k=10)))
        estimators.append(('GB', 
                           GradientBoostingClassifier(learning_rate = 0.1, 
                                                      min_samples_leaf = 1, 
                                                      n_estimators = 200, 
                                                      min_weight_fraction_leaf = 0, 
                                                      min_samples_split = 4, 
                                                      max_depth = 2)))
    model = Pipeline(estimators)
    # evaluate pipeline
    kfold = StratifiedShuffleSplit(labels, 100, random_state = 42)
    results = cross_val_score(model, features, labels, cv=kfold, scoring='f1')
    print(results.mean())

###############################################################################
## FEATURE IMPORTANCE WITH FINAL MODEL 
###############################################################################

def plot_feature_importance(features, labels, column_names, top_n=25):
    """ plot feature importance for GB model on a 80% stratisfied train dataset 
    """
    Xtr,Xte,ytr,yte=cross_validation.train_test_split(features, 
                                                      labels, 
                                                      test_size=0.2, 
                                                      random_state=42, 
                                                      stratify=labels)

    clf = GradientBoostingClassifier(learning_rate = 0.1,
                                     min_samples_leaf = 1, 
                                     n_estimators = 200, 
                                     min_weight_fraction_leaf = 0, 
                                     min_samples_split = 4,
                                     max_depth = 2)
    clf.fit(features, labels)
    feature_imp = clf.feature_importances_ 
    imp_dict = dict(zip(column_names, 
                        feature_imp))
    top_features = sorted(imp_dict, 
                          key=imp_dict.get, 
                          reverse=True)[0:top_n]
    top_importances = [imp_dict[feature] for feature 
                          in top_features]
    df = pd.DataFrame(data={'feature': top_features, 
                            'importance': top_importances})
    p = sns.barplot(df.importance, df.feature)
    return p



###############################################################################
## EXECUTE FUNCTIONS
###############################################################################
    
def main():
    # run all my own function
    enron_data = get_data_as_df("final_project_dataset.pkl")
    get_outliers(enron_data["bonus"])
    remove_colums(enron_data)
    quick_and_dirty_summary(enron_data)
    quick_and_dirty_univariate(enron_data)
    quick_and_dirty_bivariate(enron_data)
    plot_heatmap(enron_data)
    build_features(enron_data)
    my_dataset = to_dic(enron_data)
    labels, features = to_lists_label_feature_scheme(my_dataset, enron_data)
    spot_check_models_default(features, labels, "f1", norm="none")
    spot_check_models_default(features, labels, "f1", norm="stand")
    spot_check_models_default(features, labels, "f1", norm="scale")
    tune_ADA(features, labels)
    tune_GB(features, labels)
    stack_models(features, labels)
    red_dim(features, labels, red = "none")
    red_dim(features, labels, red = "pca")
    red_dim(features, labels, red = "kbest")
    plot_feature_importance(features, 
                        labels, 
                        list(enron_data.iloc[:, 1:22].columns.values),
                        25)
    
    # dump classifier and data so that tester.py can be used for final evaluation
    my_dataset = my_dataset
    columns = ["to_messages", "deferral_payments", "long_term_incentive", 
               "from_poi_to_this_person"]    
    enron_data.drop(columns, inplace=True, axis=1)
    features_list = list(enron_data.columns.values)
    clf = Pipeline(steps=[#("SKB", SelectKBest(k = 8)), 
                          ("GB", GradientBoostingClassifier(learning_rate = 0.1, 
                                                            min_samples_leaf = 1, 
                                                            n_estimators = 200, 
                                                            min_weight_fraction_leaf = 0, 
                                                            min_samples_split = 4, 
                                                            max_depth = 2))])
    dump_classifier_and_data(clf, my_dataset, features_list)
    
if __name__ == '__main__':
    main()
 
