import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix,f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, balanced_accuracy_score, fbeta_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, chi2 , SelectPercentile , f_classif
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE



def fit_score(model,X_test,y_test):
    y_pred = model.predict(X_test)
    probs=model.predict_proba(X_test)
    probs_ov=probs[:,1]

    if(model.best_params_):
      print("Best Parameters", model.best_params_)
    print('AUC-score :',  [roc_auc_score(y_test, probs_ov)],)
    precision, recall,_ = precision_recall_curve(y_test, probs_ov)
    print("Precision-Recall-auc: ", [auc(recall, precision)])
    print('Balanced accuracy score', [balanced_accuracy_score(y_test,y_pred)])   
    print('Recall :' ,  [recall_score(y_test, y_pred)])
    print( 'Precision :' , [precision_score(y_test, y_pred)],)
    print('F1_score' , [fbeta_score(y_test,y_pred, beta=1.0)])  
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("\n\nroc_curve:")
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, probs_ov)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='x = y')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='roc_curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    print("\n\nprauc_curve:")
    x_y = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [x_y, x_y], linestyle='--')
    plt.plot(recall, precision, marker='.', label='pr_auc_curve')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def variance(data, threshold):
  var = data.var()
  columns = data.columns
  variable = []
  for i in range(0,len(var)):
      if var[i]>=threshold:   
        variable.append(columns[i])
  return variable

def correlation(data, threshold):
  correlated_features = set()
  for i in range(len(data.corr().columns)):
    for j in range(i):
        if abs(data.corr().iloc[i, j]) > threshold:
            colname = data.corr().columns[i]
            correlated_features.add(colname)
  return correlated_features

def randomForestSelector(X, y, size):
  model = RandomForestClassifier(random_state=1, max_depth=10)
  model.fit(X, y)
  features = X.columns
  importances = model.feature_importances_

  indices = np.argsort(importances)[-size:]
  plt.figure(figsize=(20, 10))
  plt.title('Feature Importances')
  plt.barh(range(len(indices)), importances[indices], color='b', align='center')
  plt.yticks(range(len(indices)), [features[i] for i in indices])
  plt.xlabel('Relative Importance')
  plt.show()
  foreset_variable = [features[i] for i in indices]
  return foreset_variable

def randomForestSelectorRanges(X, y, min, max):
  model = RandomForestClassifier(random_state=0, max_depth=10).fit(X, y)
  features = X.columns
  importances = model.feature_importances_
  thresholds = []
  praucs = []
  accuracys = []
  recalls = []
  precisions = []
  aucs = []
  for i in range(min, max):
    indices = np.argsort(importances)[-i:]
    foreset_variable = [features[i] for i in indices]
    x_train , x_test,y_train, y_test = train_test_split(X[foreset_variable ], y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(random_state=0, C=0.1).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    probs = clf.predict_proba(x_test)
    probs_rfs = probs[:,1]
    precision, recall, _ = precision_recall_curve(y_test, probs_rfs)
    prauc = auc(recall, precision)
    thresholds.append(i)
    praucs.append(prauc)
    accuracys.append(accuracy_score(y_pred, y_test))
    recalls.append(recall_score(y_pred, y_test))
    precisions.append(precision_score(y_pred, y_test))
    aucs.append(roc_auc_score(y_test, probs_rfs))
  plt.figure(figsize=(10,7))
  plt.plot(thresholds, praucs, marker=".", label='praucs')
  plt.plot(thresholds, accuracys, marker=".", label='accuracy')
  plt.plot(thresholds, recalls, marker=".", label='recall')
  plt.plot(thresholds, precisions, marker=".", label='precision')
  plt.plot(thresholds, aucs, marker=".", label='auc')
  plt.title('Metrics value for each threshold')
  plt.xlabel('threshold')
  plt.ylabel('Metrics')
  plt.legend()
  plt.show()

def Anova(X,Y,size):
    anova = f_classif(X,Y)
    p_values_anova = pd.Series(anova[1], index = X.columns)
    p_values_anova.sort_values(ascending = True , inplace= True)
    
    return p_values_anova.index[:size]


def gridSearch(X_train,Y_train, list_models):
  grid_models = list()
  for i in range(len(list_models)):
    grid = GridSearchCV(list_models[i]["Model"],list_models[i]["params"],cv=5,scoring='recall', verbose=1, n_jobs=-1)
    grid.fit(X_train,Y_train)
    grid_models.append(grid)
  return  grid_models

def tSNE(X, nComponent, perplexity=30, lr=200, randomState=0):
  return TSNE(n_components=nComponent, perplexity=perplexity, learning_rate=lr, random_state=randomState, verbose=1).fit_transform(X)

def plotTSNE2D(X, y):
  plt.figure(figsize=(16,10))
  plt.scatter(X[:,0], X[:,1], c=y)
  plt.xlabel('component_1')
  plt.ylabel('component_2')
  plt.show()

def plotHistory(history, loss=False, acc=False):
  if acc == True:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
  if loss == True:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def pipeline(data):
    rob_scaler = RobustScaler()
    scaled_amount = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
    scaled_time = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))
    data.insert(0, 'scaled_amount', scaled_amount)
    data.insert(1, 'scaled_time', scaled_time)
    data.drop(['Time','Amount'], axis=1, inplace=True)
    
    randomForestFeatures = randomForestSelector(X_RF, Y_RF, 8)
    
    cgans = cGAN(8)
    X_gans = data.loc[:, randomForestFeatures]
    y_train_gans = data['Class'].values.reshape(-1,1)
    pos_index = np.where(y_train_gans==1)[0]
    neg_index = np.where(y_train_gans==0)[0]
    cgans.train(X_gans.values, y_train_gans, pos_index, neg_index, epochs=2000)
    noise = np.random.normal(0, 1, (3508, 32))
    sampled_labels = np.ones(3508).reshape(-1, 1)
    gen_samples = cgans.generator.predict([noise, sampled_labels])
    data_class_fraud=data.loc[data['Class'] == 1]
    data_fraud_gans= np.concatenate((data_class_fraud.loc[:, randomForestFeatures],gen_samples), axis=0) 
    data_fraud_gans = np.concatenate((data_fraud_gans,np.ones((4000,1))),axis=1)
    data_nonfraud_gans= data[data['Class']==0].sample(n=4000).loc[:, randomForestFeatures]
    data_nonfraud_gans['Class']=0
    data_gans= np.concatenate((data_fraud_gans,data_nonfraud_gans), axis=0)
    np.random.shuffle(data_gans)
    Y_gans=data_gans[:, -1]
    X_gans=data_gans[:,:-1]
    clf = RandomForestClassifier(random_state=0, bootstrap=False, max_depth=20, min_samples_split=2, n_estimators=100)
    clf.fit(X_gans,Y_gans)
    return clf

