# This code predicts the sale/purchase of insurance products using mahcine learning models and compares the performance of three models

# ## PREDICTIVE MODELLING
# #### LOADING LIBRARIES & READING DATA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import numpy as np

# load data
df= pd.read_csv('filepath')

# view top 5 observations
df.head()

# view bottom 5 observations
df.tail()

# ## PART I: DESCRIPTIVE STATISTICS & DATA EXPLORATION

# ### <font color= red> *A. Descriptive Statistics*

# check shape of data i.e number of records and columns
df.shape

# check if any missing value in any variable
df.isnull().any()

# check the data type of each column, number of columns, memory usage,and the number of records in the dataset
df.info()

#identifying missing value in specific variables as observed in previous step
edu_nan_count=df['education'].isna().sum()
mar_nan_count=df['marriage'].isna().sum()
ho_nan_count=df['house_owner'].isna().sum()

print("The number of missing values in education column is: " + str(edu_nan_count))
print("The number of missing values in marriage column is: " + str(mar_nan_count))
print("The number of missing values in house_owner column is: " + str(ho_nan_count))

# check summary of the dataset- numerical variables
df.describe()

# check summary of the dataset- categorical variables
df.describe(include= 'object')

# check outliers in numeric variable - house_val
z_scores = (df['house_val'] - np.mean(df['house_val'])) / np.std(df['house_val'])
outliers = np.where(abs(z_scores) > 3)
len(outliers[0])

# ###  <font color= red> *B. Data Exploration Through Visualisation*

# Barplot of Flag variable
df.flag.value_counts().plot.barh() # shows equal response categories

# Barplot of Gender variable
df.gender.value_counts().plot.barh() #3 categories

# Barplot of Education variable
df.education.value_counts().plot.barh()

# Barplot of age variable
df.age.value_counts().plot.barh()

# Barplot of online purchase variable
df.online.value_counts().plot.barh()

# Barplot of Marriage variable
df.marriage.value_counts().plot.barh() #missing values are there

# Barplot of child variable
df.child.value_counts().plot.barh()

# Barplot of Occupation variable
df.occupation.value_counts().plot.barh()

# Barplot of mortgage
df.mortgage.value_counts().plot.barh()

# Barplot of region variable
df.region.value_counts().plot.barh()

# Barplot of fam_income variable
df.fam_income.value_counts().plot.barh()

# ## DATA PREPROCESSING

# #### <font color = blue> Data Quality issues identified at data exploratory stage shall be addressed here. Missing values will be dealt with. Categorical data will be changed to numerical categories. House_val column will be binned. 

# creating duplicate data set to clean it
df_clean = df.copy()

# Dealing with null values in marriage column
# creating a new category 'not disclosed' as more than 35% values are missing
df_clean['marriage'].fillna("not disclosed", inplace=True)
df_clean.marriage.value_counts()

# Dropping null values in education column 
# missing values only 741- dropping them will not affect the data much
df_clean = df_clean.dropna(axis = 0,subset = ['education']) #axis=0 prompts to drop rows with null value in the chosen variable column, axis=1 would do drop column with null values

# replacing null values in house_owner column with 'owner' if value in house_val is more than 0, else,'renter'
df_clean.loc[(df_clean['house_owner'].isnull()) & (df_clean['house_val'] == 0), 'house_owner'] = 'Renter'
df_clean.loc[(df_clean['house_owner'].isnull()) & (df_clean['house_val'] > 0), 'house_owner'] = 'Owner'
df_clean.house_owner.value_counts()

# merging category '0' in child variable with 'N'
df_clean.loc[df_clean['child'] == '0', 'child']= 'N'
df_clean.child.value_counts()

# creating a new column new_house_val and replacing 0 value in house_val column with mean of house value for that region
df_clean['new_house_val'] = df_clean.groupby('region')['house_val'].transform(lambda x: x.replace(0, x.mean()).astype(int))
df_clean.tail()

# categorizing 'house_val' variable by creating new column named 'house_val_bin'
bins = [0, 100000, 500000, 1000000, 5000000, 10000000]
labels = ['Minimum val1', 'Minimum val2','Average val', 'premium1', 'premium2']
df_clean.loc[:, 'house_val_bin'] = pd.cut(x = df_clean['new_house_val'], bins = bins, labels = labels, include_lowest = True)
df_clean.info() # to verify new column is created
df_clean.tail()  # to check that changes have taken place

# creating a copy of clean data
df_clean_new = df_clean.copy()  #for visualisations before converting categorical variables into numeric

# converting categorical variables to numeric categories- all except
# occupation, region, fam_income, mortgage
df_clean['flag'] = df_clean['flag'].replace(['Y','N'],[1,0])
df_clean['gender'] = df_clean['gender'].replace(['M','F','U'],[0,1,2])
df_clean['education'] = df_clean['education'].replace(['0. <HS','1. HS','2. Some College','3. Bach','4. Grad'],
                                        [0,1,2,3,4]).astype(int)
df_clean['age'] = df_clean['age'].replace(['1_Unk','2_<=25','3_<=35','4_<=45','5_<=55','6_<=65','7_>65'],
                            [1,2,3,4,5,6,7])
df_clean['online'] = df_clean['online'].replace(['N', 'Y'], [0,1])
df_clean['marriage'] = df_clean['marriage'].replace(['Single', 'Married', 'not disclosed'], [0,1,2])
df_clean['child'] = df_clean['child'].replace(['N', 'Y', 'U'], [0,1,2])
df_clean['house_owner'] = df_clean['house_owner'].replace(['Renter', 'Owner'], [0,1])
df_clean.head()

# Writing a copy of cleaned data to a new file
df_clean.to_csv('/Users/jyotismac/Desktop/sales_data_clean.csv', index=False)
df_clean_new.to_csv('/Users/jyotismac/Desktop/sales_data_clean_new.csv', index=False)

# ## PART II: VISUALISATIONS

# import required libraries
import seaborn as sns
import plotly.express as px

# Age by house_val_bin
plt.figure(figsize=(8,5))
sns.boxplot(x = 'house_val_bin', y = 'age', hue = 'flag', data = df_clean, palette = 'rainbow')
plt.title("Age by house_val_bin")

# barplot showing Policy purchased count grouped by gender
plt.figure(figsize=(8,5))
sns.countplot(x = 'flag', hue = 'gender', data = df_clean_new, palette = 'magma')
plt.title('Count of Policy Purchased grouped by gender')
plt.xlabel('Policy Purchased')

# barplot showing Policy purchased count grouped by education
plt.figure(figsize=(8,5))
sns.countplot(x = 'flag', hue = 'education', data = df_clean_new, palette = 'magma')
plt.title('Count of Policy Purchased grouped by education')
plt.xlabel('Policy Purchased')

# barplot showing Age of house count grouped by house owner
plt.figure(figsize=(10,6))
sns.violinplot(x='house_val_bin',y="age",data=df_clean, hue='house_owner',  palette='rainbow')
plt.title("Violin Plot of Age by house_val_bin, Separated by house_owner")


df_region_house_val=df_clean_new[df_clean_new['house_val']<=df_clean_new['house_val'].quantile(0.75)]
sns.boxplot(x='mortgage', y='house_val', data=df_region_house_val, hue='flag')
plt.legend()

# histogram of age distribution over insurance purhcase
# repeated in following plots too
figure = px.histogram(df_clean_new, x = "age", 
                      color = "flag", 
                      title= "Age Distribution over Insurance Purchase")
figure.show()

# histogram of education distribution over insurance purhcase
figure = px.histogram(df_clean_new, x = "education", 
                      color = "flag", 
                      title= "Education Distribution over Insurance Purchase")
figure.show()

# histogram of gender distribution over insurance purhcase
figure = px.histogram(df_clean_new, x = "gender", 
                      color = "flag", 
                      title= "Gender Distribution over Insurance Purchase")
figure.show()

# histogram of online distribution over insurance purhcase
figure = px.histogram(df_clean_new, x = "online", 
                      color = "flag", 
                      title= "Online Distribution over Insurance Purchase")
figure.show()

# histogram of marriage distribution over insurance purhcase
figure = px.histogram(df_clean_new, x = "marriage", 
                      color = "flag", 
                      title= "Marriage Distribution over Insurance Purchase")
figure.show()

# histogram of 'child' distribution over insurance purhcase

figure = px.histogram(df_clean_new, x = "child", 
                      color = "flag", 
                      title= "Child Distribution over Insurance Purchase")
figure.show()

# histogram of occupation distribution over insurance purhcase
figure = px.histogram(df, x = "occupation", 
                      color = "flag", 
                      title= "Occupation Distribution over Insurance Purchase")
figure.show()

# histogram of Mortgage distribution over insurance purhcase
figure = px.histogram(df, x = "mortgage", 
                      color = "flag", 
                      title= "Mortgage Distribution over Insurance Purchase")
figure.show()

# histogram of house owner distribution over insurance purhcase
figure = px.histogram(df, x = "house_owner", 
                      color = "flag", 
                      title= "House Owner Distribution over Insurance Purchase")
figure.show()

# histogram of region distribution over insurance purhcase
figure = px.histogram(df, x = "region", 
                      color = "flag", 
                      title= "Region Distribution over Insurance Purchase")
figure.show()

# histogram of family income distribution over insurance purhcase
figure = px.histogram(df, x = "fam_income", 
                      color = "flag", 
                      title= "Family Income Distribution over Insurance Purchase")
figure.show()

# ## PART III: MACHINE LEARNING MODELS

  # ### A. RESAMPLING AND SPLITTING

  # #### <font color =blue> *Preparing data for machine learning models*

# reading the clean data file
new_df = pd.read_csv('/Users/jyotismac/Desktop/sales_data_clean.csv')
new_df.info()

# dropping the columns not required or whose new column is created
new_df = new_df.drop(['house_val', 'new_house_val'], axis = 1)

# creating dummy columns for remaining stringed columns - occupation, region, fam_income, mortgage
new_df = pd.get_dummies(new_df, columns=['occupation','mortgage','region','fam_income','house_val_bin'])

# checking the size of categories of response variable
new_df.flag.value_counts()

  # #### <font color=blue> *resample the data to make it equal for each response category*

# import library
from sklearn.utils import resample

# downsampling the majority class to have a 50:50 split 
purchased_class = new_df[new_df['flag'] == 1]
not_purchased_class = new_df[new_df['flag'] == 0]
not_purchased_sample = resample(not_purchased_class, replace =True,
                                n_samples= len(purchased_class), random_state =123 )
print(not_purchased_sample.shape)

# binding the resampled data
prepared_data = pd.concat([purchased_class, not_purchased_sample])
print(prepared_data.flag.value_counts())
prepared_data.info()

  # #### <font color= blue> *splitting the data into train and test sets*

# import library
from sklearn.model_selection import train_test_split

# creating x and y objects containing independent and response variables respectively
x = prepared_data.drop('flag', axis=1)
y = prepared_data.flag

# splitting the data into train and test set in the ratio of 80:20
np.random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=123)

  # ### B. MACHINE LEARNING MODEL TRAINING AND PREDICTION

  # #### <font color=blue> *Import all the required libraries for ML*

# importing the required the libraries 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

  # ### <font color=red>*B.1. LOGISTIC REGRESSION*
# running Logistic Regression Classifier
clf1 = LogisticRegression(max_iter = 1000, random_state=123, multi_class='ovr',solver='liblinear')

# training the model on train data set
model1 = clf1.fit(x_train, y_train)

# predicting probabilities 
pred_prob1 = model1.predict_proba(x_test)

# predicting the response variable on test data
y_pred1 = clf1.predict(x_test)

# calculating the accuracy of the model
accuracy1 = accuracy_score(y_test, y_pred1)

# calculating precision score of the model
p_score1 = precision_score(y_test, y_pred1)

# calculating recall score of the model
r_score1 = recall_score(y_test, y_pred1)

# calculating F-1 score of the model
F1_score1 = f1_score(y_test, y_pred1)

# Generating the classification report
print("Accuracy (train) for %s: %0.1f%% " % (clf1, accuracy1 * 100))
print("Precision Score for %s: %0.2f " % (clf1, p_score1))
print("Recall Score for %s: %0.2f " % (clf1, r_score1))
print("F1-Score for %s: %0.2f " % (clf1, F1_score1))
print("________________________________________________________________")
print ("               -------Classification Report-------")
print(classification_report(y_test, y_pred1))
print("________________________________________________________________")

# Generating the confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)
cmd1 = ConfusionMatrixDisplay(cm1, display_labels=['class 0 (Not purchased)', 'class 1 (purchased)']) #find reference
print("Confusion Matrix for %s  " % (clf1))
cmd1.plot()
plt.title(" Confusion Matrix of Logistic Regression")
plt.savefig('Logistic Regression',dpi=300)

# Cross validation using K-fold cross validation method

# Set the number of folds
num_folds = 5

X = x #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation on the model
scores = cross_val_score(clf1, X, Y, cv=kfold)

# Print the average validation score and the standard deviation
print("Average validation score of Logistic Regression model is :", scores.mean())
print("The Standard deviation of Logistic Regression is:", scores.std())

  # ### <font color=red>*B.2. RANDOM FOREST*

# running Logistic Regression Classifier
clf2 = RandomForestClassifier(n_estimators=200, random_state=123)

# training the model on train data set
model2 = clf2.fit(x_train, y_train)

# predicting probabilities 
pred_prob2 = model2.predict_proba(x_test)

# predicting the response variable on test data
y_pred2 = clf2.predict(x_test)

# calculating the accuracy of the model
accuracy2 = accuracy_score(y_test, y_pred2)

# calculating precision score of the model
p_score2 = precision_score(y_test, y_pred2)

# calculating recall score of the model
r_score2 = recall_score(y_test, y_pred2)

# calculating F-1 score of the model
F1_score2 = f1_score(y_test, y_pred2)

# Generating the classification report

print("Accuracy (train) for %s: %0.1f%% " % (clf2, accuracy2 * 100))
print("Precision Score for %s: %0.2f " % (clf2, p_score2))
print("Recall Score for %s: %0.2f " % (clf2, r_score2))
print("F1-Score for %s: %0.2f " % (clf2, F1_score2))
print("________________________________________________________________")
print ("               -------Classification Report-------")
print(classification_report(y_test, y_pred2))
print("________________________________________________________________")

# Generating the confusion matrix
cm2 = confusion_matrix(y_test, y_pred2)
cmd2 = ConfusionMatrixDisplay(cm2, display_labels=['class 0 (Not purchased)', 'class 1 (purchased)']) #find reference
print("Confusion Matrix for %s  " % (clf2))
cmd2.plot()
plt.title(" Confusion Matrix of Random Forest Classifier")
plt.savefig('Random Forest',dpi=300)

# Cross validation using K-fold cross validation method

# Set the number of folds
num_folds = 5
X = x #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation on the model
scores = cross_val_score(clf2, X, Y, cv=kfold)

# Print the average validation score and the standard deviation
print("Average validation score of Logistic Regression model is :", scores.mean())
print("The Standard deviation of Logistic Regression is:", scores.std())

  # ### <font color=red>*B.3. SUPPORT VECTOR MACHINE*

# running  Support Vector Classifier
clf3 = SVC (kernel='linear', C=10, probability=True,random_state=0)

# training the model on train data set
model3 = clf3.fit(x_train, y_train)

# predicting probabilities 
pred_prob3 = model3.predict_proba(x_test)

# predicting the response variable on test data
y_pred3 = clf3.predict(x_test)

# calculating the accuracy of the model
accuracy3 = accuracy_score(y_test, y_pred3)

# calculating precision score of the model
p_score3 = precision_score(y_test, y_pred3)

# calculating recall score of the model
r_score3 = recall_score(y_test, y_pred3)

# calculating F-1 score of the model
F1_score3 = f1_score(y_test, y_pred3)

# Generating the classification report
print("Accuracy (train) for %s: %0.1f%% " % (clf3, accuracy3 * 100))
print("Precision Score for %s: %0.2f " % (clf3, p_score3))
print("Recall Score for %s: %0.2f " % (clf3, r_score3))
print("F1-Score for %s: %0.2f " % (clf3, F1_score3))
print("________________________________________________________________")
print ("               -------Classification Report-------")
print(classification_report(y_test, y_pred3))
print("________________________________________________________________")

# Generating the confusion matrix
cm3 = confusion_matrix(y_test, y_pred3)
cmd3 = ConfusionMatrixDisplay(cm3, display_labels=['class 0 (Not purchased)', 'class 1 (purchased)']) #find reference
print("Confusion Matrix for %s  " % (clf3))
cmd3.plot()
plt.title(" Confusion Matrix of Support Vector Classifier")
plt.savefig('Support Vector Classifier',dpi=300)

# Cross validation using K-fold cross validation method

# Set the number of folds
num_folds = 5
X = x #independent variables
Y = y #response variable

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform K-fold cross-validation on the model
scores = cross_val_score(clf3, X, Y, cv=kfold)

# Print the average validation score and the standard deviation
print("Average validation score of Support Vector Classifier model is :", scores.mean())
print("The Standard deviation of Support Vector Classifier is:", scores.std())


  # ### <font color= red> *B.4 ROC CURVES*

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# AUC scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])

print(auc_score1, auc_score2, auc_score3)

# blank canvass
plt.style.use('seaborn')

# plot ROC curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='SVC')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue', label='baseline ROC for TPR = FPR')

# title
plt.title('ROC curve')

# x label
plt.xlabel('False Positive Rate')

# y label
plt.ylabel('True Positive rate')

# legend
plt.legend(loc='best')

plt.savefig('ROC',dpi=300)

# ### <font color= red>*B.5 Precision - Recall Curve*
from sklearn.metrics import precision_recall_curve

#calculate precision and recall
precision1, recall1, thresholds1 = precision_recall_curve(y_test, pred_prob1[:,1], pos_label=1)
precision2, recall2, thresholds2 = precision_recall_curve(y_test, pred_prob2[:,1], pos_label=1)
precision3, recall3, thresholds3 = precision_recall_curve(y_test, pred_prob3[:,1], pos_label=1)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall1, precision1, color='orange')
ax.plot(recall2, precision2, color='green')
ax.plot(recall3, precision3, color='red')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

# ### running additional classifiers in a loop
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
C = 10
# Create different classifiers.
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'KNN classifier': KNeighborsClassifier(C),
    'AdaBoost Classifier' : AdaBoostClassifier(),
    'LDA' : LinearDiscriminantAnalysis()
}

n_classifiers = len(classifiers)
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(x_train, np.ravel(y_train))
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels = ['class 0', 'class 1'])
    print("Confusion Matrix for %s " % (name))
    cmd.plot()
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test, y_pred))
    
# ## <font color =red>*END OF THE CODE *
