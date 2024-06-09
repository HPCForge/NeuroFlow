import pandas as pd#read in the data using pandas
from sklearn.model_selection import train_test_split

df = pd.read_csv("./out.csv")#check data has been read in properly
#print(df.head())
print(df.shape)

data = df.drop(columns=['Group'])
#print(data.head())

targets = df['Group'].values
#print(targets[4000:])

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=1, stratify=targets)
print(test_data)
from sklearn.neighbors import KNeighborsClassifier# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)# Fit the classifier to the data
knn.fit(train_data,train_targets)

#check accuracy of our model on the test data
score = knn.score(test_data, test_targets)

print(score)

from sklearn.model_selection import cross_val_score
import numpy as np#create a new KNN model
import time
knn_cv = KNeighborsClassifier(n_neighbors=7)#train model with cv of 5
cv_scores = cross_val_score(knn_cv, data, targets, cv=5)#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

from sklearn.model_selection import GridSearchCV#create new a knn model
knn2 = KNeighborsClassifier()#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)#fit model to data
knn_gscv.fit(data, targets)

#check top performing n_neighbors value
print(knn_gscv.best_params_)

print(knn_gscv.best_score_)
import matplotlib.pyplot as plt
import seaborn as sn
start_time = time.time()
preds=knn_gscv.best_estimator_.predict(test_data)
print(time.time()-start_time)
from sklearn.metrics import confusion_matrix
classes = ('A','B', 'EB', 'S','SS','SW','U')                                    
cf_matrix = confusion_matrix(test_targets, preds)                                    
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])                            
df_cm = df_cm.reindex(["B", "EB", "S", "SS", "SW", "U", "A"])                   
df_cm = df_cm.reindex(columns=["B", "EB", "S", "SS", "SW", "U", "A"])           
df_cm = df_cm.round(3)                                                                                
plt.figure(figsize = (15,10))                                                   
plt.title('Fourier-based Classifier', fontsize="40")                                           
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 34}, cbar=False, fmt=".2g")          
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)               
ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)                           
plt.savefig('unifiedFTReordered.png')                                          
print(df_cm)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])
print(df_cm)
