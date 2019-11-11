import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.naive_bayes import GaussianNB 
import seaborn as sn
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# real = real music
# gen = generated music

path = './genre_data_norm_0_1.csv'
gen_path = './gen_data_norm_0_1.csv'
val_frac = 0.1
real_data = pd.read_csv(path)

genres = real_data['label'].unique()
genre2int = {g: i for g, i in zip(genres, range(len(genres)))}    
int2genre = {i: g for g, i in genre2int.items()}

def encode_genres(data):
	"""
	encodes the 8 genres into integers (from 0 to 8)
	"""
    for i in range(len(data)):
        data.at[i, 'label'] = genre2int[data.at[i, 'label']]

def get_data_and_labels(dataset):
	"""
	separates the labels from the features
	"""
    data = dataset.iloc[:, :-1]
    labels = dataset.iloc[:, -1]
    data = np.array(data.values, dtype=float)
    labels = np.array(labels.values, dtype=float)
    return data, labels

def get_k_best(X, y, k):
    """
    returns the indices of the k best attributes for each emotion type
    """
    kbest = SelectKBest(f_regression, k)
    kbest.fit(X, y)
    best_attributes = kbest.get_support(True)
    return best_attributes

def reduce_attr(data, attr_indices):
    """
    saves the data at certain indices (which are in attr_indices)
    into the new dataset
    """
    new = []
    for i in range(len(data)):
        new.append([data[i][index] for index in attr_indices])
    return new

"""
GENRE ENCODING AND DATA PREP
"""
encode_genres(real_data)
real_train = real_data.iloc[:int(len(real_data) * (1 - val_frac)) + 1, :]
real_val = real_data.iloc[-int(len(real_data) * val_frac):, :]
gen_data = pd.read_csv(gen_path)
encode_genres(gen_data)

real_train_X, real_train_y = get_data_and_labels(real_train)
real_val_X, real_val_y = get_data_and_labels(real_val)
gen_data_X, gen_data_y = get_data_and_labels(gen_data)


print("LOGISTIC REGRESSION")
log = LogisticRegression()
log = log.fit(real_train_X, real_train_y)
pred = log.predict(real_val_X)
cm = confusion_matrix(real_val_y, pred)
df_cm=pd.DataFrame(cm, index = [int2genre[i] for i in range(8)], columns = [int2genre[i] for i in range(8)])
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
accuracy = log.score(real_val_X, real_val_y)
print(f'accuracy= {accuracy * 100}%')


print("\nGAUSSIAN NAIVE BAYES")
gnb = GaussianNB()
gnb = gnb.fit(real_train_X, real_train_y) 
pred = gnb.predict(real_val_X) 
cm = confusion_matrix(real_val_y, pred)
cm = np.array(cm, dtype=int)
df_cm=pd.DataFrame(cm, index = [int2genre[i] for i in range(8)], columns = [int2genre[i] for i in range(8)])
sn.heatmap(df_cm, annot=True)
accuracy = gnb.score(real_val_X, real_val_y)
print(f'accuracy= {accuracy * 100}%')


print("\nK_NEAREST NEIGHBOUR")
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(real_train_X, real_train_y)
pred = knn.predict(real_val_X)
cm = confusion_matrix(real_val_y, pred)
df_cm=pd.DataFrame(cm, index = [int2genre[i] for i in range(8)], columns = [int2genre[i] for i in range(8)])
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
accuracy = knn.score(real_val_X, real_val_y)
print(f'accuracy= {accuracy * 100}%')