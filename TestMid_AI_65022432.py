from sklearn.tree import DecisionTreeClassifier, plot_tree#tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt #tree
import seaborn as sns #graph
import pandas as pd
import numpy as np
 
file_path = 'D:/65022432/data/'
file_name = 'car_data.csv'
 
df = pd.read_csv(file_path + file_name)

#preprocessing
en = LabelEncoder()
cols = ['Purchased']
df[cols] = df[cols].apply(en.fit_transform)
 
df.drop(columns=['User ID'],inplace=True)
    
df.dropna(inplace=True)
 
x = df.iloc[:,1:4]
y = df.iloc[:,0]

#train_model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)
 
#validate
score_train = model.score(x_train, y_train)
print(f'Accuracy_train: {score_train:.2f}')
 
score_test = model.score(x_test, y_test)
print(f'Accuracy_test: {score_test:.2f}')
 
#tree
feature = x.columns.tolist()
Data_class = y.tolist()
 
plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names = feature,
              class_names=Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16,
              max_depth=5
              )
plt.show()
 
#graph
feature_importances = model.feature_importances_
feature_names = ['Age','AnnualSalary','Purchased']
 
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x = feature_importances, y = feature_names)
