import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

labelecoder_X_1 = LabelEncoder()
X[:,1] = labelecoder_X_1.fit_transform(X[:,1])
labelecoder_X_2 = LabelEncoder()
X[:,2] = labelecoder_X_2.fit_transform(X[:,2])

onehotencoder = make_column_transformer((OneHotEncoder(categories='auto', sparse=False), [1]), remainder='passthrough')
X = onehotencoder.fit_transform(X)
X = X[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)