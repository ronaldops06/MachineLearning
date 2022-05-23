import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# importar dados de treino
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# separando somente as colunas necessárias (número 2)
training_set = dataset_train.iloc[:,1:2].values

# faz a normalização (coloca os valores na mesma escala, entre 0 e 1)
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# cria uma estrutura de dados com os valores de um intervalo de tempo (utilizado 60 dias)
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

# transforma os dados no formado requerido pelo keras (3D)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# inicializa a rede
regressor = Sequential()

# cria a primeira camada da rede
# units(neurônios) = 50
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
# desabilita alguns neurônios para evitar over shape(super ajuste)
regressor.add(Dropout(0.2))

# cria a segunda camada da rede
# units(neurônios) = 50
regressor.add(LSTM(units=50, return_sequences=True))
# desabilita alguns neurônios para evitar over shape(super ajuste)
regressor.add(Dropout(0.2))

# cria a terceira camada da rede
# units(neurônios) = 50
regressor.add(LSTM(units=50, return_sequences=True))
# desabilita alguns neurônios para evitar over shape(super ajuste)
regressor.add(Dropout(0.2))

# cria a quarta camada da rede
# units(neurônios) = 50
regressor.add(LSTM(units=50))
# desabilita alguns neurônios para evitar over shape(super ajuste)
regressor.add(Dropout(0.2))

# cria a camada de saída
# units(neurônios) = 1
regressor.add(Dense(units=1))

# compila a rede
regressor.compile(optimizer='adam', loss='mean_squared_error')
# executa o treinamento da rede
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# previsões
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# axis(eixo) = 0 (horizontal-linhas)
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Dados reais de ações do Google')
plt.plot(predicted_stock_price, color='blue', label='Dados previstos de ações do Google')
plt.title('Previsão de preções de ações')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()
