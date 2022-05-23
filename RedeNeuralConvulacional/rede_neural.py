from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

# cria o mapa de características (max pool) - primeira camada
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32, (3,3), activation='relu'))

# cria o mapa de características (max pool) - segunda camada
classifier.add(MaxPooling2D(pool_size = (2,2)))

# adiciona a camada de flattening
classifier.add(Flatten())


# criar camada de entrada
classifier.add(Dense(units=128, activation='relu'))
# criar camada de saída
classifier.add(Dense(units=1, activation='sigmoid'))

# compilar a rede neural
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# pré-processa as imagens para evitar super ajuste
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set', target_size=(64,64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set', target_size=(64,64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)