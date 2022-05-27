import numpy as np # подключение необходимых библиотек
import math
from sklearn.metrics import accuracy_score
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import os
import keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.utils import np_utils



#фукция для вывода изображения
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    return plt


#загрузка обучающего и тестового набора данных из базы данных mnist
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_val, y_val) = mnist.load_data()

#нормирование значений подающихся на вход нейронной сети
x_train=x_train = x_train.astype(np.float) / 255 - 0.5
x_val = x_val.astype(np.float) / 255 - 0.5

#приведение входных наборов данных к необходимой размерности
x_train=x_train.reshape(-1,28*28)
x_val=x_val.reshape(-1,28*28)

#приведение выходных наборов данных к векторному виду
y_train = np_utils.to_categorical(y=y_train, num_classes=10)
y_val = np_utils.to_categorical(y=y_val, num_classes=10)

#выбор диапазона значений обучающей и тестовой выборки
x_train=x_train[0:1000]
y_train=y_train[0:1000]
x_val=x_val[0:100]
y_val=y_val[0:100]
x_test=x_val[10:15]
y_test=y_val[10:15]

#задание скорость обучения и количества эпох
learningrate=0.25
ep=50

#инициализация сети объетом model, добавление трех слоев в сеть
#слои строятся в соответствии с фукнцией sigmoid
K.clear_session()
model = M.Sequential()
model.add(L.Dense(output_dim=128, input_dim=784,activation='sigmoid'))
model.add(L.Dense(output_dim=128, activation='sigmoid'))
model.add(L.Dense(output_dim=10, activation='sigmoid'))

#определение параметра optimizer для model
optimizer1=keras.optimizers.RMSprop(learningrate, rho=0.7)

#компиляция сети с средне-квадратичной ошибкой
model.compile(optimizer1,loss='mse',metrics=None)

#задание пути для записи рельзутатов callbacks
BASE_DATA_PATH = 'C:\mnist'
mylog_dir = os.path.join( BASE_DATA_PATH, "train_log")
#определение параметров callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=mylog_dir,  write_graph=False, write_images=False , profile_batch=28, embeddings_freq=0,embeddings_layer_names=None, embeddings_metadata=None,embeddings_data=None, update_freq='epoch')
print(os.path.join(mylog_dir, 'train'))

#обучение нейронной сети с использованием проверочных данных
model.fit(x_train, y_train,  batch_size=28, epochs=ep, callbacks=[tensorboard_callback], validation_data=[x_val, y_val])

#тестирование сети и вывод ошибки
results=model.evaluate(x_val,y_val,batch_size=28)
print(results)

indexes=[0,1,2,3,4,5,6,7,8,9]
#разница между требуемым результатом и  предполагаемым результатом обработки массива x_test сетью
predictions=y_test-model.predict(x_test,batch_size=28)


#вывод результатов распознавания
for i in range(5):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    l=gen_image(x_test[i-1])
    plt.subplot(1,2,2)
    thisplot = plt.bar(indexes, predictions[i-1])
    plt.ylim([0, 1])      


  
        
