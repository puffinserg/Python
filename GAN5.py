import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Импорт необходимых библиотек
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Dropout
from tensorflow.keras.losses import BinaryCrossentropy

#from tensorflow.keras.layers import concatenate

BATCH_SIZE = 16     #размер батча
hidden_dim = 100    #размерность скрытого пространства

# Определение генератора
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=hidden_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(3, activation='softmax'))
    model.add(Reshape((3, 1)))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model

# Определение дискриминатора
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model

def generator_loss(fake_output):
  loss = cross_entropy(tf.ones_like(fake_output), fake_output)
  return loss

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

# Загрузка исторических данных EUR/USD
df = yf.download("EURUSD=X", start="2010-01-01", end="2023-04-26")  #(3469, 6)  <class 'pandas.core.frame.DataFrame'>

# Создание новых признаков на основе старых данных
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_100'] = df['Close'].rolling(window=100).mean()

# Удаление строк с пропущенными значениями
df.dropna(inplace=True) #(3370, 11)

df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().copy() #(3369, 12)

# Преобразование данных в массив numpy
X = np.array(df[['Open', 'High', 'Low', 'Close', 'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100']]) #(3369, 9)
# Создание массива Y
Y = np.zeros((X.shape[0], 3, 1))    #(3369, 3, 1)

# Заполнение массива Y значениями
for i in range(X.shape[0]):
    if df['Returns'][i] > 0:
        Y[i] = np.array([1, 0, 0]).reshape(3, 1)
    elif df['Returns'][i] < 0:
        Y[i] = np.array([0, 1, 0]).reshape(3, 1)
    else:
        Y[i] = np.array([0, 0, 1]).reshape(3, 1)

BUFFER_SIZE = X.shape[0]
BUFFER_SIZE = BUFFER_SIZE // BATCH_SIZE * BATCH_SIZE
X = X[:BUFFER_SIZE] #(3360, 9, 1)
Y = Y[:BUFFER_SIZE] #(3360, 3, 1)

# Нормализация данных
X = (X - np.min(X)) / (np.max(X) - np.min(X))
X = X.reshape(-1, X.shape[1], 1)
print("Size X:",X.shape," Size Y:", Y.shape)
XY = np.concatenate((X, Y), axis=1)
train_dataset = tf.data.Dataset.from_tensor_slices(XY).batch(BATCH_SIZE)     #(32, 12, 1) <class 'tensorflow.python.data.ops.batch_op._BatchDataset'>

def dropout_and_batch():
  return Dropout(0.3)(BatchNormalization())

generator = build_generator(hidden_dim)             #<keras.engine.sequential.Sequential object at 0x000001C348C1E8C0>
input_shape = [12,1]
discriminator = build_discriminator(input_shape)    #<keras.engine.sequential.Sequential object at 0x000001C785F2D990>
#print(generator.summary())
#print(discriminator.summary())

# потери
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)    #<keras.losses.BinaryCrossentropy object at 0x000002887248EFB0>

generator_optimizer = tf.keras.optimizers.Adam(1e-4)        #<keras.optimizers.adam.Adam object at 0x0000017BC5C1BE50>
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)    #<keras.optimizers.adam.Adam object at 0x000002DB3CE60550>

# обучение
@tf.function
def train_step(images):
  print("images.shape=",images.shape)
  noise = tf.random.normal([BATCH_SIZE, hidden_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    print("images.shape=", images.shape, type(images.shape))
    print("generated_images=", generated_images,type(generated_images))
    real_output = discriminator(images, training=True)
    print("real_output=",real_output.shape)
    images = tf.cast(images, tf.float32)
    fake_output = discriminator(tf.concat([images[:,:9,:], generated_images], axis=1), training=True)
    print("fake_output=", fake_output.shape)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return gen_loss, disc_loss


def train(dataset, epochs):
  history = []
  MAX_PRINT_LABEL = 10
  th = BUFFER_SIZE // (BATCH_SIZE * MAX_PRINT_LABEL)

  for epoch in range(1, epochs + 1):
    print(f'{epoch}/{EPOCHS}: ', end='')

    start = time.time()
    n = 0

    gen_loss_epoch = 0
    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
      gen_loss_epoch += K.mean(gen_loss)
      if (n % th == 0): print('=', end='')
      n += 1

    history += [gen_loss_epoch / n]
    print(': ' + str(history[-1]))
    print('Время эпохи {} составляет {} секунд'.format(epoch, time.time() - start))

  return history

# запуск процесса обучения
EPOCHS = 20
print("EPOCHS=",EPOCHS)
history = train(train_dataset, EPOCHS)

plt.plot(history)
plt.grid(True)
plt.show()

# Переопределение Y
Y1 = generator.predict(X)
#model.fit(X, Y, epochs=20, batch_size=32)
print(Y1.shape, Y.shape)