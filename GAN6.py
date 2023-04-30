# Импортируем необходимые библиотеки

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate, Reshape, Flatten, Conv1DTranspose, Conv1D
from tensorflow.keras.losses import BinaryCrossentropy

# Загрузка и обработка исторических данных
data = yf.download("EURUSD=X", start="2010-01-01", end="2023-04-30")
data = data.dropna()
data = data.drop(['Adj Close'], axis=1)

# Преобразование данных в форму, пригодную для обучения
X_train = []
y_train = []
for i in range(len(data)-60):
    X_train.append(np.array(data.iloc[i:i+60, :]))
    if data.iloc[i+60, 3] > data.iloc[i+59, 3]:
        y_train.append([1, 0, 0]) # Sell
    elif data.iloc[i+60, 3] < data.iloc[i+59, 3]:
        y_train.append([0, 1, 0]) # Buy
    else:
        y_train.append([0, 0, 1]) # NoTrade
X_train = np.array(X_train)
y_train = np.array(y_train)

# Определение генератора
def build_generator(latent_dim, n_outputs=3):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1536, activation='relu'))
    model.add(Dense(X_train.shape[1]*n_outputs, activation='tanh'))
    model.add(Reshape((X_train.shape[1], n_outputs)))
    return model

# Определение дискриминатора
def build_discriminator():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 3)))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model

# Сборка модели cGAN
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    gen_input, gen_condition = generator.input
    gen_output = generator.output
    cgan_output = discriminator([gen_output, gen_condition])
    model = Model(inputs=[gen_input, gen_condition], outputs=[cgan_output])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Определение размерности скрытого пространства
latent_dim = 100

# Создание экземпляров моделей
generator = build_generator(latent_dim)
discriminator = build_discriminator()
cgan = build_cgan(generator, discriminator)

# Обучение модели cGAN
def train_cgan(generator, discriminator, cgan, X_train, y_train, latent_dim, n_epochs=5000, n_batch=128):
    # Указание меток классов для дискриминатора
    fake_labels = np.zeros((n_batch, 1))
    real_labels = np.ones((n_batch, 1))

    # Цикл обучения модели
    for epoch in range(n_epochs):
        # Обучение дискриминатора
        # Выбор случайных реальных образцов
        idx = np.random.randint(0, X_train.shape[0], n_batch)
        X_real, y_real = X_train[idx], y_train[idx]
        # Генерация случайных латентных переменных и соответствующих торговых сигналов
        z_input = np.random.randn(n_batch, latent_dim)
        y_fake = generator.predict([z_input, y_real])
        # Обучение дискриминатора на реальных и сгенерированных образцах
        d_loss_real = discriminator.train_on_batch([X_real, y_real], real_labels)
        d_loss_fake = discriminator.train_on_batch([y_fake, y_real], fake_labels)
        # Расчет среднего значения функции потерь
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Обучение генератора
        # Генерация случайных латентных переменных и соответствующих торговых сигналов
        z_input = np.random.randn(n_batch, latent_dim)
        y_real = np.random.choice(y_train, size=n_batch)
        # Обучение генератора на сгенерированных образцах и их классах
        g_loss = cgan.train_on_batch([z_input, y_real], real_labels)

        # Вывод функции потерь на каждой эпохе
        print(f"Epoch: {epoch + 1}/{n_epochs}, d_loss={d_loss}, g_loss={g_loss}")


def plot_results(history, signals):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # График качества обучения
    axs[0].plot(history.history['loss'], label='train')
    axs[0].plot(history.history['val_loss'], label='validation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # График прибыли
    buy_signals = signals[:, :, 1]
    sell_signals = signals[:, :, 0]
    profit = np.zeros(buy_signals.shape)
    for i in range(1, len(profit)):
        if buy_signals[i] == 1:
            if sell_signals[i - 1] == 1:
                profit[i] = profit[i - 1] + (data.iloc[i + 59, 3] - data.iloc[i + 58, 3])
            else:
                profit[i] = profit[i - 1] - 0.0002
        elif sell_signals[i] == 1:
            if buy_signals[i - 1] == 1:
                profit[i] = profit[i - 1] + (data.iloc[i + 58, 3] - data.iloc[i + 59, 3])
            else:
                profit[i] = profit[i - 1] - 0.0002
        else:
            profit[i] = profit[i - 1]
    axs[1].plot(profit)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Profit')
    plt.show()