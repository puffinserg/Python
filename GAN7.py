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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Reshape, Flatten, Conv1DTranspose, Conv1D
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
X_train = np.array(X_train)     #(3411, 60, 5)
y_train = np.array(y_train)     #(3411, 3)

# Определение генератора
def build_generator(latent_dim, n_outputs=3):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1536, activation='relu'))
    model.add(Dense(X_train.shape[1] * n_outputs, activation='tanh'))
    model.add(Reshape((X_train.shape[1], n_outputs)))

    # define the inputs to the generator as a list of two tensors
    gen_input = Input(shape=(latent_dim,))
    gen_condition = Input(shape=(n_outputs,))
    generator_output = model(gen_input)

    # concatenate the generated image with the label
    cgan_output = Concatenate()([generator_output, gen_condition])

    # define the generator model with two inputs
    model = Model(inputs=[gen_input, gen_condition], outputs=[cgan_output])
    return model


# Определение дискриминатора
def build_discriminator():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2] + 1)))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # define the inputs to the discriminator as a list of two tensors
    disc_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
    disc_condition = Input(shape=(X_train.shape[2] + 1,))
    discriminator_output = model(disc_input)

    # concatenate the discriminator output with the label
    cgan_output = Concatenate()([discriminator_output, disc_condition])

    # define the discriminator model with two inputs
    model = Model(inputs=[disc_input, disc_condition], outputs=[cgan_output])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

def generate_real_samples(data, labels, n_samples):
    # Выбор случайных образцов
    idx = np.random.randint(0, data.shape[0], n_samples)
    # Извлечение данных и меток
    X, y = data[idx], labels[idx]
    # Генерация случайных значений просадки
    max_drawdown = np.random.uniform(0, 1, n_samples)
    # Вычисление прибыли с учетом просадки
    y_with_drawdown = (1 - max_drawdown)[:, None] * y
    # Возвращение реальных образцов, меток и значений просадки
    return X, y_with_drawdown, max_drawdown

# Сборка модели cGAN
def build_cgan(generator, discriminator, latent_dim, n_outputs=3):
    discriminator.trainable = False
    gen_input, gen_condition = generator.input
    gen_output = generator.output
    cgan_output = discriminator([gen_output, gen_condition])

    # добавляем дополнительный входной слой для вектора условий
    condition_input = Input(shape=(n_outputs,))
    # добавляем слой для повторения вектора условий
    condition_layer = RepeatVector(latent_dim)(condition_input)
    # объединяем повторенный вектор условий с выходом генератора
    combined_input = Concatenate()([gen_output, condition_layer])
    cgan_output = discriminator(combined_input)

    # определяем модель cGAN с двумя входами (шум, условия) и одним выходом (вероятность реалистичности данных)
    model = Model(inputs=[gen_input, gen_condition, condition_input], outputs=[cgan_output])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=generator_loss, optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return model

#Определение функции потерь генератора
def generator_loss(y_true, y_pred):
    # Вычисление прибыли
    profit = K.mean(y_pred)

    # Вычисление просадки
    max_values = K.maximum(y_true, y_pred)
    drawdowns = 1 - K.abs(y_true - max_values) / max_values
    drawdown = K.mean(drawdowns)

    # Общая функция потерь - максимизация прибыли и минимизация просадки
    loss = -profit + drawdown
    return loss

# Определение размерности скрытого пространства
latent_dim = 100 # примерное значение, можно изменить в зависимости от требуемой точности
n_outputs = 3 # примерное значение, соответствующее количеству условий

# Создание экземпляров моделей
generator = build_generator(latent_dim, n_outputs)
discriminator = build_discriminator()
cgan = build_cgan(generator, discriminator, latent_dim, n_outputs)

# Обучение модели cGAN
def train_cgan(generator, discriminator, cgan, X_train, y_train, latent_dim, n_epochs=5000, n_batch=128):
    # Указание меток классов для дискриминатора
    fake_labels = np.zeros((n_batch, 1))
    real_labels = np.ones((n_batch, 1))

    # Цикл обучения модели
    for epoch in range(n_epochs):
        # Обучение дискриминатора
        # Выбор случайных реальных образцов
        X_real, y_real, _ = generate_real_samples(X_train, y_train, n_batch)
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

        # Вычисление прибыли и просадки на этой эпохе
        z_input = np.random.randn(X_train.shape[0], latent_dim)     #вместо X_train.shape[0] использовать n_batch ?
        generated_signals = generator.predict([z_input, y_train])
        profit = np.mean(generated_signals)
        max_values = np.maximum(y_train, generated_signals)
        drawdowns = 1 - np.abs(y_train - max_values) / max_values
        drawdown = np.mean(drawdowns)

        # Вывод прибыли и просадки
        print(
            f"Epoch: {epoch + 1}/{n_epochs}, profit={profit:.4f}, drawdown={drawdown:.4f}, d_loss={d_loss}, g_loss={g_loss}")


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