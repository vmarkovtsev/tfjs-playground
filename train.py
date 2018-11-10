import csv

from keras import layers, optimizers, Model
import numpy


def train(X, Y):
    hyperp = {
        "learning_rate": 0.001,
        "lstm_units": 192,
    }

    input = layers.Input([140], dtype="uint8")
    head = layers.Embedding(input_dim=219, output_dim=219, embeddings_initializer="identity",
                            trainable=False)(input)
    for _ in range(2):
        head = layers.LSTM(units=hyperp["lstm_units"],
                           unit_forget_bias=True, return_sequences=True)(head)
    head = layers.TimeDistributed(layer=layers.Dense(units=219, activation='softmax'))(head)
    model = Model(inputs=input, outputs=head)
    print("compile")
    model.compile(optimizer=optimizers.RMSprop(lr=hyperp["learning_rate"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    model.fit(X, Y, validation_split=0.2, batch_size=100, shuffle=True, epochs=2)


def load():
    X = []
    with open("Sentiment Analysis Dataset.csv") as fin:
        fin.readline()
        for ir, row in enumerate(csv.reader(fin)):
            if ir > 100000:
                break
            arr = numpy.zeros(140, dtype="uint8")
            for i, b in enumerate(row[-1].strip().encode("utf-8")):
                if i >= 140:
                    break
                arr[i] = b - 8
            X.append(arr)
    print("loaded")
    X = numpy.array(X)
    return X, numpy.expand_dims(X, -1)


if __name__ == "__main__":
    train(*load())
