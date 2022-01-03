from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Dense, Dropout, Input, Attention, BatchNormalization, MultiHeadAttention, \
    Activation, LSTM, Multiply, Bidirectional, GRU, LayerNormalization
from tensorflow.keras.optimizers import Adam


def build_model(max_seq_len, vectors, learning_rate):
    image_input = Input(shape=(2048,), name='Image_Input')
    image_feat = Dense(1024, activation="tanh", name='image_dense')(image_input)

    ques_input = Input(shape=(max_seq_len,), dtype='int32', name='Question_Input')
    ques_feat_w = embedding_layer(vectors=vectors, max_length=max_seq_len)(ques_input)
    LSTM1 = Bidirectional(GRU(512, return_sequences=True))(ques_feat_w)
    LSTM1 = Dropout(0.5)(LSTM1)
    LSTM1 = Bidirectional(GRU(512, return_sequences=False))(LSTM1)
    LSTM1 = Dropout(0.5)(LSTM1)
    Ques_D = Dense(1024, activation='tanh')(LSTM1)
    Ques_D = Dropout(0.5)(Ques_D)

    fv = Multiply(name="multiply")([image_feat, Ques_D])

    F = Dense(1024, activation='tanh', name="fc_1")(fv)
    F = Dropout(0.5, name="fc1_drop_1")(F)
    F = Dense(1000, activation='tanh', name="fc2")(F)
    F = Dropout(0.5, name="fc2_drop")(F)
    output = Dense(1000, activation='sigmoid')(F)

    model = Model(inputs=[image_input, ques_input], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def embedding_layer(vectors, max_length):
    return models.Sequential(
        [
            layers.Embedding(
                vectors.shape[0],
                vectors.shape[1],
                input_length=max_length,
                weights=[vectors],
                trainable=False,
            )
        ]
    )
