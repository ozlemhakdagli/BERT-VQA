from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Dense, Dropout, Input, Attention, BatchNormalization, MultiHeadAttention, \
    Activation, LSTM, Multiply, Bidirectional, GRU, LayerNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def build_model(max_seq_len, learning_rate, bert_layer):
    image_input = Input(shape=(2048,), name='Image_Input')
    image_feat = Dense(1024, activation="tanh", name='image_dense')(image_input)

    word_ids = Input(shape=(max_seq_len,), dtype=tf.int32, name='inputs/0')
    mask = Input(shape=(max_seq_len,), dtype=tf.int32, name='inputs/1')
    segment_ids = Input(shape=(max_seq_len,), dtype=tf.int32, name='inputs/2')
    _, ques_feat_w = bert_layer([word_ids, mask, segment_ids])

    LSTM1 = Bidirectional(LSTM(512, return_sequences=True))(ques_feat_w)
    LSTM1 = Dropout(0.5)(LSTM1)
    LSTM1 = Bidirectional(LSTM(512, return_sequences=False))(LSTM1)
    LSTM1 = Dropout(0.5)(LSTM1)
    Ques_D = Dense(1024, activation='tanh')(LSTM1)
    Ques_D = Dropout(0.5)(Ques_D)

    fv = Multiply(name="multiply")([image_feat, Ques_D])

    F = Dense(1024, activation='tanh', name="fc_1")(fv)
    F = Dropout(0.5, name="fc1_drop_1")(F)
    F = Dense(1000, activation='tanh', name="fc2")(F)
    F = Dropout(0.5, name="fc2_drop")(F)
    output = Dense(1000, activation='sigmoid')(F)

    model = Model(inputs=[image_input, word_ids, mask, segment_ids], outputs=output)

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
