import tensorflow as tf
from tensorflow.keras import layers, Model, models
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.layers import GRU

from old_code.model_old_layers import ContextVector, PhraseLevelFeatures, AttentionMaps


def build_model(max_answers, max_seq_len, vectors, dim_hidden, dim_attention, dropout_rate, reg_value):
    """
    Defines the Keras model.
    Arguments
    ----------
    max_answers : Number of output targets of the model.
    max_seq_len : Length of input sequences
    dim_hidden       : Hidden dimension
    dim_attention       : Hidden attention dimension
    dropout_rate      : Dropout rate
    reg_value   : Regularization value
    vectors     : vector file
    Returns
    ----------
    Returns the Keras model.
    """
    # inputs
    image_input = Input(shape=(49, 512,), name='Image_Input')
    image_feat = Dropout(dropout_rate)(image_input)

    ques_input = Input(shape=(max_seq_len,), name='Question_Input')
    ques_feat_w = word_embedding_layer(vectors=vectors, max_length=max_seq_len)(ques_input)
    ques_feat_w = GRU(dim_hidden, return_sequences=True, input_shape=(None, max_seq_len, dim_hidden))(ques_feat_w)
    ques_feat_w = Dropout(dropout_rate)(ques_feat_w)

    # word level
    Hv_w, Hq_w = AttentionMaps(dim_attention, reg_value, name='AttentionMaps_Word')(image_feat, ques_feat_w)  # noqa
    v_w, q_w = ContextVector(reg_value, name='ContextVector_Word')(image_feat, ques_feat_w, Hv_w, Hq_w)  # noqa
    feat_w = tf.add(v_w, q_w)
    h_w = Dense(dim_hidden, activation='tanh', name='h_w_Dense',
                kernel_regularizer=tf.keras.regularizers.l2(reg_value))(feat_w)

    # phrase level
    ques_feat_p = PhraseLevelFeatures(dim_hidden, name='PhraseLevelFeatures')(ques_feat_w)  # noqa

    Hv_p, Hq_p = AttentionMaps(dim_attention, reg_value, name='AttentionMaps_Phrase')(image_feat, ques_feat_p)  # noqa
    v_p, q_p = ContextVector(reg_value, name='ContextVector_Phrase')(image_feat, ques_feat_p, Hv_p, Hq_p)  # noqa
    feat_p = concatenate([tf.add(v_p, q_p), h_w], -1)
    h_p = Dense(dim_hidden, activation='tanh', name='h_p_Dense',
                kernel_regularizer=tf.keras.regularizers.l2(reg_value))(feat_p)

    # sentence level
    ques_feat_s = GRU(dim_hidden, return_sequences=True, input_shape=(None, max_seq_len, dim_hidden))(ques_feat_p)

    Hv_s, Hq_s = AttentionMaps(dim_attention, reg_value, name='AttentionMaps_Sent')(image_feat, ques_feat_s)  # noqa
    v_s, q_s = ContextVector(reg_value, name='ContextVector_Sent')(image_feat, ques_feat_p, Hv_s, Hq_s)  # noqa
    feat_s = concatenate([tf.add(v_s, q_s), h_p], -1)
    h_s = Dense(2 * dim_hidden, activation='tanh', name='h_s_Dense',
                kernel_regularizer=tf.keras.regularizers.l2(reg_value))(feat_s)

    z = Dense(2 * dim_hidden, activation='tanh', name='z_Dense',
              kernel_regularizer=tf.keras.regularizers.l2(reg_value))(h_s)
    z = Dropout(dropout_rate, seed=16)(z)

    # result
    result = Dense(max_answers, activation='softmax')(z)

    model = Model(inputs=[image_input, ques_input], outputs=result)

    return model


def word_embedding_layer(vectors, max_length):
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
