import argparse
import os
import pickle

import keras_tuner as kt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Multiply, Bidirectional
from tensorflow.keras.optimizers import Adam

from utilities.utils import load_configurations, set_memory_growth

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--img_batch_size", type=int, default=configs['img_batch_size'],
                    help="image feature extraction batch size")

parser.add_argument("--bert_tf_hub", type=str, default=configs['bert_tf_hub_contextualized'],
                    help="When used with imported config parameter from yaml file it gives the path of the "
                         "BERT tensorflow-hub model. It is also the type of the transformer,"
                         "This will also be used to label resulting graphs and other files.")

parser.add_argument("--max_len_sequences", type=int, default=configs['max_length'],
                    help="Maximum length threshold for a sequence ")

parser.add_argument("--epochs", type=int, default=configs['epochs'],
                    help="epochs")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")
args = parser.parse_args()

with open(args.processed_path + 'train_data.pkl', 'rb') as f:
    image_list_tr, question_tr_token, question_tr_mask, question_tr_segments, train_answer_label = pickle.load(f)

with open(args.processed_path + 'val_data.pkl', 'rb') as f:
    image_list_vl, question_vl_token, question_vl_mask, question_vl_segments, val_answer_label = pickle.load(f)

bert_layer = hub.KerasLayer(handle=args.bert_tf_hub, trainable=False)


def build_model(hp):
    dense_units = hp.Int("units", min_value=512, max_value=1536, step=64)
    bidir_units = hp.Int("bi_units", min_value=512, max_value=1024, step=64)
    drop_out = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    activation = hp.Choice("activation", ["relu", "tanh", "gelu", "swish", "selu", "elu"])
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

    image_input = Input(shape=(2048,), name='Image_Input')
    image_feat = Dense(dense_units, activation=activation, name='image_dense')(image_input)

    word_ids = Input(shape=(args.max_len_sequences,), dtype=tf.int32, name='inputs/0')
    mask = Input(shape=(args.max_len_sequences,), dtype=tf.int32, name='inputs/1')
    segment_ids = Input(shape=(args.max_len_sequences,), dtype=tf.int32, name='inputs/2')
    _, ques_feat_w = bert_layer([word_ids, mask, segment_ids])

    LSTM1 = Bidirectional(LSTM(bidir_units, return_sequences=True))(ques_feat_w)
    LSTM1 = Dropout(drop_out)(LSTM1)
    LSTM1 = Bidirectional(LSTM(bidir_units, return_sequences=False))(LSTM1)
    LSTM1 = Dropout(drop_out)(LSTM1)
    Ques_D = Dense(dense_units, activation=activation)(LSTM1)
    Ques_D = Dropout(drop_out)(Ques_D)

    fv = Multiply(name="multiply")([image_feat, Ques_D])

    F = Dense(dense_units, activation=activation, name="fc_1")(fv)
    F = Dropout(drop_out, name="fc1_drop_1")(F)
    F = Dense(1000, activation=activation, name="fc2")(F)
    F = Dropout(drop_out, name="fc2_drop")(F)
    output = Dense(1000, activation='sigmoid')(F)

    model = Model(inputs=[image_input, word_ids, mask, segment_ids], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


hp = kt.HyperParameters()

tuner = kt.Hyperband(hypermodel=build_model,
                     objective='val_loss',
                     max_epochs=25,
                     factor=3,
                     hyperband_iterations=2,
                     directory=args.results_path,
                     project_name='hyper_search')

stop_early = EarlyStopping(monitor='val_loss', patience=5)

tuner.search([image_list_tr, question_tr_token, question_tr_mask, question_tr_segments],
             train_answer_label,
             validation_data=([image_list_vl, question_vl_token, question_vl_mask, question_vl_segments],
                              val_answer_label),
             epochs=args.epochs,
             batch_size=args.img_batch_size,
             callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("units", best_hps.get("units"))
print("bi_units", best_hps.get("bi_units"))
print("dropout", best_hps.get("dropout"))
print("activation", best_hps.get("activation"))
print("learning_rate", best_hps.get("learning_rate"))
