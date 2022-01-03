import argparse
import os
import pickle

import keras_tuner as kt
from tensorflow.keras import layers, Model, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Multiply, Attention, BatchNormalization, \
    MultiHeadAttention, Activation
from tensorflow.keras.optimizers import Adam

from utilities.utils import set_memory_growth, load_configurations

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()

parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--img_batch_size", type=int, default=configs['img_batch_size'],
                    help="image feature extraction batch size")

parser.add_argument("--max_answer", type=int, default=configs['max_answer'],
                    help="max answer size")

parser.add_argument("--max_len_sequences", type=int, default=configs['max_length'],
                    help="Maximum length threshold for a sequence ")

parser.add_argument("--embedding_type", type=str, default=configs['embedding_type'],
                    help="embedding_type_for_spacy")

parser.add_argument("--epochs", type=int, default=configs['epochs'],
                    help="epochs")

parser.add_argument("--hidden_dimension", type=int, default=configs['hidden_dimension'],
                    help="hidden_dimension")

parser.add_argument("--attention_dimension", type=int, default=configs['attention_dimension'],
                    help="attention_dimension")

parser.add_argument("--learning_rate", type=float, default=configs['learning_rate'],
                    help="learning_rate")

parser.add_argument("--dropout_rate", type=float, default=configs['dropout_rate'],
                    help="dropout_rate")

parser.add_argument("--reg_value", type=float, default=configs['reg_value'],
                    help="reg_value")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")
args = parser.parse_args()

with open(configs[args.embedding_type] + "weights.pkl", "rb") as f:
    vectors1 = pickle.load(f)
print("word weights are loaded.")

with open(args.processed_path + 'train_data.pkl', 'rb') as f:
    train_img_id, train_ques_embedding, train_answer_label = pickle.load(f)

with open(args.processed_path + 'val_data.pkl', 'rb') as f:
    val_img_id, val_ques_embedding, val_answer_label = pickle.load(f)
transformer_layer = 6


def build_model(hp):
    dense_units = hp.Int("units", min_value=512, max_value=1536, step=32)
    drop_out = hp.Float("rate", min_value=0.0, max_value=0.5, step=0.1)
    epsilon = hp.Float("epsilon", min_value=1e-9, max_value=1e-2, sampling="log")
    num_head = hp.Int("num_heads", min_value=2, max_value=16, step=2)
    key_dim = hp.Int("key_dim", min_value=16, max_value=256, step=16)
    att_drop = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    activation = hp.Choice("activation", ["relu", "tanh", "gelu", "swish", "selu", "elu"])
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")

    # inputs
    image_input = Input(shape=(2048,), name='Image_Input')
    image_feat = Dense(dense_units, activation=None, name='image_dense')(image_input)
    image_feat = Dropout(drop_out, name="image_drop")(image_feat)
    image_feat = BatchNormalization(name='image_batch_norm', epsilon=epsilon)(image_feat)
    image_feat = Activation(activation, name='image_activation')(image_feat)
    image_feat = Attention(name='image_soft_attention')([image_feat, image_feat])

    ques_input = Input(shape=(args.max_len_sequences,), name='Question_Input')
    ques_feat_w = embedding_layer(vectors=vectors1, max_length=args.max_len_sequences)(ques_input)

    Ques_D = None
    for i, _ in enumerate(range(transformer_layer)):
        attention_output = MultiHeadAttention(num_heads=num_head, key_dim=key_dim, dropout=att_drop,
                                              name='multi_head_attention')(ques_feat_w, ques_feat_w)
        attention_output = Dropout(drop_out, name='multi_head_dropout')(attention_output)

        x2 = layers.Add(name='multi_head_add_tensor_1')([attention_output, ques_feat_w])
        x2 = layers.LayerNormalization(epsilon=epsilon, name='multi_head_layer_norm_1')(x2)

        x3 = Dense(300, activation=activation, name='multi_head_dense')(x2)
        Ques_D = layers.Add(name='multi_head_add_tensor_2')([x3, x2])
        Ques_D = layers.LayerNormalization(epsilon=epsilon, name='multi_head_layer_norm_2')(Ques_D)
    Ques_D = GRU(dense_units, activation=activation, return_sequences=False, name='GRU_Q')(Ques_D)

    fv = Multiply(name="multiply")([image_feat, Ques_D])

    F = Dense(dense_units, activation=activation, name="fc_1")(fv)
    F = Dropout(drop_out, name="fc1_drop_1")(F)
    F = Dense(dense_units, activation=activation, name="fc2")(F)
    F = Dropout(drop_out, name="fc2_drop")(F)
    output = Dense(1000, activation='softmax')(F)

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


hp = kt.HyperParameters()

tuner = kt.Hyperband(hypermodel=build_model,
                     objective='val_loss',
                     max_epochs=30,
                     factor=5,
                     directory=args.results_path,
                     project_name='hyper_search')

stop_early = EarlyStopping(monitor='val_loss', patience=5)

tuner.search([train_img_id, train_ques_embedding],
             train_answer_label,
             validation_data=([val_img_id, val_ques_embedding], val_answer_label),
             epochs=args.epochs,
             batch_size=args.img_batch_size,
             callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("units", best_hps.get("units"))
print("rate", best_hps.get("rate"))
print("epsilon", best_hps.get("epsilon"))
print("num_heads", best_hps.get("num_heads"))
print("key_dim", best_hps.get("key_dim"))
print("dropout", best_hps.get("dropout"))
print("activation", best_hps.get("activation"))
print("learning_rate", best_hps.get("learning_rate"))
