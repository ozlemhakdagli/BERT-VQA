import argparse
import json
import pickle
import time

import joblib
import numpy as np
import plac
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tqdm import tqdm

from lookup_based.lstm_model import build_model
from utilities.utils import load_configurations, set_memory_growth

set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()

parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--img_batch_size", type=int, default=configs['img_batch_size'],
                    help="image feature extraction batch size")

parser.add_argument("--buffer_size", type=int, default=configs['buffer_size'],
                    help="train test buffer size")

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


def load_datasets(image_list, questions, answers, BUFFER_SIZE, BATCH_SIZE):
    dataset_load = tf.data.Dataset.from_tensor_slices((image_list, questions, answers))

    # Use map to load the numpy files in parallel
    dataset_load = dataset_load.map(lambda item1, item2, item3: tf.numpy_function(
        map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle and batch
    dataset_load = dataset_load.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset_load = dataset_load.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset_load


# loading the numpy files
def map_func(img_name, ques, ans):
    img_tensor = np.load(
        file=args.processed_path + 'features_vgg19/' + str(img_name.decode('utf-8')) + '.npy')
    return img_tensor, ques, ans


# @tf.function
def train_step(model, img, ques, ans, optimizer, loss_object, train_loss, train_score):
    with tf.GradientTape() as tape:
        # forward pass
        predictions = model([img, ques], training=True)
        loss = loss_object(ans, predictions)

    # backward pass
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # record results
    train_loss(loss)
    train_score(ans, predictions)

    # all gradients
    grads_ = list(zip(grads, model.trainable_variables))
    return grads_


def test_step(model, img, ques, ans, loss_object, val_loss, val_score):
    predictions = model([img, ques])
    loss = loss_object(ans, predictions)

    # record results
    val_loss(loss)
    val_score(ans, predictions)


def main():
    SAVE_CKPT_FREQ = 5
    BATCH_SIZE = args.img_batch_size
    BUFFER_SIZE = args.buffer_size
    EPOCHS = args.epochs
    max_answers = args.max_answer
    max_seq_len = args.max_len_sequences
    dim_hidden = args.hidden_dimension
    dim_attention = args.attention_dimension
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    reg_value = args.reg_value

    with open(configs[args.embedding_type] + "weights.pkl", "rb") as f:
        vectors = pickle.load(f)
    print("word weights are loaded.")

    with open(args.processed_path + 'train_data.pkl', 'rb') as f:
        train_img_id, train_ques_embedding, train_answer_label = joblib.load(f)

    with open(args.processed_path + 'val_data.pkl', 'rb') as f:
        val_img_id, val_ques_embedding, val_answer_label = joblib.load(f)

    dataset_tr = load_datasets(image_list=train_img_id,
                               questions=train_ques_embedding,
                               answers=train_answer_label,
                               BUFFER_SIZE=BUFFER_SIZE,
                               BATCH_SIZE=BATCH_SIZE)

    dataset_vl = load_datasets(image_list=val_img_id,
                               questions=val_ques_embedding,
                               answers=val_answer_label,
                               BUFFER_SIZE=BUFFER_SIZE,
                               BATCH_SIZE=BATCH_SIZE)

    model = build_model(max_answers=max_answers, max_seq_len=max_seq_len, vectors=vectors,
                        dim_hidden=dim_hidden, dim_attention=dim_attention,
                        dropout_rate=dropout_rate, learning_rate=learning_rate)

    model.summary()

    steps_per_epoch = int(np.ceil(len(train_img_id) / BATCH_SIZE))
    boundaries = [100 * steps_per_epoch, 170 * steps_per_epoch]
    values = [learning_rate, learning_rate / 4, learning_rate / 8]

    # we reduce the l_rate after 50th epoch (from 1e-4 to 1e-5)
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='auto')

    train_score = F1Score(num_classes=max_answers, average='micro', name='train_score')
    val_score = F1Score(num_classes=max_answers, average='micro', name='val_score')

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    checkpoint_directory = args.results_path + "training_checkpoints/" + str(learning_rate) + "_" + str(dim_hidden)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=15)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        START_EPOCH = int(manager.latest_checkpoint.split('-')[-1]) * SAVE_CKPT_FREQ
        print("Resume training from epoch: {}".format(START_EPOCH))
    else:
        print("Initializing from scratch")
        START_EPOCH = 0

    print("train starts now.")
    epoch_list = []
    val_score_list = []
    train_score_list = []
    val_loss_list = []
    train_loss_list = []

    for epoch in range(START_EPOCH, EPOCHS):
        start = time.time()
        for img, ques, ans in tqdm(dataset_tr):
            train_step(model=model, img=img, ques=ques,
                       ans=ans, optimizer=optimizer, loss_object=loss_object,
                       train_loss=train_loss, train_score=train_score)

        for img, ques, ans in tqdm(dataset_vl):
            test_step(model=model, img=img, ques=ques,
                      ans=ans, loss_object=loss_object,
                      val_loss=val_loss, val_score=val_score)

        epoch_list.append(epoch + 1)
        train_score_list.append(float(train_score.result()))
        train_loss_list.append(float(train_loss.result()))
        val_score_list.append(float(val_score.result()))
        val_loss_list.append(float(val_loss.result()))

        template = 'Epoch {}, loss: {:.4f}, f1_score: {:.4f}, val loss: {:.4f}, val f1_score: {:.4f}, time: {:.0f} sec'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_score.result(),
                              val_loss.result(),
                              val_score.result(),
                              (time.time() - start)))

        # Reset metrics every epoch
        train_loss.reset_states()
        train_score.reset_states()
        val_loss.reset_states()
        val_score.reset_states()

        ckpt.step.assign_add(1)
        if int(ckpt.step) % SAVE_CKPT_FREQ == 0:
            # dataset_eval = load_datasets_eval(image_list=images_val, questions=question_data_val,
            #                                   BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)
            #
            # y = predict_answers(dataset=dataset_eval, model=model, label_encoder=label_encoder)
            # val_score = model_metric(y, answers_val)
            # print(val_score)

            # save checkpoint every SAVE_CKPT_FREQ step
            path = args.results_path + "training_checkpoints/final_model/" + str(int(ckpt.step)) + "_model.h5"
            model.save(filepath=path, save_format='h5')

            manager.save()
            print('Saved checkpoint.')
            track_dict = {'epoch': epoch_list, 'train_loss': train_loss_list, 'train_score': train_score_list,
                          'val_loss': val_loss_list, 'val_score': val_score_list}
            with open(args.results_path + 'training_checkpoints/' + str(epoch)
                      + '_scores.txt', 'w') as convert_file:
                convert_file.write(json.dumps(track_dict))


if __name__ == "__main__":
    plac.call(main)
