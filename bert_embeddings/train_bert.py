import argparse
import os
import pickle

import matplotlib.pyplot as plt
import plac
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

import tensorflow_hub as hub
from model_bert import build_model
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

parser.add_argument("--learning_rate", type=float, default=configs['learning_rate'],
                    help="learning_rate")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")
args = parser.parse_args()


def decay_schedule(epoch, lr):
    if epoch == 0:
        print("epoch=", epoch, "and learning rate=", lr)
        return lr
    elif epoch % 50 == 0:
        lr = lr / 2
        print("epoch=", epoch, "and learning rate=", lr)
        return lr
    else:
        return lr


def main():
    BATCH_SIZE = args.img_batch_size
    EPOCHS = args.epochs
    max_seq_len = args.max_len_sequences
    learning_rate = args.learning_rate

    with open(args.processed_path + 'train_data.pkl', 'rb') as f:
        image_list_tr, question_tr_token, question_tr_mask, question_tr_segments, train_answer_label = pickle.load(f)

    with open(args.processed_path + 'val_data.pkl', 'rb') as f:
        image_list_vl, question_vl_token, question_vl_mask, question_vl_segments, val_answer_label = pickle.load(f)

    bert_layer = hub.KerasLayer(handle=args.bert_tf_hub, trainable=False)

    lr_scheduler = LearningRateScheduler(decay_schedule)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,
                       patience=3, restore_best_weights=True)

    model = build_model(max_seq_len=max_seq_len, learning_rate=learning_rate, bert_layer=bert_layer)

    model.summary()

    history = model.fit([image_list_tr, question_tr_token, question_tr_mask, question_tr_segments],
                        train_answer_label,
                        validation_data=([image_list_vl, question_vl_token, question_vl_mask, question_vl_segments],
                                         val_answer_label),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        callbacks=[lr_scheduler, es])

    print("Saving to", args.results_path)
    model.save(args.results_path + "model", save_format="h5")

    with open(args.results_path + 'result_history.txt', 'w') as file:
        file.write(str(history.history))

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.results_path + 'accuracy.png', bbox_inches='tight')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args.results_path + 'loss.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plac.call(main)
