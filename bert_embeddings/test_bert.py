import argparse
import os
import pickle

import numpy as np
import plac
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import plac
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

import tensorflow_hub as hub
from model_bert import build_model
from utilities.utils import load_configurations, set_memory_growth
from utilities.utils import load_configurations, set_memory_growth

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")

parser.add_argument("--bert_tf_hub", type=str, default=configs['bert_tf_hub_contextualized'],
                    help="When used with imported config parameter from yaml file it gives the path of the "
                         "BERT tensorflow-hub model. It is also the type of the transformer,"
                         "This will also be used to label resulting graphs and other files.")
args = parser.parse_args()


def model_metric(predictions, truths):
    """
    Measures the accuracy of the predictions

    Input:
        predictions : predictions
        truths      : ground truth answers

    Returns:
        Accuracy measure of the model
    """

    total = 0
    correct_val = 0.0

    for prediction, truth in zip(predictions, truths):

        temp_count = 0
        total += 1

        for _truth in truth.split(';'):
            if prediction == _truth:
                temp_count += 1

        # accuracy = min((# humans that provided that answer/3) , 1)
        if temp_count > 2:
            correct_val += 1
        else:
            correct_val += float(temp_count) / 3

    return (correct_val / total) * 100


def predict_answers(image_weights, question_tokens, question_mask, question_segment, model, labelencoder):
    """
    Prediction function
    """

    predictions = []
    batch_size = 512
    # for i in range(0, len(question_tokens), batch_size):
    #     img = np.asarray(image_weights[i:i + batch_size])
    #     token = np.asarray(question_tokens[i:i + batch_size])
    #     mask = np.asarray(question_mask[i:i + batch_size])
    #     segment = np.asarray(question_segment[i:i + batch_size])
    # print("prediction begins")
    # preds = model.predict([image_weights, question_tokens, question_mask, question_segment],
    #                       batch_size=1024,
    #                       steps=860,
    #                       verbose=1)
    # print(preds)
    # print("done")
    # predictions.extend(preds)

    for img, token, mask, segment in tqdm(zip(image_weights, question_tokens, question_mask, question_segment)):
        preds = model.predict([img, token, mask, segment])
        predictions.extend(preds)

    y_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
    y_predict = (labelencoder.inverse_transform(y_classes))
    return y_predict


def main():
    with open(args.processed_path + 'test_data.pkl', 'rb') as f:
        image_list_test, question_token_test, \
        question_mask_test, question_segments_test, answers_list_test = pickle.load(f)

    with open(args.processed_path + 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("Loading NLI model")
    model = load_model(args.results_path + "model", custom_objects={"tf": tf, "KerasLayer": hub.KerasLayer})
    print("NLI model loaded")

    pred_ready_img = [img.reshape(1, -1) for img in image_list_test]
    pred_ready_token = [ques.reshape(1, -1) for ques in question_token_test]
    pred_ready_mask = [ques.reshape(1, -1) for ques in question_mask_test]
    pred_ready_segment = [ques.reshape(1, -1) for ques in question_segments_test]

    print("total test samples", len(pred_ready_img))
    # pred_ready_img = pred_ready_img[0:30000]
    # pred_ready_tokens = pred_ready_tokens[0:30000]

    predictions = predict_answers(pred_ready_img, pred_ready_token, pred_ready_mask,
                                  pred_ready_segment, model, label_encoder)
    score = model_metric(predictions, answers_list_test)

    print("total accuracy of the model=", score)


if __name__ == "__main__":
    plac.call(main)
