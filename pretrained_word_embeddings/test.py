import argparse
import os
import pickle

import numpy as np
import plac
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

from utilities.utils import load_configurations, set_memory_growth

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
set_memory_growth()
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")
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


# def chunks(img, token, n):
#     for i in tqdm(range(0, len(img), n)):
#         yield img[i:i + n], token[i, i + n]


# for i in range(0, len(image_weights), 512):
# [image_weights[i:i + n]
# for img, ques in tqdm(zip(image_weights, question_tokens)):
def predict_answers(image_weights, question_tokens, model, labelencoder):
    """
    Prediction function
    """

    predictions = []
    for img, token in tqdm(zip(image_weights, question_tokens)):
        preds = model.predict([img, token])
        predictions.extend(preds)

    y_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
    y_predict = (labelencoder.inverse_transform(y_classes))
    return y_predict


def main():
    with open(args.processed_path + 'test_data.pkl', 'rb') as f:
        image_list_test, question_list_test, answers_list_test = pickle.load(f)

    with open(args.processed_path + 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("Loading NLI model")
    model = load_model(args.results_path + "model_lstm")
    print("NLI model loaded")

    pred_ready_img = [img.reshape(1, -1) for img in image_list_test]
    pred_ready_tokens = [ques.reshape(1, -1) for ques in question_list_test]

    pred_ready_img = pred_ready_img[0:30000]
    pred_ready_tokens = pred_ready_tokens[0:30000]

    predictions = predict_answers(pred_ready_img, pred_ready_tokens, model, label_encoder)
    score = model_metric(predictions, answers_list_test)

    print("total accuracy of the model=", score)


if __name__ == "__main__":
    plac.call(main)
