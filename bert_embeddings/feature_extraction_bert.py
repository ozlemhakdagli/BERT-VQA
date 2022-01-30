import argparse
import json
import operator
import pickle
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
import plac
import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import utils
from tqdm import tqdm

from utilities.utils import load_configurations, process_sentence

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

configs = load_configurations()

parser = argparse.ArgumentParser()
parser.add_argument("--vqa_datasets", type=str, default=configs['vqa_dataset_path'],
                    help="main VQA files data path")

parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--bert_tf_hub", type=str, default=configs['bert_tf_hub_contextualized'],
                    help="When used with imported config parameter from yaml file it gives the path of the "
                         "BERT tensorflow-hub model. It is also the type of the transformer,"
                         "This will also be used to label resulting graphs and other files.")

parser.add_argument("--max_answer", type=str, default=configs['max_answer'],
                    help="top n answer selection usually preferred 1000")

parser.add_argument("--max_len_sequences", type=str, default=configs['max_length'],
                    help="Maximum length threshold for a sequence")
args = parser.parse_args()


def get_all_answer(answers_list):
    """
    Joins a list of answers by ";" to a single string for each question. A string in the format "ans1;ans2;...;ans10"
    :param answers_list: answer list per question
    :return: ';' joined answers
    """

    answers = []
    for i in range(len(answers_list)):
        answers.append(answers_list[i]['answer'])
    return ';'.join(answers)


def process_question_annotation(subset):
    """
    Creates a json file from raw VQA2 dataset
    :param subset: subset information train2014 or val2014
    :return: processed raw data
    """
    ques = json.load(open(args.vqa_datasets + f'v2_OpenEnded_mscoco_{subset}_questions.json', 'r'))
    anno = json.load(open(args.vqa_datasets + f'v2_mscoco_{subset}_annotations.json', 'r'))
    imdir = '%s/COCO_%s_%012d.jpg'  ## COCO_train2014_000000291417.jpg
    data = []

    print("\nExtraction from raw data started for " + subset)
    for i in tqdm(range(len(anno['annotations']))):
        image_path = imdir % (subset, subset, anno['annotations'][i]['image_id'])
        image_id = subset + '_' + str(anno['annotations'][i]['image_id']).zfill(6)
        answer = anno['annotations'][i]['multiple_choice_answer']
        question = ques['questions'][i]['question']

        answer_list = get_all_answer(anno['annotations'][i]['answers'])

        data.append({'img_path': image_path,
                     'image_id': image_id,
                     'question': question,
                     'answer': answer,
                     'answer_list': answer_list})

    return data
    # save to disk
    # json.dump(data, open(f'/content/drive/My Drive/Capstone/VQA 2/data/vqa_raw_{subset}.json', 'w'))


def filter_top_answers_data(images_list, img_id_list, questions_list, answer_list, answers_list, top_answer_size):
    """
    Filters the dataset to top 1000 answers and cleans the question information.

    :param images_list: image path list
    :param img_id_list: image id list
    :param questions_list: question data
    :param answer_list: answers data
    :param answers_list: top 10 answer per question
    :param top_answer_size: top answers size to  filter the data
    :return: filtered and cleaned data
    """
    answer_freq = defaultdict(int)

    for answer in answer_list:
        answer_freq[answer] += 1

    sorted_freq = sorted(answer_freq.items(), key=operator.itemgetter(1), reverse=True)[0: top_answer_size]
    # sorted_freq = dict(sorted(answer_freq.items(), key=lambda item: item[1], reverse=True))

    top_answers, top_freq = zip(*sorted_freq)

    filtered_img_list = []
    filtered_img_id_list = []
    filtered_question_list = []
    filtered_answer_list = []
    filtered_answers_list = []

    print("\nFiltering from raw data started")
    for image, image_id, question, answer, answers in tqdm(zip(images_list, img_id_list, questions_list, answer_list,
                                                               answers_list)):
        if answer in top_answers:
            filtered_img_list.append(image)
            filtered_img_id_list.append(image_id)
            filtered_question_list.append(question)
            filtered_answer_list.append(answer)
            filtered_answers_list.append(answers)

    print('Data size reduced by',
          np.round(((len(questions_list) - len(filtered_question_list)) / len(questions_list)) * 100, 2), '%')

    """ Cleans the filtered question sentences """
    filtered_question_list = (pd.Series(filtered_question_list).apply(process_sentence)).to_list()

    return filtered_img_list, filtered_img_id_list, filtered_question_list, filtered_answer_list, filtered_answers_list


def convert_examples_to_features(texts, max_length, tokenizer):
    """
    Bert requires special pre-processing before feeding information to BERT model. Each text must be converted in to
    token_id, pad_mask and segment_ids. Bert takes three inputs as described and converts text in to contextualized
    word embeddings.
    :param texts: opinion sentence.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :param tokenizer: Bert tokenizer object.
    :return: token_ids, masks_ids, segments_ids.
    """

    all_tokens = []
    all_masks = []
    all_segments = []

    for text in tqdm(texts):
        text = tokenizer.tokenize(text)

        text = text[:max_length - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_length - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_length

        all_tokens.append(tf.convert_to_tensor(tokens))
        all_masks.append(tf.convert_to_tensor(pad_masks))
        all_segments.append(tf.convert_to_tensor(segment_ids))

    return all_tokens, all_masks, all_segments


def get_answers_matrix(answers, encoder):
    """
    One-hot-encodes the answers.
    :param answers: list of answer
    :param encoder: a scikit-learn LabelEncoder object
    :return: A numpy array of shape (# of answers, # of class)
    """
    y = encoder.transform(answers)  # string to numerical class
    nb_classes = encoder.classes_.shape[0]
    Y = utils.to_categorical(y, nb_classes)

    print("label encoding completed")

    return Y


def load_img_features(filtered_img_list):
    img_features = []
    for img_id in filtered_img_list:
        img_features.append(np.load(file=args.processed_path + 'features_xception/' + img_id + '.npy'))

    return img_features


def main():
    max_answers = args.max_answer
    max_sequence_length = args.max_len_sequences
    TRAIN_INDEX = None
    VAL_INDEX = None

    """ Extracts information from json files and """
    data_train = process_question_annotation('train2014')
    data_val = process_question_annotation('val2014')
    full_data = data_train + data_val

    """ Filter dataset to top 1000 answers """
    filtered_img_list, filtered_img_id_list, filtered_question_list, filtered_answer_list, filtered_answers_list = \
        filter_top_answers_data(
            list(map(itemgetter('img_path'), full_data)),
            list(map(itemgetter('image_id'), full_data)),
            list(map(itemgetter('question'), full_data)),
            list(map(itemgetter('answer'), full_data)),
            list(map(itemgetter('answer_list'), full_data)),
            max_answers)

    with open(args.processed_path + 'vqa_processed_data.json', 'wb') as f:
        pickle.dump((filtered_img_list, filtered_img_id_list, filtered_question_list,
                     filtered_answer_list, filtered_answers_list), f)

    """ Convert examples to ids using spacy """
    bert_layer = hub.KerasLayer(handle=args.bert_tf_hub, trainable=False)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    tokens, masks, segments = convert_examples_to_features(texts=filtered_question_list,
                                                           max_length=max_sequence_length,
                                                           tokenizer=tokenizer)

    """ Encode Answers to labels """
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(filtered_answer_list)
    answer_matrix = get_answers_matrix(filtered_answer_list, label_encoder)

    with open(args.processed_path + 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    """read image features from file"""
    img_features = load_img_features(filtered_img_list=filtered_img_id_list)

    """ split the data 90-10 for train and validation"""
    shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    for train_index, val_index in shuffle_split.split(filtered_img_id_list, filtered_answer_list):
        TRAIN_INDEX = train_index
        VAL_INDEX = val_index

    image_list_tr, image_list_vl = np.array(img_features)[TRAIN_INDEX.astype(int)], \
                                   np.array(img_features)[VAL_INDEX.astype(int)]

    question_tr_token, question_vl_token = np.array(tokens)[TRAIN_INDEX.astype(int)], \
                                           np.array(tokens)[VAL_INDEX.astype(int)]

    question_tr_mask, question_vl_mask = np.array(masks)[TRAIN_INDEX.astype(int)], \
                                         np.array(masks)[VAL_INDEX.astype(int)]

    question_tr_segments, question_vl_segments = np.array(segments)[TRAIN_INDEX.astype(int)], \
                                                 np.array(segments)[VAL_INDEX.astype(int)]

    answer_tr, answer_vl = np.array(answer_matrix)[TRAIN_INDEX.astype(int)], \
                           np.array(answer_matrix)[VAL_INDEX.astype(int)]

    with open(args.processed_path + 'train_data.pkl', 'wb') as f:
        pickle.dump((image_list_tr, question_tr_token,
                     question_tr_mask, question_tr_segments, answer_tr), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.processed_path + 'val_data.pkl', 'wb') as f:
        pickle.dump((image_list_vl, question_vl_token,
                     question_vl_mask, question_vl_segments, answer_vl), f, protocol=pickle.HIGHEST_PROTOCOL)

    """ Test dataset creation """
    answers_list_test = np.array(filtered_answers_list)[VAL_INDEX.astype(int)]

    with open(args.processed_path + 'test_data.pkl', 'wb') as f:
        pickle.dump((image_list_vl, question_vl_token,
                     question_vl_mask, question_vl_segments, answers_list_test), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    plac.call(main)
