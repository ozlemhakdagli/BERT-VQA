import argparse
import json
import operator
import pickle
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
import plac
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import utils
from tqdm import tqdm

from utilities.utils import load_configurations, process_sentence, load_spacy_nlp, get_embeddings

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

parser.add_argument("--max_answer", type=str, default=configs['max_answer'],
                    help="top n answer selection usually preferred 1000")

parser.add_argument("--embedding_type", type=str, default=configs['embedding_type'],
                    help="embedding_type_for_spacy")

parser.add_argument("--num_unk", type=str, default=configs['nr_unk'],
                    help="number of unknown words to be randomly filled")

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


def convert_examples_to_features(nlp, sentences, num_unk, max_length):
    """
    This function takes text as list and converts them to word-ids matrix based on lookup table.
    Extracted word-ids will be converted to vectors in the embedding layer of the training models.
    :param nlp: spacy nlp object
    :param sentences: sentence to be processed.
    :param num_unk: unknown word count that will be filled with norm_random vectors.
    :param max_length: max length of the sentence. longer ones will be pruned, shorter ones will be padded.
    :return: returns word-ids as a list.
    """

    print("\nTotal number of text to be processed:", len(sentences))
    sentences_as_ids = []

    for sent in tqdm(sentences):
        doc = nlp(sent)
        word_ids = []
        for i, token in enumerate(doc):
            if token.has_vector and token.vector_norm == 0:
                continue
            if i > max_length:
                break
            if token.has_vector:
                word_ids.append(token.rank + num_unk + 1)
            else:
                # if we don't have a vector, pick an OOV entry
                word_ids.append(token.rank % num_unk + 1)

        word_id_vec = np.zeros(max_length, dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sentences_as_ids.append(word_id_vec)

    return sentences_as_ids


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
    num_unk = args.num_unk
    embedding_type = args.embedding_type
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
    nlp = load_spacy_nlp(configs=configs, transformer_type=embedding_type)
    token_list = convert_examples_to_features(nlp=nlp, sentences=filtered_question_list,
                                              num_unk=num_unk, max_length=max_sequence_length)

    """ Extract vectors to file """
    vectors = get_embeddings(nlp.vocab, nr_unk=num_unk)
    with open(configs[args.embedding_type] + "weights.pkl", "wb") as f:
        pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

    """ Encode Answers to labels """
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(filtered_answer_list)
    answer_matrix = get_answers_matrix(filtered_answer_list, label_encoder)

    with open(args.processed_path + 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    """read image features from file"""
    img_features = load_img_features(filtered_img_list=filtered_img_id_list)

    """ split the data 65-35 for train and validation"""
    shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    for train_index, val_index in shuffle_split.split(filtered_img_id_list, filtered_answer_list):
        TRAIN_INDEX = train_index
        VAL_INDEX = val_index

    image_list_tr, image_list_vl = np.array(img_features)[TRAIN_INDEX.astype(int)], \
                                   np.array(img_features)[VAL_INDEX.astype(int)]

    question_tr, question_vl = np.array(token_list)[TRAIN_INDEX.astype(int)], \
                               np.array(token_list)[VAL_INDEX.astype(int)]

    answer_tr, answer_vl = np.array(answer_matrix)[TRAIN_INDEX.astype(int)], \
                           np.array(answer_matrix)[VAL_INDEX.astype(int)]

    with open(args.processed_path + 'train_data.pkl', 'wb') as f:
        pickle.dump((image_list_tr, question_tr, answer_tr), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.processed_path + 'val_data.pkl', 'wb') as f:
        pickle.dump((image_list_vl, question_vl, answer_vl), f, protocol=pickle.HIGHEST_PROTOCOL)

    """ Test dataset creation """
    answers_list_test = np.array(filtered_answers_list)[VAL_INDEX.astype(int)]

    with open(args.processed_path + 'test_data.pkl', 'wb') as f:
        pickle.dump((image_list_vl, question_vl, answers_list_test), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    plac.call(main)
