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
