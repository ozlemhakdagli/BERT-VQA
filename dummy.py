import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from utilities.utils import load_configurations

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
configs = load_configurations()
parser = argparse.ArgumentParser()
parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")

parser.add_argument("--buffer_size", type=int, default=configs['buffer_size'],
                    help="train test buffer size")

parser.add_argument("--img_batch_size", type=int, default=configs['img_batch_size'],
                    help="image feature extraction batch size")

parser.add_argument("--results_path", type=str, default=configs['results'],
                    help="results path")
args = parser.parse_args()

# anno = json.load(open("E:\\VQA\\vqa_dataset\\v2_mscoco_val2014_annotations.json"))
# ques = json.load(open("E:\\VQA\\vqa_dataset\\v2_OpenEnded_mscoco_val2014_questions.json"))
#
# # raw_ques = json.load(open("E:\\VQA\\\processed_data\\vqa_raw_train2014.json"))
# print(len(anno['annotations']))
# print(len(ques['questions']))

# for i in anno['annotations']:
#     print(i)
#     break
# for i in ques['questions']:
#     print(i)
#     break

#
# for i in range(0, 10):
#     print(anno['annotations'][i])
#     print(ques['questions'][i])
#     print("############")

# with open('E:/VQA/processed_data/features_vgg19/train2014_000072.npy', 'rb') as f:
#     a = pickle.load(f)
#
# print(a)

# img_tensor = np.load('E:/VQA/processed_data/features_vgg19/train2014_001216.npy')
# print(np.array(img_tensor).shape)


# vgg = VGG19(weights="imagenet")
# vgg = tf.keras.models.Sequential(vgg.layers[:-2])
# vgg.summary()

# with open("/media/ulgen/Samsung/VQA/embeddings/glove/glove.txt", "rb") as f:
#     a = pickle.load(f)
#
# print(a[0:10])
# print("##############")

with open(args.processed_path + 'test_data.pkl', 'rb') as f:
    image_list_test, question_list_test, answers_list_test = pickle.load(f)
    print(len(image_list_test))

# def convert_bin_word2vec_to_tex(path):
#     """
#     Before converting to txt from bin file, extract the gz file. Due to possible memory problems, reading from gz is
#     not provided in this script.
#     :param path: path of the bin file of word2vec
#     :return: None
#     """
#     loc = Path(path)
#     model = KeyedVectors.load_word2vec_format(loc, binary=True)
#     loc = loc.with_suffix('.txt')
#     model.wv.save_word2vec_format(loc)
#     print("word2vec vectors are saved as txt file to:", loc)
#
#
# convert_bin_word2vec_to_tex("/media/ulgen/Samsung/VQA/embeddings/word2vec/GoogleNews-vectors-negative300.bin")