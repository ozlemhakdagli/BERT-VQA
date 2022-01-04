import argparse
import os

import numpy as np
import plac
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, \
    preprocess_input as resnet_preprocess_input

from utilities.utils import load_configurations, set_memory_growth

set_memory_growth()
K.set_image_data_format('channels_first')
configs = load_configurations()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

parser = argparse.ArgumentParser()
parser.add_argument("--vqa_datasets", type=str, default=configs['vqa_dataset_path'],
                    help="main VQA files data path")

parser.add_argument("--processed_path", type=str, default=configs['vqa_processed_data_path'],
                    help="processed files path")
args = parser.parse_args()


def image_feature_extractor_xception(target_path, image_list, img_batch_size, subset):
    model = Xception(include_top=False, pooling='avg', input_tensor=Input(shape=(3, 299, 299)))

    # add a progress bar
    progbar = utils.Progbar(int(np.ceil(len(image_list) / float(img_batch_size))))

    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(image_list), img_batch_size)):
        # extract batch of images and prepare them to pass it through the VGG
        # network for feature extraction

        progbar.update(b + 1)

        batch_range = range(i, min(i + img_batch_size, len(image_list)))
        batchPaths = image_list[batch_range[0]: batch_range[-1] + 1]

        batchImages = []
        batchIds = []
        # loop over the images and labels in the current batch
        for imagePath in batchPaths:
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            img = image.load_img(args.vqa_datasets + subset + '/' + imagePath,
                                 target_size=(299, 299), interpolation='lanczos')
            img = image.img_to_array(img)

            # preprocess the image by
            # (1) expanding the dimensions to include batch dim and
            # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
            img = np.expand_dims(img, axis=0)
            img = xception_preprocess_input(img)

            # add the image to the batch
            batchImages.append(img)
            # image ids of the batch
            batchIds.append(imagePath.split('.')[0][-6:])

        batchImages = np.vstack(batchImages)  # (BATCH_SIZE, 3, 224, 224)
        # pass the images through the network and use the outputs as our actual features
        features = model.predict(batchImages)  # (BATCH_SIZE, 512, 7, 7)
        # features = tf.reshape(features, (features.shape[0], features.shape[1], -1))  # (BATCH_SIZE, 512, 49)
        # features = tf.transpose(features, perm=[0, 2, 1])  # (BATCH_SIZE, 49, 512)

        # loop over the batch to save them locally
        for id, feat in zip(batchIds, features):
            np.save(os.path.join(target_path, subset + '_' + id), feat)


def image_feature_extractor_resnet_v2(target_path, image_list, img_batch_size, subset):
    model = InceptionResNetV2(include_top=False, pooling='avg', input_tensor=Input(shape=(3, 299, 299)))

    # add a progress bar
    progbar = utils.Progbar(int(np.ceil(len(image_list) / float(img_batch_size))))

    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(image_list), img_batch_size)):
        # extract batch of images and prepare them to pass it through the VGG
        # network for feature extraction

        progbar.update(b + 1)

        batch_range = range(i, min(i + img_batch_size, len(image_list)))
        batchPaths = image_list[batch_range[0]: batch_range[-1] + 1]

        batchImages = []
        batchIds = []
        # loop over the images and labels in the current batch
        for imagePath in batchPaths:
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            img = image.load_img(args.vqa_datasets + subset + '/' + imagePath,
                                 target_size=(299, 299), interpolation='lanczos')
            img = image.img_to_array(img)

            # preprocess the image by
            # (1) expanding the dimensions to include batch dim and
            # (2) subtracting the mean RGB pixel intensity from the ImageNet dataset
            img = np.expand_dims(img, axis=0)
            img = resnet_preprocess_input(img)

            # add the image to the batch
            batchImages.append(img)
            # image ids of the batch
            batchIds.append(imagePath.split('.')[0][-6:])

        batchImages = np.vstack(batchImages)  # (BATCH_SIZE, 3, 224, 224)
        # pass the images through the network and use the outputs as our actual features
        features = model.predict(batchImages)  # (BATCH_SIZE, 512, 7, 7)
        # features = tf.reshape(features, (features.shape[0], features.shape[1], -1))  # (BATCH_SIZE, 512, 49)
        # features = tf.transpose(features, perm=[0, 2, 1])  # (BATCH_SIZE, 49, 512)

        # loop over the batch to save them locally
        for id, feat in zip(batchIds, features):
            np.save(os.path.join(target_path, subset + '_' + id), feat)


def main():
    subset = str('val2014')
    image_list = os.listdir(args.vqa_datasets + subset)
    img_batch_size = 512
    target_path = args.processed_path + 'features_resnet'

    # image_feature_extractor_xception(target_path=target_path,
    #                                  image_list=image_list,
    #                                  img_batch_size=img_batch_size,
    #                                  subset=subset)

    image_feature_extractor_resnet_v2(target_path=target_path,
                                      image_list=image_list,
                                      img_batch_size=img_batch_size,
                                      subset=subset)


if __name__ == "__main__":
    plac.call(main)
