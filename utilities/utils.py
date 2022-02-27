import re

import en_core_web_lg
import numpy as np
import spacy
import tensorflow as tf
import yaml
from gensim.models import KeyedVectors
from tqdm import tqdm
from pathlib import Path


def set_memory_growth():
    """
    This function sets memory growth for tensorflow. With TF2, it is optimized to use graphic memory efficiently.
    TF2 only allocates the required size and releases the rest.
    :return: None
    """
    print('Using tf version: {}'.format(tf.__version__))
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("experimental memory growth activated")
    except:
        print("memory growth could not be activated")
        pass


def load_configurations():
    """
    Loads the configuration yaml file that contains all the paths and neural network parameters. Majority of the
    settings are handled in configuration yaml file. This enables single centralized control over parameters and paths.
    :return: returns configurations.
    """
    with open('/home/ozlem/Documents/pycharmProjects/BERT-VQA/configurations.yaml', 'r') as stream:
        try:
            configurations = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

        return configurations


def load_spacy_nlp(configs, transformer_type):
    """
    Loads spacy based NLP object which converts raw text into ids of tokens. Currently, this project supports three
    types of NLP embeddings objects. Glove - Fasttext - Ontonotes5
    :param configs: path of the transformer object.
    :param transformer_type: type definition of the transformer.
    :return: returns NLP object.
    """

    if transformer_type == 'ontonotes5':
        print("Loading", transformer_type, "NLP object")
        nlp = en_core_web_lg.load(disable=['parser', 'tagger', 'ner', 'textcat', 'lemmatizer',
                                           'attribute_ruler', 'tok2vec'])
    else:
        print(transformer_type, "NLP object is loaded")
        nlp = spacy.load(configs[transformer_type], disable=['parser', 'tagger', 'ner', 'textcat', 'lemmatizer',
                                                             'attribute_ruler', 'tok2vec'])
    """shows the unique vector size/count."""
    # print(transformer_type, "unique vector size / count", len(nlp.vocab.vectors))
    return nlp


def get_embeddings(vocab, nr_unk):
    """
    This function takes the embeddings from nlp object and adds random weights for unknown words. Later it saves
    the whole new vector object to disk as pkl file. It will be used in embedding layer to match the word ids with
    corresponding vectors.
    :param vocab: nlp vocabulary object.
    :param nr_unk: unknown word vector size that will be used to create random_norm vectors for out of vocabulary words.
    :return: returns new vector table.
    """
    print("text embeddings are being extracted")
    # the extra +1 is for a zero vector representing sentence-final padding
    num_vectors = max(lex.rank for lex in vocab) + 2
    # create random vectors for OOV tokens
    oov = np.random.normal(size=(nr_unk, vocab.vectors_length))
    oov = oov / oov.sum(axis=1, keepdims=True)

    vectors = np.zeros((num_vectors + nr_unk, vocab.vectors_length), dtype='float32')
    vectors[1:(nr_unk + 1), ] = oov
    for lex in tqdm(vocab):
        if lex.has_vector and lex.vector_norm > 0:
            vectors[nr_unk + lex.rank + 1] = np.asarray(lex.vector / lex.vector_norm)

    print("Embedding extraction completed")

    return vectors


def process_sentence(sentence):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [';', r"/", '[', ']', '"', '{', '}',
             '(', ')', '=', '+', '\\', '_', '-',
             '>', '<', '@', '`', ',', '?', '!']
    articles = ['a', 'an', 'the']
    manual_map = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10'
    }
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
        "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
        "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
        "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
        "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
        "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
        "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
        "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
        "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
        "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
        "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
        "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
        "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
        "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
        "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
        "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
        "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
        "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
        "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
        "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"
    }

    in_text = sentence.replace('\n', ' ')
    in_text = in_text.replace('\t', ' ')
    in_text = in_text.strip()
    out_text = in_text

    for p in punct:
        if (p + ' ' in in_text or ' ' + p in in_text) or (re.search(commaStrip, in_text) is not None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    in_text = periodStrip.sub("", out_text, re.UNICODE)

    outText = []
    tempText = in_text.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    out_text = ' '.join(outText)

    return out_text


def convert_bin_word2vec_to_tex(path):
    """
    Before converting to txt from bin file, extract the gz file. Due to possible memory problems, reading from gz is
    not provided in this script.
    :param path: path of the bin file of word2vec
    :return: None
    """
    loc = Path(path)
    model = KeyedVectors.load_word2vec_format(loc, binary=True)
    loc = loc.with_suffix('.txt')
    model.wv.save_word2vec_format(loc)
    print("word2vec vectors are saved as txt file to:", loc)
