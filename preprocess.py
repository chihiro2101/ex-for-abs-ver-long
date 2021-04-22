import math
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
import os
import re
import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity


# def word_frequencies(list_sentences, title):
#     list_sentences_frequency = {}
#     tmp_dict = {'word_frequencies': '', 'value_word_in_sent': ''}
#     # Each sentence
#     for index, sentence in enumerate(list_sentences):
#         sentence = list_sentences_to_string(sentence)
#         word_frequencies = {}  # đếm mỗi từ xuất hiện bao nhiêu lần
#         for word in nltk.word_tokenize(sentence):
#             if word not in word_frequencies.keys():
#                 word_frequencies[word] = 1
#             else:
#                 word_frequencies[word] += 1
#         list_sentences_frequency[index] = word_frequencies

#         word_in_sent = {}
#         value_word_in_sent = {}
#         n = len(list_sentences)
#         for word in word_frequencies.keys():
#             word_in_sent[word] = 0
#             for sentence in list_sentences:
#                 if word in nltk.word_tokenize(sentence):
#                     word_in_sent[word] += 1
#             # trả về kiểu {'hi': 0.5, 'hello': 0.6,...}
#             value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

#         tmp_dict['word_frequencies'] = word_frequencies
#         tmp_dict['value_word_in_sent'] = value_word_in_sent
#         list_sentences_frequency[index] = tmp_dict

#     # All sentences
#     list_sentences_string = list_sentences_to_string(list_sentences)
#     word_frequencies = {}
#     for word in nltk.word_tokenize(list_sentences_string):
#         if word not in word_frequencies.keys():
#             word_frequencies[word] = 1
#         else:
#             word_frequencies[word] += 1

#     word_in_sent = {}
#     value_word_in_sent = {}
#     n = len(list_sentences)
#     for word in word_frequencies.keys():
#         word_in_sent[word] = 0
#         for sentence in list_sentences:
#             if word in nltk.word_tokenize(sentence):
#                 word_in_sent[word] += 1
#         value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

#     tmp_dict['word_frequencies'] = word_frequencies
#     tmp_dict['value_word_in_sent'] = value_word_in_sent
#     list_sentences_frequency['list_sentences'] = tmp_dict

#     # Title
#     title = list_sentences_to_string(title)
#     word_frequencies = {}
#     for word in nltk.word_tokenize(title):
#         if word not in word_frequencies.keys():
#             word_frequencies[word] = 1
#         else:
#             word_frequencies[word] += 1

#     word_in_sent = {}
#     value_word_in_sent = {}
#     n = len(list_sentences)
#     for word in word_frequencies.keys():
#         word_in_sent[word] = 0
#         for sentence in list_sentences:
#             if word in nltk.word_tokenize(sentence):
#                 word_in_sent[word] += 1
#         value_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))

#     tmp_dict['word_frequencies'] = word_frequencies
#     tmp_dict['value_word_in_sent'] = value_word_in_sent
#     list_sentences_frequency['title'] = tmp_dict
#     return list_sentences_frequency


# def return_vocab(list_sentences_frequency, key='list_sentences'):
#     return list(list_sentences_frequency[key]['word_frequencies'])


# def weight(list_sentences_frequency, key):
#     word_frequencies = list_sentences_frequency[key]['word_frequencies']
#     value_word_in_sents = list_sentences_frequency[key]['value_word_in_sent']
#     if len(word_frequencies) == 0:
#         maximum_frequency = 0
#     else:
#         maximum_frequency = max(word_frequencies.values())

#     for word in word_frequencies.keys():
#         if maximum_frequency == 0:
#             word_frequencies[word] = 0
#         else:
#             word_frequencies[word] = (
#                 word_frequencies[word]/maximum_frequency)*value_word_in_sents[word]
#     return word_frequencies


def normalize(vec):
    return vec / np.sqrt(np.sum(vec ** 2))


def simCos(vec1, vec2):
    norm_vec1 = normalize(vec1)
    norm_vec2 = normalize(vec2)
    return np.sum(norm_vec1 * norm_vec2)


def sim_2_sent(df_tfidf):
    matrix_cossim = cosine_similarity(df_tfidf, df_tfidf)
    return matrix_cossim


def sim_with_title(list_sentences_frequency, title):
    simWithTitle = []
    for sent_vector in list_sentences_frequency:
        simT = cosine_similarity([sent_vector], [title])[0][0]
        simWithTitle.append(simT)
    return simWithTitle


def sim_with_doc(list_sentences_frequency, document_vector):
    simWithDoc = []
    for sent_vector in list_sentences_frequency:
        simD = cosine_similarity([sent_vector], [document_vector])[0][0]
        simWithDoc.append(simD)
    return simWithDoc


def count_noun(sentences, option = False):
    if option == False:
        number_of_nouns = [0]*len(sentences)
    else:
        number_of_nouns = []
        for sentence in sentences:
            text = nltk.word_tokenize(sentence)
            post = nltk.pos_tag(text)
            # noun_list = ['NN', 'NNP', 'NNS', 'NNPS']
            noun_list = ['NNP']
            num = 0
            for k, v in post:
                if v in noun_list:
                    num += 1
            number_of_nouns.append(num)
    return number_of_nouns


def preprocess_raw_sent(raw_sent):
    symbols = "!\"#$%&()*+-./:;,\'<=>?@[\]^_`{|}~\n"
    for i in symbols:
        raw_sent = raw_sent.replace(i, '')
    remove_number = "".join((item for item in raw_sent if not item.isdigit())).strip()
    stopwords = nltk.corpus.stopwords.words('english')
    text_tokens = nltk.word_tokenize(remove_number)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    # stemmer= PorterStemmer()
    # stemmed_word = [stemmer.stem(word) for word in tokens_without_sw]

    preprocessed_sent = (" ").join(tokens_without_sw)
    return preprocessed_sent


# def preprocess_numberOfNNP(raw_sent):
#     words = nltk.word_tokenize(raw_sent)
#     preprocess_words = ""
#     stopwords = nltk.corpus.stopwords.words('english')
#     # stemmer= PorterStemmer()
#     for word in words:
#         if word.isalpha():
#             if word not in stopwords:
#                 word = " " + word
#                 preprocess_words += word
#     preprocess_words = preprocess_words.strip()
#     return preprocess_words
