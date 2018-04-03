'''
Demoes the intent classifier
'''
import os
import json
import sys
import csv

import torch
import torchtext.data as data
import pycrfsuite

from text_dl.inference import fields_factory
from text_dl.modules.embeddings import embedding_factory
from text_dl.models import models_factory
from text_dl.common.devices import use_cuda

from datasets import load_windows, load_articles, generate_examples, Article, Window
import features.features as features


#TODO: FOr now I will do this here. Later I will redraw it to
#something better.  Pytorch text is not designed to help you 
#out with the first pipeline for training and for testing
#so I need to change that.
def transform_query(query_text, name_field_t, dataset):
    #1. Create example from query text
    example = data.Example.fromlist([query_text], [name_field_t])
    examples = [example]

    #2. Create batch using example and dataset
    device_type = None if use_cuda else -1
    batch = data.Batch(data = examples, dataset = dataset, device = device_type)

    #3. Return batch
    return batch

def read_labels(labels_file_path):
    
    reader = csv.reader(open(labels_file_path, 'r'))
    labels_intent = {}
    for row in reader:
        key, value = row
        labels_intent[int(key)] = value
    
    return labels_intent

def get_query_ner_tags(query_text, tagger, feature_generators, skip_chain_left, skip_chain_right):
    query_text = query_text.lower()
    article = Article(query_text)
    window = Window(article.tokens)
    window.apply_features(feature_generators)
    feature_values_lists = []
    for word_idx in range(len(window.tokens)):
        fvl = window.get_feature_values_list(word_idx, skip_chain_left, skip_chain_right)
        feature_values_lists.append(fvl)
    tagged_sequence = tagger.tag(feature_values_lists)
    return " ".join(tagged_sequence)

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    cl_conf = conf['cl_conf']
    ner_conf = conf['ner_conf']

    #1. Load classification model
    model_weights_path = cl_conf['model_weights_path']
    original_conf_file = cl_conf['original_conf_file']
    labels_file_path = cl_conf['labels_file_path']
    labels_intent = read_labels(labels_file_path)

    with open(original_conf_file) as f:
        original_conf = json.load(f)
    model_config = original_conf['model']
    generator_config = original_conf['generator']
    
    vocabulary, (name, text_field), dataset = fields_factory(generator_config)
    embedding = embedding_factory(vocabulary, train_embedding = False)
    model_config['params']['embedding'] = embedding
    del model_config['params']['train_embedding']

    model = models_factory(model_config)
    if use_cuda:
        model.cuda()
    model_weights = torch.load(model_weights_path)
    model.load_state_dict(model_weights)
    model.eval()

    #2. Load ner model
    gaz_filepaths = ner_conf.get('gaz_filepaths', None)
    brown_clusters_filepath = ner_conf.get('brown_clusters_filepath', None)
    w2v_clusters_filepath = ner_conf.get('w2v_clusters_filepath', None)
    lda_model_filepath = ner_conf.get('lda_model_filepath', None)
    lda_dictionary_filepath = ner_conf.get('lda_dictionary_filepath', None)
    lda_cache_filepath = ner_conf.get('lda_cache_filepath', None)
    skip_chain_left = ner_conf['skip_chain_left']
    skip_chain_right = ner_conf['skip_chain_right']

    print("Creating feature extractors... ")
    tagger = pycrfsuite.Tagger()
    tagger.open("train")
    feature_generators = features.create_features(gaz_filepaths, brown_clusters_filepath,
                                    w2v_clusters_filepath, lda_model_filepath, 
                                    lda_dictionary_filepath, lda_cache_filepath, 
                                    verbose = True, lda_window_left_size = 5,
                                    lda_window_right_size = 5)


    query_text = None
    while True:
        query_text = input("Your question: ")
        if query_text == "exit":
            break
        tags = get_query_ner_tags(query_text, tagger, feature_generators, skip_chain_left, skip_chain_right)
        batch = transform_query([query_text, tags], [(name1, text_field), (name2, tag_field)], dataset)
        prediction = model.forward(batch.text)
        maximum = torch.max(prediction, 1)
        arg_max = maximum[1][0]
        arg_max = arg_max.data.item()
        print(labels_intent[arg_max])

if __name__ == "__main__":
    main()
