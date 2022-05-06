import collections
import itertools

import json
import os
from pprint import pprint

from matplotlib import pyplot as plt
import numpy as np
import vocabulary as voc
from DataSets import DataSet


def get_datasets_from_vocab_config(vocab_config: tuple, vocab_path: str):
    n_features = vocab_config[0]
    sw = vocab_config[1][0]
    stem = vocab_config[1][1]
    recipe_part = vocab_config[2]

    # Create vocabularies and datasets
    vocab = voc.load_vocabulary(vocab_path, n_features, include_stopwords=sw, stem=stem, recipe_part=recipe_part)
    XTrain, YTrain = DataSet("train").create(vocab, n_features)

    if vocab[1] is None:
        vocab = voc.load_vocabulary(vocab_path, n_features, include_stopwords=sw, stem=stem,
                                    recipe_part=recipe_part)
    XTest, YTest = DataSet("test").create(vocab, n_features)

    return XTrain, YTrain, XTest, YTest


def get_vocabulary_from_vocab_config(vocab_config: tuple, vocab_path: str):
    n_features = vocab_config[0]
    sw = vocab_config[1][0]
    stem = vocab_config[1][1]
    recipe_part = vocab_config[2]

    # Create vocabularies and datasets
    vocab = voc.load_vocabulary(vocab_path, n_features, include_stopwords=sw, stem=stem, recipe_part=recipe_part)

    return vocab[1]


def format_key(configuration: tuple):
    formatted_key = configuration[0].__str__() + "-" + str(configuration[1][0]) + "-" + str(configuration[1][1]) + \
                    "-" + configuration[2]
    return formatted_key


def get_classes(path_dir: str):
    class_names = {}
    class_label = 0
    for f in os.listdir("recipes/" + path_dir):
        class_name = f.split("-")[0]
        if class_name not in class_names:
            class_names[class_name] = {"label": class_label}
            class_label += 1

    for class_name in class_names:
        recipes_by_class = [f for f in os.listdir("recipes/" + path_dir) if f.__contains__(class_name)]
        class_names[class_name]["files_name_" + path_dir] = recipes_by_class

    return class_names


def get_most_common_n_words_by_class(path_dir: str, n_words: int, files_list: list):
    vocabulary_by_class = collections.Counter()
    for file in files_list:
        vocabulary_by_class.update(
            voc.read_document_for_voc("recipes/" + path_dir + "/" + file, include_stopwords=False))

    pprint(vocabulary_by_class.most_common(n_words))


def get_tuple_conf(str_conf: str):
    conf_list = str_conf.split("-")
    n_features = int(conf_list[0])
    sw = True if conf_list[1] == "True" else False
    stem = True if conf_list[2] == "True" else False
    recipe_part = conf_list[3]

    return n_features, (sw, stem), recipe_part


def plot_accuracies_by_model_type(model_type: str):
    fig_title = ""
    if model_type == "naive_bayes":
        suffix = "nb"
        fig_title = "Multinomial Naive Bayes"
    elif model_type == "log_reg":
        suffix = "lr"
        fig_title = "Multinomial Logistic Regression"
    elif model_type == "mlp":
        suffix = model_type
        fig_title = "Multilayer Perceptron Classifier"
    elif model_type == "svm":
        suffix = model_type
        fig_title = "Support Vector Machine"

    best_models = {}
    best_overall_model = {}

    with open(f"models/{model_type}/best_models_{suffix}.json", "r", encoding='utf-8') as best_models_json:
        tmp = json.load(best_models_json)
        for conf, model in tmp.items():
            voc_conf = get_tuple_conf(conf)
            best_models[voc_conf] = model

    with open(f"models/{model_type}/best_overall_model_{suffix}.json", "r", encoding='utf-8') as best_model_json:
        tmp = json.load(best_model_json)
        for conf, model in tmp.items():
            voc_conf = get_tuple_conf(conf)
            best_overall_model[voc_conf] = model

    list_n_features = [1000, 3000, 5000, 10000]
    recipes_part = ["all", "ingredients"]

    fig_num = 1
    fig = plt.figure(num=fig_num)
    fig.suptitle(fig_title, fontsize='xx-large')
    axs = fig.subplots(2, 1)

    for ax_ind, ax in enumerate(axs.flat):
        # voc_config, sub_title = list(vocab_variations.items())[outer_ind]
        recipe_part = recipes_part[ax_ind]
        ax.set_title(recipe_part.capitalize())
        ax.set_ylabel('accuracies')

        x_label_names = []
        train_accuracies = []
        test_accuracies = []
        for n_features in list_n_features:
            for voc_config in list(vocab_variations.keys()):
                model_by_conf = best_models[(n_features, voc_config, recipe_part)]
                train_accuracies.append(model_by_conf[suffix]["train_acc"])
                test_accuracies.append(model_by_conf[suffix]["test_acc"])
                x_label_names.append(f"{n_features} {vocab_variations[voc_config]}")

        ax.set_xticks(np.arange(len(list_n_features) * len(list(vocab_variations.keys()))))

        ax.plot(np.arange(len(x_label_names)), train_accuracies, label="train")
        ax.plot(np.arange(len(x_label_names)), test_accuracies, label="test")

        model_key = list(best_overall_model.keys())[0]
        if recipe_part == model_key[2]:
            x_point = list(itertools.product(list_n_features, list(vocab_variations.keys()))) \
                .index((model_key[0], model_key[1]))
            ax.plot([x_point], [best_overall_model[model_key][suffix]["test_acc"]], 'ro', label="best\nmodel")

        if ax_ind == 1:
            ax.xaxis.set_ticklabels(x_label_names)
        else:
            ax.xaxis.set_ticklabels([])

        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        ax.grid()
    plt.show()


if __name__ == '__main__':

    vocab_variations = {(True, False): "basic",
                        (False, False): "stopwords",
                        (True, True): "stem",
                        (False, True): "stem\nstopwords"}

    model_types = ["naive_bayes", "log_reg", "svm", "mlp"]
    for mod_type in model_types:
        plot_accuracies_by_model_type(mod_type)
