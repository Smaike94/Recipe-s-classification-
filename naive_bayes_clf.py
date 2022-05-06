import itertools
from pprint import pprint
import json
import seaborn as sns
import pvml
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import utils


def n_relevant_words(n, vocab, weights, class_label, classes):
    voc_words = [w for w in vocab.keys()]
    index = weights[class_label, :].argsort()
    positive_words = {}
    negative_words = {}

    class_name = list(classes.keys())[class_label]

    f = open("models/naive_bayes/" + n.__str__() + "_relevant_words.txt", "a", encoding='utf-8')

    if n != 0:
        for i in index[:n]:
            negative_words[voc_words[i]] = weights[class_label, i]

        for i in index[-n:]:
            positive_words[voc_words[i]] = weights[class_label, i]

        pprint(n.__str__() + " Less relevant words for class: " + class_name, f)
        pprint(negative_words, f)
        pprint(n.__str__() + " Most relevant words for class: " + class_name, f)
        pprint(positive_words, f)
        pprint(" ", f)

    f.close()

    return positive_words, negative_words


if __name__ == '__main__':
    path_vocab = "train"
    classes_info = utils.get_classes(path_vocab)

    vocab_variations = {(True, False): "basic",
                        (False, False): "stopwords",
                        (True, True): "stem",
                        (False, True): "stem e stopwords"}

    list_n_features = [1000, 3000, 5000, 10000]
    recipes_part = ["all", "ingredients"]

    list_vocab_config = list(vocab_variations.keys())
    list_configurations = list(itertools.product(list_n_features, list_vocab_config, recipes_part))

    best_models_by_configuration = {}
    best_overall_clf_nb = None
    best_overall_config = None
    best_overall_train_acc = 0
    best_overall_test_acc = 0

    # configuration[0] = n_features
    # configuration[1][0] = sw
    # configuration[1][1] = stem
    # configuration[2] = recip_part

    for configuration in list_configurations:

        XTrain, YTrain, XTest, YTest = utils.get_datasets_from_vocab_config(configuration, path_vocab)
        # XTrain, XTest = pvml.l2_normalization(XTrain, XTest)

        # ----------------------- MODEL EVALUATIONS -----------------------------#

        # Multinomial Naive Bayes
        best_clf_nb = MultinomialNB()
        best_clf_nb.fit(XTrain, YTrain)

        train_acc = best_clf_nb.score(XTrain, YTrain) * 100
        test_acc = best_clf_nb.score(XTest, YTest) * 100

        if test_acc > best_overall_test_acc:
            best_overall_train_acc = train_acc
            best_overall_test_acc = test_acc
            best_overall_clf_nb = best_clf_nb
            best_overall_config = configuration

        key = utils.format_key(configuration)
        best_models_by_configuration[key] = {"nb": {"model": best_clf_nb.__str__(),
                                                    "train_acc": train_acc,
                                                    "test_acc": test_acc,
                                                    "params": best_clf_nb.get_params(),
                                                    }
                                             }
    # ------------------------------------------------------------------------------------ #
    # ------------------------ Statistics for best overall model ------------------------- #

    #  Recover dataset for the best overall configuration
    _, y_train, x_test, y_test = utils.get_datasets_from_vocab_config(best_overall_config, path_vocab)
    vocab = utils.get_vocabulary_from_vocab_config(best_overall_config, path_vocab)

    n_words = 10
    clf_nb_weights = best_overall_clf_nb.feature_log_prob_
    for class_label in np.unique(y_train):
        n_relevant_words(n_words, vocab, clf_nb_weights, class_label, classes_info)

    # x_test = pvml.l2_normalization(x_test)
    y_pred = best_overall_clf_nb.predict(x_test)

    prec_nb = list(precision_score(y_test, y_pred, average=None))
    prec_nb_dict = {}
    for ind, prec in enumerate(prec_nb):
        class_name = list(classes_info.keys())[ind]
        prec_nb_dict[class_name] = prec

    recall_nb = list(recall_score(y_test, y_pred, average=None))
    recall_nb_dict = {}
    for ind, rec in enumerate(recall_nb):
        class_name = list(classes_info.keys())[ind]
        recall_nb_dict[class_name] = rec

    cf_matrix = confusion_matrix(y_test, y_pred, normalize="true")
    sns_cf_matrix = sns.heatmap(cf_matrix, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                                xticklabels=list(classes_info.keys()), yticklabels=list(classes_info.keys()))
    sns_cf_matrix.get_figure().savefig('models/naive_bayes/nb_conf_mat.png', dpi=400)

    # ------------------------------------------------------------------------------------ #
    # ------------------------ Dictionary of best overall model ------------------------- #

    best_key = utils.format_key(best_overall_config)
    best_overall_model_clf_nb = {best_key: {"nb": {"model": best_overall_clf_nb.__str__(),
                                                   "train_acc": best_overall_train_acc,
                                                   "test_acc": best_overall_test_acc,
                                                   "params": best_overall_clf_nb.get_params(),
                                                   "precision": prec_nb_dict,
                                                   "recall": recall_nb_dict
                                                   }
                                            }}

    # --------------------- Save best models and best overall model ---------------------- #

    f = open("models/naive_bayes/best_models_nb.json", "w", encoding='utf-8')
    json.dump(best_models_by_configuration, f, indent=4)
    f.close()

    f = open("models/naive_bayes/best_overall_model_nb.json", "w", encoding='utf-8')
    json.dump(best_overall_model_clf_nb, f, indent=4)
    f.close()
    # ------------------------------------------------------------------------------------- #
