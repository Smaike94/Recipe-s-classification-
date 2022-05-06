import itertools
import json
import pvml
import parfit as pf
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC

import utils


def svm(x_train, y_train, x_test, y_test):
    # Model selection using linear and polynomial kernel

    grid_svm = {
        'C': [1e0, 1e-2, 1e-3, 1e-4],
        'max_iter': [1000],
        'kernel': ["linear", "poly"],
        'degree': [3, 5, 7],
        'gamma': [1],
        'coef0': [1],
        'decision_function_shape': ["ovr"]
    }

    # Model selection using radial kernel with different values for gamma

    paramGrid_svm = ParameterGrid(grid_svm)
    bestModel_svm, bestScore_svm, allModels_svm, allScores_svm = pf.bestFit(SVC, paramGrid_svm,
                                                                            x_train, y_train, x_test, y_test, verbose=0,
                                                                            metric=accuracy_score, showPlot=False)

    return bestModel_svm


def svm_radial_kernel(x_train, y_train, x_test, y_test):
    grid_rksvm = {
        'C': [1e0, 1e-2, 1e-3, 1e-4],
        'max_iter': [1000],
        'kernel': ["rbf"],
        'gamma': [1e0, 1e-2, 1e-3, 1e-9],
        'decision_function_shape': ["ovr"]
    }

    paramGrid_rksvm = ParameterGrid(grid_rksvm)
    bestModel_rksvm, bestScore_rksvm, allModels_rksvm, allScores_rksvm = pf.bestFit(SVC, paramGrid_rksvm,
                                                                                    x_train, y_train, x_test, y_test,
                                                                                    verbose=0,
                                                                                    metric=accuracy_score,
                                                                                    showPlot=False)

    return bestModel_rksvm


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
    best_overall_clf_svm = None
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

        #  SVM multiclass
        best_clf_lpsvm = svm(XTrain, YTrain, XTest, YTest)
        best_clf_rksvm = svm_radial_kernel(XTrain, YTrain, XTest, YTest)
        if best_clf_lpsvm.score(XTest, YTest) >= best_clf_rksvm.score(XTest, YTest):
            best_clf_svm = best_clf_lpsvm
        else:
            best_clf_svm = best_clf_rksvm

        train_acc = best_clf_svm.score(XTrain, YTrain) * 100
        test_acc = best_clf_svm.score(XTest, YTest) * 100

        if test_acc > best_overall_test_acc:
            best_overall_train_acc = train_acc
            best_overall_test_acc = test_acc
            best_overall_clf_svm = best_clf_svm
            best_overall_config = configuration

        key = utils.format_key(configuration)
        best_models_by_configuration[key] = {"svm": {"model": best_clf_svm.__str__(),
                                                     "train_acc": train_acc,
                                                     "test_acc": test_acc,
                                                     "params": best_clf_svm.get_params(),
                                                     }
                                             }

    # ------------------------------------------------------------------------------------ #

    # ------------------------ Statistics for best overall model ------------------------- #

    _, y_train, x_test, y_test = utils.get_datasets_from_vocab_config(best_overall_config, path_vocab)

    # x_test = pvml.l2_normalization(x_test)
    y_pred = best_overall_clf_svm.predict(x_test)

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
    sns_cf_matrix = sns.heatmap(cf_matrix, annot=True, fmt='.2%', cmap='Greens', cbar=False,
                                xticklabels=list(classes_info.keys()), yticklabels=list(classes_info.keys()))
    sns_cf_matrix.get_figure().savefig('models/svm/svm_conf.png', dpi=400)

    # ------------------------------------------------------------------------------------ #
    # ------------------------ Dictionary of best overall model ------------------------- #

    best_key = utils.format_key(best_overall_config)
    best_overall_model_clf_svm = {best_key: {"svm": {"model": best_overall_clf_svm.__str__(),
                                                     "train_acc": best_overall_train_acc,
                                                     "test_acc": best_overall_test_acc,
                                                     "params": best_overall_clf_svm.get_params(),
                                                     "precision": prec_nb_dict,
                                                     "recall": recall_nb_dict
                                                     }
                                             }}

    # --------------------- Save best models and best overall model ---------------------- #
    f = open("models/svm/best_models_svm.json", "w", encoding='utf-8')
    json.dump(best_models_by_configuration, f, indent=4)
    f.close()

    f = open("models/svm/best_overall_model_svm.json", "w", encoding='utf-8')
    json.dump(best_overall_model_clf_svm, f, indent=4)
    f.close()
    # ------------------------------------------------------------------------------------- #
