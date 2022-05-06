import itertools
import json
import seaborn as sns
import parfit as pf
import pvml
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
import utils


def multilayer_perceptron_classifier(x_train, y_train, x_test, y_test):
    grid_mlp = {
        'hidden_layer_sizes': [(10,), (20,), (50,)],
        'activation': ["relu"],
        'solver': ["sgd"],
        'alpha': [1e-3, 1, 1e3],
        'batch_size': ["auto"],
        'learning_rate': ["adaptive"],
        'learning_rate_init': [1e-4],
        'max_iter': [1000],
        'momentum': [0.9],
        'verbose': [False],
    }

    paramGrid_mlp = ParameterGrid(grid_mlp)
    bestModel_mlp, bestScore_mlp, allModels_mlp, allScores_mlp = pf.bestFit(MLPClassifier, paramGrid_mlp,
                                                                            x_train, y_train, x_test, y_test, verbose=0,
                                                                            metric=accuracy_score, showPlot=False)

    return bestModel_mlp


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
    best_overall_clf_mlp = None
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

        #  Neural network
        best_clf_mlp = multilayer_perceptron_classifier(XTrain, YTrain, XTest, YTest)
        train_acc = best_clf_mlp.score(XTrain, YTrain) * 100
        test_acc = best_clf_mlp.score(XTest, YTest) * 100

        if test_acc > best_overall_test_acc:
            best_overall_train_acc = train_acc
            best_overall_test_acc = test_acc
            best_overall_clf_mlp = best_clf_mlp
            best_overall_config = configuration

        key = utils.format_key(configuration)
        best_models_by_configuration[key] = {"mlp": {"model": best_clf_mlp.__str__(),
                                                     "train_acc": train_acc,
                                                     "test_acc": test_acc,
                                                     "params": best_clf_mlp.get_params(),
                                                     }
                                             }

    # ------------------------------------------------------------------------------------ #
    # ------------------------ Statistics for best overall model ------------------------- #

    _, y_train, x_test, y_test = utils.get_datasets_from_vocab_config(best_overall_config, path_vocab)

    # x_test = pvml.l2_normalization(x_test)
    y_pred = best_overall_clf_mlp.predict(x_test)

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
    sns_cf_matrix = sns.heatmap(cf_matrix, annot=True, fmt='.2%', cmap='Greys', cbar=False,
                                xticklabels=list(classes_info.keys()), yticklabels=list(classes_info.keys()))
    sns_cf_matrix.get_figure().savefig('models/mlp/mlp_conf_mat.png', dpi=400)

    # ------------------------------------------------------------------------------------ #
    # ------------------------ Dictionary of best overall model ------------------------- #

    best_key = utils.format_key(best_overall_config)
    best_overall_model_clf_mlp = {best_key: {"mlp": {"model": best_overall_clf_mlp.__str__(),
                                                     "train_acc": best_overall_train_acc,
                                                     "test_acc": best_overall_test_acc,
                                                     "params": best_overall_clf_mlp.get_params(),
                                                     "precision": prec_nb_dict,
                                                     "recall": recall_nb_dict
                                                     }
                                             }}

    # --------------------- Save best models and best overall model ---------------------- #
    f = open("models/mlp/best_models_mlp.json", "w", encoding='utf-8')
    json.dump(best_models_by_configuration, f, indent=4)
    f.close()

    f = open("models/mlp/best_overall_model_mlp.json", "w", encoding='utf-8')
    json.dump(best_overall_model_clf_mlp, f, indent=4)
    f.close()
    # ------------------------------------------------------------------------------------- #
