import pandas as pd
import matplotlib.pyplot as plt
from utils import get_classes
from sklearn.feature_extraction.text import TfidfVectorizer
import vocabulary as voc


def tfidf_analysis_by_class(vocab_config: tuple, path: str):
    formatted_recipes_by_class = []
    most_common_words_for_each_class = []

    n_features = vocab_config[0]
    sw = vocab_config[1][0]
    stem = vocab_config[1][1]
    recipe_part = vocab_config[2]

    class_info = get_classes(path)
    for _class in class_info:
        for f in class_info[_class]["files_name_" + path]:
            formatted_recipes_by_class.append(
                " ".join(voc.read_document_for_voc("recipes/" + path + "/" + f, include_stopwords=sw,
                                                   stem=stem, recipe_part=recipe_part)))

        inverse_recipes_length = [1 / (len(recipe)) for recipe in formatted_recipes_by_class]

        # In this step TfidfVectorizer create a table in which rows are documents and for each columns
        # there is the tf-idf statistic not normalized over the length of the document.
        # So, for example the [i][j] represent the tf-idf stat for the j-th word present in the i-th recipe
        # and tf-idf are expressed in this way = (n_occurrences of j-th word) * (idf of j-th word)
        tf_idf_vec = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        tf_idf_data = tf_idf_vec.fit_transform(formatted_recipes_by_class)
        tf_idf_dataframe = pd.DataFrame(tf_idf_data.toarray(), columns=tf_idf_vec.get_feature_names())
        # Divide each word occurrence by the length of its document so the tf-tdf measure now is expressed
        # in this way [(n_occurrences of j-th word)/(i-th document length)] * (idf of j-th word)
        tmp = tf_idf_dataframe.mul(inverse_recipes_length, axis=0)

        features_idf_dict = {}
        for item in sorted(tf_idf_vec.vocabulary_.items()):
            feature = item[0]
            index = item[1]
            feature_idf = tf_idf_vec.idf_[index]
            features_idf_dict[index] = {"feature": feature, "idf": feature_idf}

        # With this dataframe now we have feature and related idf value
        features_tfidf_df = pd.DataFrame.from_dict(features_idf_dict, orient='index')
        # These two next transformations allows to recover the meant tf for each word. Since idf is
        # equal among all documents making the mean and then divide it for idf, will return
        # the mean term frequency for each word. So with this dataframe I obtained the idf value and mean tf
        # for each word and so it is possible to filter for the most common words, according to certain values
        # of idf and mean tf
        features_tfidf_df = features_tfidf_df.assign(tf_idf_mean=tmp.mean().sort_index().values * 100)
        features_tfidf_df["tf_mean"] = features_tfidf_df["tf_idf_mean"] / features_tfidf_df["idf"]

        max_tf = features_tfidf_df["tf_mean"].max()
        min_tf = features_tfidf_df["tf_mean"].min()
        max_idf = features_tfidf_df["idf"].max()
        min_idf = features_tfidf_df["idf"].min()

        # Filter for word with low idf and high term frequency, the integer values used here have been obtained
        # through some experiments looking at the scatter plot of the two measures
        most_common_word_by_class = features_tfidf_df.loc[(features_tfidf_df['tf_mean'] >= (max_tf - min_tf) / 3)
                                                          & (features_tfidf_df['idf'] <= (min_idf + 2))]

        most_common_words_for_each_class.append(list(most_common_word_by_class["feature"]))

        ax = features_tfidf_df.plot.scatter(x="idf", y="tf_mean")
        for i, txt in enumerate(features_tfidf_df.feature):
            ax.annotate(txt, (features_tfidf_df.idf.iat[i], features_tfidf_df.tf_mean.iat[i]))
        # plt.savefig("tfidf_" + _class +"_.png")
        plt.show()
        formatted_recipes_by_class.clear()

    return most_common_words_for_each_class


def samples_histogram():
    classes_info_train = get_classes("train")
    classes_info_test = get_classes("test")

    plt.title("Number of samples for each class")
    samples_train = []
    samples_test = []
    for class_name in classes_info_train.keys():
        samples_train.append(len(classes_info_train[class_name]["files_name_train"]))
        samples_test.append(len(classes_info_test[class_name]["files_name_test"]))

    plt.bar(list(classes_info_train.keys()), samples_train, label="train")
    plt.bar(list(classes_info_train.keys()), samples_test, label="test")

    plt.legend()
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path_vocab = "train"

    # --- Number of samples for each class ---#

    samples_histogram()

    # --- Data analysis and exploration --- #

    # Input of these methods are tuple with the same format of vocabulary configuration
    # number of features is not important so it's set None
    # the other tuple tells that the already stopwords present in stopwords.txt have to not be considered from the
    # analysis. Then no stemmed words and all recipe text have been considered
    common_words_for_class = tfidf_analysis_by_class((None, (False, False), "all"), path_vocab)

    # ------------------- Find most common words among classes ---------------------------------- #
    expressions = []
    for ind, el in enumerate(common_words_for_class):
        expressions.append("set(common_words_for_class[" + ind.__str__() + "])")
    expression = "&".join(expressions)
    most_common_words_present_in_all_classes = list(eval(expression))

    # ------------------ Add these domain specific words to stopwords -------------------------- #

    with open("recipes/stopwords.txt", "a", encoding="utf-8") as f:
        for word in most_common_words_present_in_all_classes:
            print(word, file=f)


