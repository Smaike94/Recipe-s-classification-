import os
import numpy as np
import vocabulary as voc
from sklearn.feature_extraction.text import CountVectorizer
import h5py


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


class DataSet:
    def __init__(self, path: str):
        self.path = path

    def create(self, vocab: tuple, n_features: int):
        vocabulary: dict = vocab[1]
        add_info: str = vocab[0]

        if add_info.__contains__("stem"):
            stem = True
        else:
            stem = False
        if add_info.__contains__("SW"):
            sw = False
        else:
            sw = True

        if add_info.__contains__("all"):
            recip_part = "all"

        if add_info.__contains__("ingredients"):
            recip_part = "ingredients"

        Xset = []
        Yset = []
        try:
            with open("datasets/dataset_" + self.path + "_" + n_features.__str__() + "_" + add_info + ".hdf5",
                      "r") as ds:
                f = h5py.File(ds.name, 'r')
                dataSet = f["dataset"]
                Xset = dataSet[:, :-1]
                Yset = dataSet[:, -1]
        except IOError:

            formatted_recipes = []
            classes = get_classes(self.path)
            for f in os.listdir("recipes/" + self.path):
                formatted_recipes.append(" ".join(voc.read_document_for_voc("recipes/" + self.path + "/" + f,
                                                                            include_stopwords=sw, stem=stem,
                                                                            recipe_part=recip_part)))
                class_name = f.split("-")[0]
                Yset.append(classes[class_name]["label"])

            countvec = CountVectorizer(max_features=n_features, vocabulary=vocabulary)
            count_data = countvec.fit_transform(formatted_recipes)

            Xset = np.stack(count_data.toarray())
            Yset = np.stack(Yset)
            dataset = np.column_stack((Xset, Yset.T))

            with h5py.File("datasets/dataset_" + self.path + "_" + n_features.__str__() + "_" + add_info + ".hdf5",
                           "w") as f:
                f.create_dataset("dataset", data=dataset, dtype='i', compression='gzip')
            f.close()

            if vocabulary is None:
                f = open("vocabularies/vocabulary_" + n_features.__str__() + "_" + add_info + ".txt",
                         "w", encoding='utf-8')

                for feature in countvec.get_feature_names():
                    print(feature, file=f)
                f.close()

        return Xset, Yset
