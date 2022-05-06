from stemmer import porter
import re


def import_vocabulary(filename):
    f = open(filename, encoding="utf-8")
    n = 0
    voc = {}
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.close()
    return voc


def read_document_for_voc(filename, include_stopwords=True, stem=False, recipe_part="all"):
    p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789°"
    table = str.maketrans(p, " " * len(p))

    f = open(filename, encoding="utf-8")
    text = f.read()
    text = text.lower()

    if recipe_part == "ingredients":
        try:
            text = text[text.index("ingredients:"): text.index("preparation:")]
        except ValueError:
            title, ingredients, preparation = text.split("\n\n")
            text = ingredients

    text = text.translate(table)
    f.close()

    if not include_stopwords:
        fs = open("recipes/stopwords.txt")
        stopwords = fs.read()
        stopwords = stopwords.translate(table)
        stopwords_list = stopwords.split()
        fs.close()

    words = []
    # Here text has been tokenized with the same regular expression used by TfidfVectorizer and CountVectorizer
    # in order to be consistent and to not have different set of tokens
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    for w in token_pattern.findall(string=text):
        if len(w) > 2 and (w != "ingredients" and w != "preparation"):
            if not include_stopwords:
                w = w.lower()
                if w not in stopwords_list:
                    if stem:
                        words.append(porter.stem(w).lower())
                    else:
                        words.append(w)
            else:
                if stem:
                    words.append(porter.stem(w).lower())
                else:
                    words.append(w)

    return words


def load_vocabulary(path: str, n_features: int, include_stopwords=True, stem=False, recipe_part="all"):

    add_info = path
    if not include_stopwords:
        add_info += "_SW"
    if stem:
        add_info += "_stem"
    add_info += "_" + recipe_part

    try:
        with open("vocabularies/vocabulary_" + n_features.__str__() + "_" + add_info + ".txt", "r") as v:
            sorted_common_voc = import_vocabulary(v.name.__str__())

    except IOError:
        sorted_common_voc = None

    return add_info, sorted_common_voc
