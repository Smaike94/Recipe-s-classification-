# Recipes-classification

This project aim to correctly classify recipes to their country. 
The task regards a multiclass classification problem so properly models have been tested. 
Files log_reg_clf.py, mlp_cls.py, svm_clf.py, naive_bayes_clf.py include all the code for testing accuracy and performance for related model.

The way used for threating text into numbers is BoW, i.e. Bag of Words. So, *Vocabulary.py* aims to create vocabulary from training text, that will be used as meter
for counting the occurences of vocabulary's words in each recipe. 
Then vocabulary created will be used from *DataSets.py* that will instanstiate numerical dataset suitbale for feeding machine learning models.
