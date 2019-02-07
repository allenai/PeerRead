"""
  train linear classifier using CV and find the best model on dev set
"""

import sys, pickle,random
from collections import Counter
from sklearn import datasets,preprocessing,model_selection
from sklearn import linear_model,svm,neural_network,ensemble

def get_data(features_if, scale=False, n_features = None):
  data = datasets.load_svmlight_file(features_if, n_features=n_features)
  if scale:
    new_x = preprocessing.scale(data[0].toarray())
    return new_x, data[1]
  else:
    return data[0], data[1]


def main(args, scale=False):
    if len(args) < 5:
        print("Usage:",args[0],"<train if> <dev if> <test if> <of>")
        return -1

    ###########################
    # data loading
    ###########################
    n_features = sum(1 for line in open(args[5])) #None #train_features.shape[1]
    train_features, train_labels = get_data(args[1], scale=scale,n_features=n_features)
    dev_features, dev_labels = get_data(args[2], scale=scale, n_features=n_features)
    test_features, test_labels = get_data(args[3], scale=scale, n_features=n_features)


    ###########################
    # majority
    ###########################
    train_counter = Counter(train_labels)
    dev_counter = Counter(dev_labels)
    test_counter = Counter(test_labels)
    print(train_counter, train_features.shape)
    print(dev_counter, dev_features.shape)
    print(test_counter, test_features.shape)
    print("Train majority: {}, Dev majority: {} Test majorit: {}".format(
      round(100.0*train_counter[0]/(train_counter[0]+train_counter[1]),3),
      round(100.0*dev_counter[0]/(dev_counter[0]+dev_counter[1]),3),
      round(100.0*test_counter[0]/(test_counter[0]+test_counter[1]),3)))


    ###########################
    #classifiers
    ###########################
    clfs = []
    best_classifier = None
    best_v = 0
    for c in [.1, .25, .5, 1.0]:
      for clf in [
          linear_model.LogisticRegression(C=c, dual=True),
          linear_model.LogisticRegression(C=c, penalty='l1'),
          svm.SVC(kernel='rbf', C=c)]:
        clfs.append(clf)
    clfs += [
      neural_network.MLPClassifier(alpha=1),
      ensemble.AdaBoostClassifier()]
    random.shuffle(clfs)
    print('Total number of classifiers',len(clfs))

    ###########################
    # training (CV) and testing
    ###########################
    for cidx, clf in enumerate(clfs):
      scores = model_selection.cross_val_score(clf, train_features, train_labels, cv=5, n_jobs=8)
      v = sum(scores)*1.0/len(scores)
      if v > best_v:
        #print("New best v!",v*100.0,clf)
        best_classifier = clf
        best_v = v

    print("Best v:",best_v*100.0,", Best clf: ",best_classifier)
    best_classifier.fit(train_features, train_labels)

    # train
    train_y_hat = best_classifier.predict(train_features)
    train_score = 100.0 * sum(train_labels == train_y_hat) / len(train_y_hat)
    print('Train accuracy: %.2f in %d examples' %(round(train_score,3), sum(train_labels)))
    # dev
    dev_y_hat = best_classifier.predict(dev_features)
    dev_score = 100.0 * sum(dev_labels == dev_y_hat) / len(dev_y_hat)
    print('Dev accuracy: %.2f in %d examples' %(round(dev_score,3), sum(dev_labels)))
    # test
    test_y_hat = best_classifier.predict(test_features)
    test_score = 100.0 * sum(test_labels == test_y_hat) / len(test_y_hat)
    print('Test accuracy: %.2f in %d examples' %(round(test_score,3),sum(test_labels)))


if __name__ == "__main__": sys.exit(main(sys.argv))
