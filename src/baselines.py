import gensim
import numpy as np
from sklearn.svm import LinearSVC
import sklearn.metrics as metric
from os import listdir
from os.path import isfile, join
from sklearn.naive_bayes import BernoulliNB

word_vectors_file = "../data/GoogleNews-vectors-negative300.bin"
test_filenames = ['../data/Grimms/emmood/the_turnip.emmood',
                  '../data/Grimms/emmood/the_story_of_the_youth_who_went_forth_to_learn_what_fear_was.emmood',
                  '../data/Grimms/emmood/22_the_riddle.emmood',
                  '../data/Grimms/emmood/29_the_devil_with_the_three_golden_hairs.emmood',
                  '../data/Grimms/emmood/the_golden_goose.emmood',
                  '../data/Grimms/emmood/the_twelve_dancing_princesses.emmood',
                  '../data/Grimms/emmood/73_the_wolf_and_the_fox.emmood', 'data/Grimms/emmood/briar_rose.emmood',
                  '../data/Potter/emmood/the_roly-poly_pudding.emmood', 'data/HCAndersen/emmood/drop_wat.emmood',
                  '../data/HCAndersen/emmood/lovelies.emmood', 'data/HCAndersen/emmood/last_dre.emmood',
                  '../data/HCAndersen/emmood/bell.emmood', 'data/HCAndersen/emmood/races.emmood',
                  '../data/HCAndersen/emmood/buckwhet.emmood', 'data/HCAndersen/emmood/heaven.emmood']

is_binary = True


class Lang:
    # statistic for natural sentence, including word to index, word count,etc
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def word2vec(w2v, texts):
    model = w2v
    vocab = model.wv.vocab

    def filter_words(words, vocab):
        new_words = []

        # print(words)
        for w in words:
            if w in vocab:
                new_words.append(w)

        if len(new_words) == 0:
            new_words = ['0']
        return new_words

    word_vectors = []
    for text in texts:
        words = filter_words(text.split(), vocab)
        w_vevtors = model[words]

        word_vector = np.sum(w_vevtors, axis=0)
        word_vectors.append(word_vector / len(w_vevtors))

    return word_vectors


def load_data():
    paths = ["data/Grimms/emmood/", "data/Potter/emmood/", "data/HCAndersen/emmood/"]

    binary_label_list = ["N", "E"]
    label_list = ["A", "D", "F", "H", "N", "Sa", "Su+", "Su-"]

    label_list = binary_label_list if is_binary else label_list

    total_filenames = []
    for p in paths:
        onlyfiles = [p + f for f in listdir(p) if isfile(join(p, f))]

        total_filenames += onlyfiles

    raw_sents = []
    chapter_indices = []
    train_labels = []

    c_index = 0
    for data_dir in total_filenames:
        if data_dir in test_filenames:
            continue
        with open(data_dir, "r") as f:
            lines = f.readlines()

            for (i, line) in enumerate(lines):
                a, b, c, d = line.split('\t')
                text_a = d[:-1]
                label = b.split(':')[0]

                # label = 1 if label is not "N" else 0
                label = label_list.index(label)

                chapter_indices.append(c_index)
                train_labels.append(label)
                raw_sents.append(text_a)
            c_index += 1
    train_n_count = train_labels.count("N") / len(train_labels)

    test_raw_sents = []
    test_chapter_indices = []
    test_labels = []

    c_index = 0
    for data_dir in test_filenames:
        with open(data_dir, "r") as f:
            lines = f.readlines()

            for (i, line) in enumerate(lines):
                a, b, c, d = line.split('\t')
                text_a = d[:-1]
                label = b.split(':')[0]

                # label = 1 if label is not "N" else 0
                label = label_list.index(label)

                test_chapter_indices.append(c_index)
                test_labels.append(label)
                test_raw_sents.append(text_a)
            c_index += 1

    test_n_count = test_labels.count("N") / len(test_labels)

    return raw_sents, train_labels, test_raw_sents, test_labels


def LinearSVM(train_sents, train_labels, w2v, tol=1e-5):
    clf = LinearSVC(random_state=0, tol=tol)

    word_vectors = word2vec(w2v, train_sents)

    clf.fit(word_vectors, train_labels)

    # print(clf.predict(np.array(vectors[0]).reshape(1, -1)))
    return clf


def naive_bayes(train_sents, train_labels):
    nb = BernoulliNB()

    nb.fit(train_sents, train_labels)

    return nb


def build_dict(sents, lang):
    for s in sents:
        lang.addSentence(s)

    return lang


def text2vec(sents, dict):
    vectors = []

    for s in sents:
        lis_s = s.split()
        # vectors.append([dict.word2index[w] for w in lis_s])
        vectors.append([1 if e in lis_s else 0 for e in dict.word2count])

    return vectors


if __name__ == "__main__":
    train_file = "data/sst/sst_train.txt"
    dev_file = "data/sst/sst_dev.txt"
    test_file = "data/holmes_dataset_adjust.txt"
    # w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word_vectors_file, binary=True)

    train_data, train_labels, test_data, test_labels = load_data()

    # sents_vectors = word2vec(w2v_model, test_data)
    # svm = LinearSVM(train_data, train_labels, w2v_model)
    #
    # print(svm.score(sents_vectors, test_labels))
    #
    # preds = svm.predict(sents_vectors)
    #
    # accuracy = metric.accuracy_score(test_labels, preds)
    # precision = metric.precision_score(test_labels, preds, average='weighted')
    # recall = metric.recall_score(test_labels, preds, average='weighted')
    # f1 = metric.f1_score(test_labels, preds, average='weighted')
    #
    # print("accuracy: ", accuracy)
    # print("precision: ", precision)
    # print("recall: ", recall)
    # print("F1: ", f1)
    #
    # with open("svm_results_multi", "w") as f:
    #     for (p, t) in zip(preds, test_labels):
    #         line = str(p) + "---" + str(t) + "\n"
    #         f.write(line)

    lang = Lang("sa")

    lang = build_dict(train_data + test_data, lang)
    train_sents = text2vec(train_data, lang)
    test_sents = text2vec(test_data, lang)

    clf = naive_bayes(train_sents, train_labels)

    preds = clf.predict(test_sents)
