import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
import sklearn.metrics as metric
import gensim
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, trange
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
word_vectors_file = "data/GoogleNews-vectors-negative300.bin"
test_filenames = ['data/Grimms/emmood/the_turnip.emmood',
                  'data/Grimms/emmood/the_story_of_the_youth_who_went_forth_to_learn_what_fear_was.emmood',
                  'data/Grimms/emmood/22_the_riddle.emmood',
                  'data/Grimms/emmood/29_the_devil_with_the_three_golden_hairs.emmood',
                  'data/Grimms/emmood/the_golden_goose.emmood',
                  'data/Grimms/emmood/the_twelve_dancing_princesses.emmood',
                  'data/Grimms/emmood/73_the_wolf_and_the_fox.emmood', 'data/Grimms/emmood/briar_rose.emmood',
                  'data/Potter/emmood/the_roly-poly_pudding.emmood', 'data/HCAndersen/emmood/drop_wat.emmood',
                  'data/HCAndersen/emmood/lovelies.emmood', 'data/HCAndersen/emmood/last_dre.emmood',
                  'data/HCAndersen/emmood/bell.emmood', 'data/HCAndersen/emmood/races.emmood',
                  'data/HCAndersen/emmood/buckwhet.emmood', 'data/HCAndersen/emmood/heaven.emmood']

is_binary = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class EncoderRNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, classes):
        super(EncoderRNN, self).__init__()
        self.encoder = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                                     bidirectional=True).to(device)
        self.hidden_size = hidden_size
        self.classifier = torch.nn.Linear(hidden_size*2, classes)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.2, inplace=True)

    def forward(self, inputs):
        _, (hn, cn) = self.encoder(inputs)
        hn = self.dropout(self.sigmoid(hn)).view(-1, 1, self.hidden_size*2)
        logits = self.classifier(hn)

        return logits


class EncoderCNN(torch.nn.Module):
    def __init__(self, input_channel, max_length, batch_size, class_num):
        super(EncoderCNN, self).__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.class_num = class_num
        self.input_channel = input_channel
        self.softmax = torch.nn.Softmax(dim=1)

        self.conv_21 = torch.nn.Conv1d(input_channel, 1, kernel_size=2)
        self.conv_22 = torch.nn.Conv1d(input_channel, 1, kernel_size=2)

        self.conv_31 = torch.nn.Conv1d(input_channel, 1, kernel_size=3)
        self.conv_32 = torch.nn.Conv1d(input_channel, 1, kernel_size=3)

        self.conv_41 = torch.nn.Conv1d(input_channel, 1, kernel_size=4)
        self.conv_42 = torch.nn.Conv1d(input_channel, 1, kernel_size=4)

        self.maxp2 = torch.nn.MaxPool1d(max_length - 1, stride=1)
        self.maxp3 = torch.nn.MaxPool1d(max_length - 2, stride=1)
        self.maxp4 = torch.nn.MaxPool1d(max_length - 3, stride=1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.linear = torch.nn.Linear(6, self.class_num)

    def forward(self, _input):
        _input = _input.view(-1, self.input_channel, self.max_length)

        conv_21 = self.conv_21(_input)
        conv_22 = self.conv_22(_input)
        conv2 = torch.cat([conv_21, conv_22], 1)

        conv_31 = self.conv_31(_input)
        conv_32 = self.conv_32(_input)
        conv3 = torch.cat([conv_31, conv_32], 1)

        conv_41 = self.conv_41(_input)
        conv_42 = self.conv_42(_input)
        conv4 = torch.cat([conv_41, conv_42], 1)

        mconv2 = self.maxp2(conv2)
        mconv3 = self.maxp3(conv3)
        mconv4 = self.maxp4(conv4)

        conv = torch.cat([mconv2, mconv3, mconv4], 1)
        conv = torch.transpose(conv, 1, 2)

        sconv = self.sigmoid(conv)

        logits = self.linear(sconv)

        return logits


class EncoderTransformer(torch.nn.Module):

    def __init__(self, input_size, n_labels, n_head=8, n_layers=6):
        super(EncoderTransformer, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_labels = n_labels

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.n_head)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        self.classifier = torch.nn.Linear(self.input_size, self.n_labels)

    def forward(self, inputs):
        hidden_states = self.transformer_encoder(inputs)
        hidden_state = hidden_states[:, 0, :]

        logits = self.classifier(hidden_state)

        return logits


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


def word2vec(model, texts):

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
        words = filter_words(tknzr.tokenize(text), vocab)
        w_vevtors = model[words]

        word_vectors.append(w_vevtors)
        # word_vector = np.sum(w_vevtors, axis=0)
        # word_vectors.append(word_vector / len(w_vevtors))

    return word_vectors


def load_raw_data():
    paths = ["data/Grimms/emmood/", "data/Potter/emmood/", "data/HCAndersen/emmood/"]
    label_list = ["A", "D", "F", "H", "N", "Sa", "Su+", "Su-"]

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
                raw_sents.append(text_a.replace("\"", " "))
            c_index += 1

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
                test_raw_sents.append(text_a.replace("\"", " "))
            c_index += 1

    return raw_sents, train_labels, test_raw_sents, test_labels


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


def pad_data(input_data):
    pad_vector = np.zeros([1, 300])
    padded_data = []

    lengths = [len(s) for s in input_data]
    max_length = max(lengths)

    for sequence in input_data:
        length = len(sequence)
        resi = max_length - length

        zeros = np.array([pad_vector for _ in range(resi)]).reshape(-1, 300)

        sequence = np.concatenate((zeros, sequence), axis=0)

        padded_data.append(sequence)

    return max_length, padded_data


def build_dataset(w2v_model, train_data, train_labels, eval_data, eval_labels):
    raw_data = train_data + eval_data
    input_vectors = word2vec(w2v_model, raw_data)
    max_length, padded_data = pad_data(input_vectors)

    train_input_tensors = torch.tensor(padded_data[:len(train_labels)], dtype=torch.float)
    train_label_tensors = torch.tensor(train_labels, dtype=torch.long)

    eval_input_tensors = torch.tensor(padded_data[len(train_labels):], dtype=torch.float)
    eval_label_tensors = torch.tensor(eval_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_input_tensors, train_label_tensors)
    eval_dataset = TensorDataset(eval_input_tensors, eval_label_tensors)

    return max_length, train_dataset, eval_dataset


def train(train_dataset, encoder, epochs, batch_size, num_labels, lr=1e-3, weight_decay=1e-3):
    """ Train the model """

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    epochs_trained = 0
    train_iterator = trange(epochs_trained, int(epochs), desc="Epoch")

    loss_track = []
    encoder.to(device)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            encoder.zero_grad()

            logits = encoder(batch[0])
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), batch[1].view(-1))

            loss.backward()

            loss_track.append(loss.item())

            optimizer.step()

    return loss_track, encoder


def evaluate(test_dataset, encoder, batch_size, num_labels):
    eval_sampler = RandomSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=batch_size)
    preds = None
    encoder.to(device)
    for step, batch in enumerate(eval_dataloader):
        encoder.eval()
        batch = tuple(t.to(device) for t in batch)
        inputs = batch[0]

        logits = encoder(inputs).view(-1, num_labels)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[1].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, batch[1].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)

    return preds


if __name__ == "__main__":

    num_labels = 8
    batch_size = 16
    hidden_size = 64
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word_vectors_file, binary=True)
    train_data, train_labels, test_data, test_labels = load_raw_data()

    max_length, train_dataset, eval_dataset = build_dataset(w2v_model, train_data, train_labels, test_data, test_labels)

    encoder = EncoderRNN(input_size=300, hidden_size=hidden_size, classes=num_labels)
    # encoder = EncoderCNN(input_channel=300, max_length=max_length, batch_size=batch_size, class_num=num_labels)
    # encoder = EncoderTransformer(input_size=300, n_labels=num_labels, n_head=8, n_layers=6)
    losses, encoder = train(train_dataset, encoder, 6, batch_size=batch_size, num_labels=num_labels, lr=3e-4, weight_decay=2)
    print(losses)

    preds = evaluate(eval_dataset, encoder, batch_size=16, num_labels=num_labels)

    accuracy = metric.accuracy_score(test_labels, preds)
    precision = metric.precision_score(test_labels, preds, average='macro')
    recall = metric.recall_score(test_labels, preds, average='macro')
    f1 = metric.f1_score(test_labels, preds, average='macro')

    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", f1)
