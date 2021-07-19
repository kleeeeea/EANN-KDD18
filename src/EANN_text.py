from vzmi.mlx.common.Variable.GenericContainer.List.NestedList.OfGivenShape.MultiDimensionalArray.Dense.Base._OfBaseValueDistribution.io_ops.to_List import DenseMultiDimensionalArray_to_List
from vzmi.mlx.common.runtime_information.OperatingSystem.Base._components.List_of_GPUs.Available._components.Indicies._bundle import LocalOperatingSystem_set_List_of_AvailableGPUs_Indicies
from vzmi.mlx.data_science.Prediction.ForMultiDimensionalArray.Base._components.Model.Base.Pytorch._components.Tensor._components.Device.set_op import PytorchModel_Tensor_set_Device
from vzmi.mlx.data_science.Prediction.ForMultiDimensionalArray.Base._components.Model.Base.Pytorch._components.Tensor._components.Value.get_ import PytorchModel_Tensor_get_Value
from vzmi.mlx.io.local_file_system.File.Base._components.Path.Base._constants import DATA_INPUT_TEXTS_ROOT__LOCAL_DIRECTORY_PATH
from vzmi.mlx.io.local_file_system.File.Directory.Base.create import create_LocalDirectory
from vzmi.mlx.software_engineering.viewing.Logger.log_op import Logger_log

LocalOperatingSystem_set_List_of_AvailableGPUs_Indicies(1)

import copy
import pickle as pickle
from random import sample

import numpy as np
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from vzmi.mlx.io.local_file_system.File.Base._components.Path.Base._constants import LOGS_ROOT_LOCAL_PATH
from vzmi.mlx.software_engineering.viewing.Logger._backends.Local._bundle import filter_warnings

filter_warnings()
IS_VERBOSE = True
# encoding=utf-8
import re
from collections import defaultdict

import jieba
import pandas as pd

data_root = DATA_INPUT_TEXTS_ROOT__LOCAL_DIRECTORY_PATH + '/weibo/'


def stopwordslist(filepath=f'{data_root}' + '/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r').readlines():
        line = line.encode("utf-8").strip()
        stopwords[line] = 1
    # stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub("[，。 :,.；|-“”—_/nbsp+&;@、《》～（）()#O！：【】]", "", string)
    return string.strip().lower()


# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
#
# def read_image():
#     image_list = {}
#     file_list = [f'{data_root}' + '/nonrumor_images/', f'{data_root}' + '/rumor_images/']
#     for path in file_list:
#         data_transforms = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#         for i, filename in enumerate(os.listdir(path)):  # assuming gif
#
#             # print(filename)
#             try:
#                 im = Image.open(path + filename).convert('RGB')
#                 im = data_transforms(im)
#                 # im = 1
#                 image_list[filename.split('/')[-1].split(".")[0].lower()] = im
#             except Exce:
#                 print(filename)
#     print(("image length " + str(len(image_list))))
#     # print("image names are " + str(image_list.keys()))
#     return image_list


def write_txt(data):
    f = open(data_root + "top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l + "\n")
        f.write(b"\n")
        f.write(b"\n")
    f.close()


text_dict = {}


def write_data(flag, image, text_only):
    def read_post(flag1):
        stop_words = stopwordslist()
        pre_path = data_root + "tweets/"
        file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt",
                     pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]
        if flag1 == "train":
            id1 = pickle.load(open(data_root + "train_id.pickle", 'rb'))
        elif flag1 == "validate":
            id1 = pickle.load(open(data_root + "validate_id.pickle", 'rb'))
        elif flag1 == "test":
            id1 = pickle.load(open(data_root + "test_id.pickle", 'rb'))
        else:
            raise NotImplementedError
        post_content1 = []
        # labels = []
        # image_ids = []
        # twitter_ids = []
        data = []
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        # key = -1
        map_id = {}
        top_data = []
        for k, f in enumerate(file_list):

            f = open(f, 'rb')
            if (k + 1) % 2 == 1:
                label = 0  ### real is 0
            else:
                label = 1  ####fake is 1

            # twitter_id = 0
            line_data = []
            # top_line_data = []

            for i, l in enumerate(f.readlines()):
                # key += 1

                # if int(key /3) in index:
                # print(key/3)
                # continue
                l = str(l, "utf-8")
                if (i + 1) % 3 == 1:
                    line_data = []
                    twitter_id = l.split('|')[0]
                    line_data.append(twitter_id)

                if (i + 1) % 3 == 2:
                    line_data.append(l.lower())

                if (i + 1) % 3 == 0:
                    l = clean_str_sst(l)

                    seg_list = jieba.cut_for_search(l)
                    new_seg_list = []
                    for word in seg_list:
                        if word not in stop_words:
                            new_seg_list.append(word)

                    clean_l = " ".join(new_seg_list)
                    if len(clean_l) > 10 and line_data[0] in id1:
                        post_content1.append(l)
                        line_data.append(l)
                        line_data.append(clean_l)
                        line_data.append(label)
                        event = int(id1[line_data[0]])
                        if event not in map_id:
                            map_id[event] = len(map_id)
                            event = map_id[event]
                        else:
                            event = map_id[event]

                        line_data.append(event)

                        data.append(line_data)

            f.close()
            # print(data)
            #     return post_content

        data_df = pd.DataFrame(np.array(data), columns=column)
        write_txt(top_data)

        return post_content1, data_df

    post_content, post = read_post(flag)
    print(("Original post length is " + str(len(post_content))))
    print(("Original data frame is " + str(post.shape)))

    # def find_most(db):
    #     maxcount = max(len(v) for v in list(db.values()))
    #     return [k for k, v in list(db.items()) if len(v) == maxcount]
    #
    # def select(train, selec_indices):
    #     temp = []
    #     for i in range(len(train)):
    #         ele = list(train[i])
    #         temp.append([ele[i] for i in selec_indices])
    #         #   temp.append(np.array(train[i])[selec_indices])
    #     return temp

    #     def balance_event(data, event_list):
    #         id = find_most(event_list)[0]
    #         remove_indice = random.sample(range(min(event_list[id]), \
    #                                             max(event_list[id])), int(len(event_list[id]) * 0.9))
    #         select_indices = np.delete(range(len(data[0])), remove_indice)
    #         return select(data, select_indices)

    def paired(text_only1=False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event = []
        label = []
        post_id = []
        image_id_list = []
        # image = []

        image_id = ""
        for i, id1 in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only1 or image_id in image:
                if not text_only1:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id1)

                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int)
        ordered_event = np.array(ordered_event, dtype=np.int)

        print(("Label number is " + str(len(label))))
        print(("Rummor number is " + str(sum(label))))
        print(("Non rummor is " + str(len(label) - sum(label))))

        #
        # if flag == "test":
        #     y = np.zeros(len(ordered_post))
        # else:
        #     y = []

        data = {"post_text"    : np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image"        : ordered_image, "social_feature": [],
                "label"        : np.array(label),
                "event_label"  : ordered_event, "post_id": np.array(post_id),
                "image_id"     : image_id_list}
        # print(data['image'][0])

        print(("data size is " + str(len(data["post_text"]))))

        return data

    paired_data = paired(text_only)

    print(("paired post length is " + str(len(paired_data["post_text"]))))
    print(("paried data has " + str(len(paired_data)) + " dimension"))
    return paired_data


def load_data(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_data():
    text_only = True
    # text_only = False
    print("Text only")
    image_list = []

    train_data = write_data("train", image_list, text_only)
    valiate_data = write_data("validate", image_list, text_only)
    test_data = write_data("test", image_list, text_only)

    print("loading data...")
    # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
    vocab, all_text = load_data(train_data, valiate_data, test_data)
    # print(str(len(all_text)))

    print(("number of sentences: " + str(len(all_text))))
    print(("vocab size: " + str(len(vocab))))
    max_l = len(max(all_text, key=len))
    print(("max sentence length: " + str(max_l)))

    #
    #
    word_embedding_path = data_root + "w2v.pickle"

    w2v = pickle.load(open(word_embedding_path, 'rb'), encoding='latin1')
    # print(temp)
    # #
    print("word2vec loaded!")
    print(("num words already in word2vec: " + str(len(w2v))))
    # w2v = add_unknown_words(w2v, vocab)
    # file_path = data_root + "event_clustering.pickle"
    # if not os.path.exists(file_path):
    #     train = []
    #     for l in train_data["post_text"]:
    #         line_data = []
    #         for word in l:
    #             line_data.append(w2v[word])
    #         line_data = np.matrix(line_data)
    #         line_data = np.array(np.mean(line_data, 0))[0]
    #         train.append(line_data)
    #     train = np.array(train)
    #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
    #     cluster.fit(train)
    #     y = np.array(cluster.labels_)
    #     pickle.dump(y, open(file_path, 'wb+'))
    # else:
    # y = pickle.load(open(file_path, 'rb'))
    # print("Event length is " + str(len(y)))
    # center_count = {}
    # for k, i in enumerate(y):
    #     if i not in center_count:
    #         center_count[i] = 1
    #     else:
    #         center_count[i] += 1
    # print(center_count)
    # train_data['event_label'] = y

    #
    print("word2vec loaded!")
    print(("num words already in word2vec: " + str(len(w2v))))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    # # rand_vecs = {}
    # # add_unknown_words(rand_vecs, vocab)
    W2 = {}
    # rand_vecs =
    w_file = open(data_root + "word_embedding.pickle", "wb")
    pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    w_file.close()
    return train_data, valiate_data, test_data


# if __name__ == "__main__":
#     image_list = read_image()
#
#     train_data = write_data("train", image_list)
#     valiate_data = write_data("validate", image_list)
#     test_data = write_data("test", image_list)
#
#     # print("loading data...")
#     # # w2v_file = '../Data/GoogleNews-vectors-negative300.bin'
#     vocab, all_text = load_data(train_data, test_data)
#     #
#     # # print(str(len(all_text)))
#     #
#     # print("number of sentences: " + str(len(all_text)))
#     # print("vocab size: " + str(len(vocab)))
#     # max_l = len(max(all_text, key=len))
#     # print("max sentence length: " + str(max_l))
#     #
#     # #
#     # #
#     # word_embedding_path = data_root + "word_embedding.pickle"
#     # if not os.path.exists(word_embedding_path):
#     #     min_count = 1
#     #     size = 32
#     #     window = 4
#     #
#     #     w2v = Word2Vec(all_text, min_count=min_count, size=size, window=window)
#     #
#     #     temp = {}
#     #     for word in w2v.wv.vocab:
#     #         temp[word] = w2v[word]
#     #     w2v = temp
#     #     pickle.dump(w2v, open(word_embedding_path, 'wb+'))
#     # else:
#     #     w2v = pickle.load(open(word_embedding_path, 'rb'))
#     # # print(temp)
#     # # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # # w2v = add_unknown_words(w2v, vocab)
#     # Whole_data = {}
#     # file_path = data_root + "event_clustering.pickle"
#     # # if not os.path.exists(file_path):
#     # #     data = []
#     # #     for l in train_data["post_text"]:
#     # #         line_data = []
#     # #         for word in l:
#     # #             line_data.append(w2v[word])
#     # #         line_data = np.matrix(line_data)
#     # #         line_data = np.array(np.mean(line_data, 0))[0]
#     # #         data.append(line_data)
#     # #
#     # #     data = np.array(data)
#     # #
#     # #     cluster = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='complete')
#     # #     cluster.fit(data)
#     # #     y = np.array(cluster.labels_)
#     # #     pickle.dump(y, open(file_path, 'wb+'))
#     # # else:
#     # # y = pickle.load(open(file_path, 'rb'))
#     # # print("Event length is " + str(len(y)))
#     # # center_count = {}
#     # # for k, i in enumerate(y):
#     # #     if i not in center_count:
#     # #         center_count[i] = 1
#     # #     else:
#     # #         center_count[i] += 1
#     # # print(center_count)
#     # # train_data['event_label'] = y
#     #
#     # #
#     # print("word2vec loaded!")
#     # print("num words already in word2vec: " + str(len(w2v)))
#     # add_unknown_words(w2v, vocab)
#     # W, word_idx_map = get_W(w2v)
#     # # # rand_vecs = {}
#     # # # add_unknown_words(rand_vecs, vocab)
#     # W2 = rand_vecs = {}
#     # pickle.dump([W, W2, word_idx_map, vocab, max_l], open(data_root + "word_embedding.pickle", "wb"))
#     # print("dataset created!")


# from logger import Logger

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print(('TEXT: %d, labe: %d, Event: %d'
               % (len(self.text), len(self.label), len(self.event_label))))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):
    # def __init__(self, lambd):
    # self.lambd = lambd

    # @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x):
    # noinspection PyCallingNonCallable
    return ReverseLayerF()(x)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, W, vocab_size):
        super(CNN_Fusion, self).__init__()
        # self.args = args

        self.event_num = 10

        vocab_size = vocab_size
        dim = 32
        # embed_dim
        emb_dim = dim

        # C = class_num
        self.hidden_size = 32
        self.lstm_size = dim
        self.social_size = 19

        # TEXT RNN

        self.embed = nn.Embedding(vocab_size, emb_dim)
        # noinspection PyArgumentList
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        ### TEXT CNN
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])

        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(0.5)

        # IMAGE
        # hidden_size = hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ####Image and Text Classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(self.hidden_size, 2))
        self.modal_classifier.add_module('m_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (PytorchModel_Tensor_get_Value(torch.zeros(1, batch_size, self.lstm_size)),
                PytorchModel_Tensor_get_Value(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, mask):
        #########CNN##################
        global IS_VERBOSE
        if IS_VERBOSE:
            Logger_log(text.shape)
            Logger_log(mask.shape)
        text = self.embed(text)
        if IS_VERBOSE:
            Logger_log(text.shape)
        text = text * mask.unsqueeze(2).expand_as(text)
        if IS_VERBOSE:
            Logger_log(text.shape)
        text = text.unsqueeze(1)
        if IS_VERBOSE:
            Logger_log(text.shape)
        texts = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        if IS_VERBOSE:
            Logger_log([text.shape for text in texts])
        if IS_VERBOSE:
            Logger_log([cv for cv in self.convs])

        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        texts = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in texts]
        if IS_VERBOSE:
            Logger_log([text.shape for text in texts])
        text = torch.cat(texts, 1)
        if IS_VERBOSE:
            Logger_log(text.shape)
        text = F.relu(self.fc1(text))
        if IS_VERBOSE:
            Logger_log(text.shape)

        # text = self.dropout(text)

        ### Class
        # class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        if IS_VERBOSE:
            Logger_log(class_output.shape)

        ## Domain
        reverse_feature = grad_reverse(text)
        if IS_VERBOSE:
            Logger_log(reverse_feature.shape)

        domain_output = self.domain_classifier(reverse_feature)
        if IS_VERBOSE:
            Logger_log(domain_output.shape)

        IS_VERBOSE = False
        return class_output, domain_output


# def PytorchModel_Tensor_get_Value(x):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print(("length is " + str(len(train[i]))))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def make_weights_for_balanced_classes(event, nclasses=15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        # noinspection PyTypeChecker
        weight[idx] = weight_per_class[val]
    return weight


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(list(range(whole_len)), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print(("train data size is " + str(len(train[3]))))
    # print()

    validation = select(train, np.delete(list(range(len(train[0]))), train_indices))
    print(("validation size is " + str(len(validation[3]))))
    print("train and validation data set has been splited")

    return train_data, validation


def main():
    print('loading data')
    #    dataset = DiabetesDataset(roottraining_file)
    #    train_loader = DataLoader(dataset=dataset,
    #                              batch_size=32,
    #                              shuffle=True,
    #                              num_workers=2)

    # MNIST Dataset
    train, validation, test, W, max_len, vocab_size = load_data_from_args()

    # train, validation = split_train_validation(train,  1)

    # weights = make_weights_for_balanced_classes(train[-1], 15)
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)  # not used

    # Data Loader (Input Pipeline)
    batch_size = 100
    # batch_size
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    print('building model')
    model = CNN_Fusion(W, vocab_size)

    if torch.cuda.is_available():
        Logger_log("CUDA")
        model.cuda()
    else:
        Logger_log("no CUDA")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()

    # Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = torch.optim.Adam([p for p in list(model.parameters()) if p.requires_grad],
                                 lr=0.001)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
    # lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size= 10, gamma= 1)

    # iter_per_epoch = len(train_loader)
    print(("loader size " + str(len(train_loader))))
    best_validate_acc = 0.000
    # best_loss = 100
    best_validate_dir = ''

    print('training model')
    # adversarial = True
    # Train the Model
    num_epochs = 100
    # .num_epochs
    for epoch in range(num_epochs):

        # p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001

        optimizer.lr = lr
        # rgs.lambd = lambd

        # start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        # domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        # test_acc_vector = []
        vali_cost_vector = []
        # test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_mask, train_labels, event_labels = \
                PytorchModel_Tensor_set_Device(train_data[0]), PytorchModel_Tensor_set_Device(train_data[1]), \
                PytorchModel_Tensor_set_Device(train_labels), PytorchModel_Tensor_set_Device(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model.__call__(train_text, train_mask)
            # ones = torch.ones(text_output.size(0))
            # ones_label = PytorchModel_Tensor_get_Value(ones.type(torch.LongTensor))
            # zeros = torch.zeros(image_output.size(0))
            # zeros_label = PytorchModel_Tensor_get_Value(zeros.type(torch.LongTensor))

            # modal_loss = criterion(text_output, ones_label)+ criterion(image_output, zeros_label)

            # noinspection PyTypeChecker
            class_loss = criterion(class_outputs, train_labels)
            # noinspection PyTypeChecker
            domain_loss = criterion(domain_outputs, event_labels)
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            # cross_entropy = True

            # if True:
            accuracy = (train_labels == argmax.squeeze()).float().mean()
            # else:
            #     _, labels = torch.max(train_labels, 1)
            #     accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.data.item())
            # domain_cost_vector.append(domain_loss.item())
            cost_vector.append(loss.data.item())
            acc_vector.append(accuracy.data.item())
            # if i == 0:
            #     train_score = to_np(class_outputs.squeeze())
            #     train_pred = to_np(argmax.squeeze())
            #     train_true = to_np(train_labels.squeeze())
            # else:
            #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
            #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
            #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_mask, validate_labels, event_labels = \
                PytorchModel_Tensor_set_Device(validate_data[0]), PytorchModel_Tensor_set_Device(validate_data[1]), \
                PytorchModel_Tensor_set_Device(validate_labels), PytorchModel_Tensor_set_Device(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            # noinspection PyTypeChecker
            vali_loss = criterion(validate_outputs, validate_labels)
            # domain_loss = criterion(domain_outputs, event_labels)
            # _, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print(('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, validate loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
               % (
                       epoch + 1, num_epochs,
                       DenseMultiDimensionalArray_to_List(np.mean(cost_vector)),
                       DenseMultiDimensionalArray_to_List(np.mean(class_cost_vector)),
                       np.mean(vali_cost_vector).item(),
                       np.mean(acc_vector).item(),
                       validate_acc.item(),
               )))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            output_file = f'{LOGS_ROOT_LOCAL_PATH}''/eann/output/'
            create_LocalDirectory(output_file)
            best_validate_dir = output_file + str(epoch + 1) + '_text.pkl'
            torch.save(model.state_dict(), best_validate_dir)

    # duration = time.time() - start_time
    # print ('Epoch: %d, Mean_Cost: %.4f, Duration: %.4f, Mean_Train_Acc: %.4f, Mean_Test_Acc: %.4f'
    # % (epoch + 1, np.mean(cost_vector), duration, np.mean(acc_vector), np.mean(test_acc_vector)))
    # best_validate_dir = output_file + 'baseline_text_weibo_GPU2_out.' + str(20) + '.pkl'

    # Test the Model
    print('testing model')
    model = CNN_Fusion(W, vocab_size)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
        Logger_log('cuda')
    else:
        Logger_log('no cuda')

    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_mask, test_labels = PytorchModel_Tensor_set_Device(test_data[0]), \
                                            PytorchModel_Tensor_set_Device(test_data[1]), \
                                            PytorchModel_Tensor_set_Device(test_labels)
        test_outputs, _ = model(test_text, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    # test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    # test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    # test_recall = metrics.recall_score(test_true, test_pred, average='macro')

    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print(("Classification Acc: %.4f, AUC-ROC: %.4f"
           % (test_accuracy, test_aucroc)))
    print(("Classification report:\n%s\n"
           % (metrics.classification_report(test_true, test_pred, digits=3))))
    print(("Classification confusion matrix:\n%s\n"
           % test_confusion_matrix))

    print('Saving results')


def word2vec(post, word_id_map, sequence_len=28):
    # W
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        # seq_len = len(sentence) - 1
        mask_seq = np.zeros(sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask


def load_data_from_args():
    train, validate, test = get_data()
    # print(train[4][0])
    word_vector_path = f'{data_root}' + '/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f)  # W, W2, word_idx_map, vocab
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    vocab_size = len(vocab)
    # sequence_len = max_len
    print("translate data to embedding")

    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, max_len)
    # , W
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, max_len)
    # , W
    test['post_text'] = word_embedding
    test['mask'] = mask
    # test[-2]= transform(test[-2])
    word_embedding, mask = word2vec(train['post_text'], word_idx_map, max_len)
    # , W
    train['post_text'] = word_embedding
    train['mask'] = mask
    print(("sequence length " + str(max_len)))
    print(("Train Data Size is " + str(len(train['post_text']))))
    print("Finished loading data ")
    return train, validate, test, W, max_len, vocab_size


def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    # print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix


def driver():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    # parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    # parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    #
    # parser.add_argument('--static', type=bool, default=True, help='')
    # parser.add_argument('--sequence_length', type=int, default=28, help='')
    # parser.add_argument('--class_num', type=int, default=2, help='')
    # parser.add_argument('--hidden_dim', type=int, default=32, help='')
    # parser.add_argument('--embed_dim', type=int, default=32, help='')
    # parser.add_argument('--vocab_size', type=int, default=300, help='')
    # parser.add_argument('--dropout', type=int, default=0.5, help='')
    # parser.add_argument('--filter_num', type=int, default=20, help='')
    # parser.add_argument('--lambd', type=int, default=1, help='')
    # parser.add_argument('--text_only', type=bool, default=True, help='')
    #
    # #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    # #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    # #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    # #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    # #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    # parser.add_argument('--d_iter', type=int, default=3, help='')
    # parser.add_argument('--batch_size', type=int, default=100, help='')
    # parser.add_argument('--num_epochs', type=int, default=100, help='')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    # parser.add_argument('--event_num', type=int, default=10, help='')

    # test = f'{data_root}' + '/test.pickle'
    # output = f'{LOGS_ROOT_LOCAL_PATH}''/eann/output/'
    # args = parser.parse_args([
    #         f'{data_root}' + '/train.pickle', test, output
    # ])
    #    print(args)
    main()


if __name__ == '__main__':
    from vzmi.mlx_dynamically_typed.common.runtime_information.SourceCodeInterpreter.Base._components.CommandlineArgument.io_op.to_FunctionArguments._alias.do_and_run import \
        do_CommandlineArgument_to_FunctionArguments_and_run

    do_CommandlineArgument_to_FunctionArguments_and_run(driver)
