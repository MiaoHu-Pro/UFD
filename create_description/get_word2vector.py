
import numpy as np
import torch


# EMBEDDING_DIM = 50
# glove = torchtext.vocab.GloVe(name="6B", dim=EMBEDDING_DIM)

def read_word_bag(in_path):
    """
    read training data (complex_triple2vector)
    :param in_path:
    :param all_data:  return data
    :return:
    """
    print("read word_bag \n")
    all_data = []
    all_word = {}

    try:
        fopen_in = open(in_path,  'r',encoding='utf-8')
    except IOError as err:
        print('file open error: {0}'.format(err))
    else:
        i = 0
        for eachLine in fopen_in:
            if eachLine:
                each = eachLine.strip()
                all_data.append(each)
                all_word[each] = i
                i += 1

        fopen_in.close()

    print("read train data over! ")
    # all_word_bag = all_data
    return all_data,all_word

def load_w2v_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    print("load bin vec \n")

    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        c = 0
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                c +=1
            else:

                f.read(binary_len)
    return word_vecs

# def load_golve_vec(word_list):
#
#     print("load bin vec \n")
#
#     word_vectors = {}
#     c = 0
#     for w in glove.itos:
#         if w in word_list:
#             word_vectors[w] = glove[w]
#             c += 1
#     # print(len(word_list), c, len(word_list) - c)
#
#     return word_vectors

# def word2vector(word_dict):
#     """
#
#     """
#     print("Word to vector ... \n")
#     # word2vector
#     word_to_idx = word_dict #数据集中所有的单词
#
#     pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict), EMBEDDING_DIM))
#
#     # word2vec = load_w2v_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
#     word2vec = load_golve_vec(word_to_idx)
#
#     for word, vector in word2vec.items():# 初始化每个词
#
#         pretrained_embeddings[word_to_idx[word]] = vector
#
#     #打印测试
#     # print("NULL -> ",word_to_idx['NULL'],pretrained_embeddings[word_to_idx['NULL']])
#     # print("contemporary -> ",pretrained_embeddings[30080])
#     # print("the - > ",pretrained_embeddings[12104])
#     # print("the - > ",pretrained_embeddings[word_to_idx['the']])
#
#     # singer_index = word_to_idx['singer']
#     # print(singer_index)
#     # print("singer - > ",pretrained_embeddings[singer_index])
#     # print("bryan - > ",pretrained_embeddings[25286])
#     #
#     pretrained_embeddings = torch.as_tensor(pretrained_embeddings)
#     return pretrained_embeddings.to(torch.float32)

# def get_word2vec(word_bag_path):
#
#     wb_list,all_word_dic = read_word_bag(word_bag_path)
#
#     pre_train_embeddings = word2vector(all_word_dic)
#
#     return all_word_dic,pre_train_embeddings
#
#     # print(wb_list)
#     # print(len(wb_list))

def get_sentence_init_embedding(pre_embeddings,word_bag,sentence_set):

    # print("get_sentence_init_embedding ... ")

    # print("\n",len(sentence_set))

    # print("sentence_set",sentence_set)
    length_list = [len(x) for x in sentence_set]
    # print(length_list)
    max_num_words = max(length_list) # get max number of words for each batch

    # print(max_num_words)#
    # time.sleep(3)

    sentence_word_index_set = []
    for i in range(len(sentence_set)):

        # 填充句子，没有达到最大长度的句子使用NULL填充。
        sentence_set[i] = sentence_set[i] + ["NULL"] * (max_num_words - len(sentence_set[i]))
        # sentence_word_index_set.append(pre_word2vec[[word_bag[x] for x in sentcence_set[i]]])
        # print(sentence_set[i])
        # time.sleep(5)
        sentence_word_index_set.append([word_bag[x] for x in sentence_set[i]])

    sentence_word_index_set = np.array(sentence_word_index_set).T # 专置

    # print(sentence_word_index_set)

    tensor_sentence_word_index_set = torch.as_tensor(sentence_word_index_set) # numpy 转为 tensor

    # print(tensor_sentence_word_index_set)

    init_embedding = pre_embeddings[tensor_sentence_word_index_set].view(max_num_words, len(sentence_set), -1)

    # print(x)
    # print(init_embedding.shape)
    # print("Finish get_sentence_init_embedding ...")
    return init_embedding

if __name__=="__main__":

    """word2vector"""
    pass
    # word_bag_path = "FB15K/word_bag.txt"
    #
    # word_bag, pre_embeddings = get_word2vec(word_bag_path) # 得到预训练词模型
    #
    # # pre_embeddings = torch.as_tensor(pre_word2vec)# numpy 转为 tensor
    #
    # # pre_embeddings = nn.Embedding(len(word_bag), 300)
    #
    # sentence_set = [['/m/06cx9', 'republic', 'form', 'of', 'state'],
    #                  ['/m/06cx9', 'has', 'a', 'relationship', 'of', 'government','form', 'of', 'government'],
    #                  ['countries', 'with', '/m/02lx0', '/m/06cx9', 'has', 'a','relationship', 'of', 'government', 'form', 'of'],
    #                  ['government', 'countries','with', '/m/06f32', '/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries','with', '/m/036b_'],
    #                  ['/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries', 'with', '/m/04hqz', '/m/06cx9', 'has', 'a', 'relationship', 'of', 'government', 'form', 'of', 'government', 'countries', 'with', '/m/016wzw']]
    #
    # get_sentence_init_embedding(pre_embeddings, word_bag, sentence_set)
    #
    # """
    # Next, put it into LSTM
    # lstm_out, self.hidden = self.lstm(x, self.hidden)
    # """




