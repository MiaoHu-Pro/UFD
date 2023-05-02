
import numpy as np
import pandas as pd
from create_description.utilities import Enti, entity_text_process, relation_text_process

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import threading

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        # time.sleep(2)
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


def read_ent_rel_2id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split('\t')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s)
            data_id.append(id_list)
    data = np.array(data_id)
    return data


def read_data2id(data_id_paht):
    data = pd.read_csv(data_id_paht)  #
    data = np.array(data)
    data_id = []
    for i in range(len(data)):
        _tmp = data[i][0]
        tmp = _tmp.split(' ')
        if tmp:
            id_list = []
            for s in tmp:
                id_list.append(s)
            data_id.append(id_list)
    data = np.array(data_id)
    return data


def read_entity_description(entity_description):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_description, encoding='utf-8')

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()
    return x_obj


def set_entity_description_obj(entity_des):
    all_entity_description_list = []
    entity_description_list = []

    # new_word_bag = []
    # new_word_bag.append('NULL')
    # new_word_bag += ['has', 'a', 'relationship', 'of', 'with', 'which', 'is', 'between', 'and']

    for i in range(len(entity_des)):

        print(i)
        entity_id = entity_des[i][0]
        symbol = entity_des[i][1]
        name = entity_des[i][2]
        mention = "that is " + entity_des[i][3]
        import ast
        neighbours_data = ast.literal_eval(entity_des[i][4])  # "list" -> list

        if len(neighbours_data) == 0:
            neighbours = ['the entity has not neighbours']
        else:
            neighbours = neighbours_data
        # print(len(neighbours))
        # print(neighbours)

        id2vector = np.random.rand(10)

        # en_des = str(symbol) + '$' + str(name) + '$' + str(mention) + '$' + str(neighbours)
        en_des = str(symbol) + '$' + str(mention) + '$' + str(neighbours)

        entity_des_word_list = entity_text_process(en_des)  # get entity des 's word list

        entity = Enti(_id=entity_id, _symbol=symbol, _label=name, _mention=mention, _neighbours=neighbours,
                      _entity2vec=id2vector, _entity_des_word_list=entity_des_word_list)

        en_des_word_list = entity.get_entity_description()
        all_entity_description_list.append(en_des_word_list)

        # 记录词表
        # new_word_bag += entity_des_word_list
        # new_word_bag = list(set(new_word_bag))

        entity_description_list.append(entity)

    print("len(all_entity_description_list) : \n", len(all_entity_description_list))

    # word_bag_path = "./benchmarks/FB15K/word_bag.txt"
    # word_bag, all_word_dic = read_word_bag(word_bag_path)
    # pre_word_embedding = word2vector(all_word_dic)
    word_bag = []
    all_word_dic = []
    pre_word_embedding = []
    print("len(word_bag)", len(word_bag))

    return entity_description_list, all_entity_description_list, word_bag, all_word_dic, pre_word_embedding


# def get_triples_description(train2id, relation2id, entity_description_obj, word_bag, pre_word_embedding):
#     print("get_triples_description - begin ... \n")
#
#     """
#     train2id : the index of train data
#     relation2id: the index of relation
#     entity_description_obj : entity object which contains id, symbol, name, description, neighbours.
#     word_bag : the number of words of relation and entity
#     pre_word_embedding: the word-vector of all words
#     return: pre-description embedding.
#     """
#     head_description_list = []
#     # relation_description_list = []
#     # tail_description_list = []
#
#     for i in range(len(train2id)):
#
#         # head_description_list = []
#
#         print(i, " --> ", train2id[i], "\n")
#         if i == 3:
#             print("i = 10 break !")
#             break
#
#         head_index = int(train2id[i][0])
#         tail_index = int(train2id[i][1])
#         relation_index = int(train2id[i][2])
#
#         head_obj = entity_description_obj[head_index]
#
#         tail_obj = entity_description_obj[tail_index]
#
#         relation_des = relation2id[relation_index][0]
#
#         relation_description = str(
#             relation_des) + ', ' + 'which is between ' + head_obj.symb + ' and ' + tail_obj.symb + ';' \
#                                + head_obj.get_random_neighbour() + ';' + tail_obj.get_random_neighbour()
#
#         """
#         obtain entity and relation description represented by word
#         """
#         # head_description_word_list = entity_text_process(head_description)
#         # relation_description_word_list = relation_text_process(relation_description)
#         # tail_description_word_list = entity_text_process(tail_description)
#         #
#         head_description_word_list = head_obj.get_entity_description()
#         relation_description_word_list = relation_text_process(relation_description)
#         tail_description_word_list = tail_obj.get_entity_description()
#
#         """ create word-bag , I have obtain word list , and obtain each word embedding using glove"""
#
#         """ next , all words become vector using World2vector
#             将获取的head_description_word_list，relation_description_word_list，tail_description_word_list
#             通过word词模型，变成向量,然后使用LSTM进行编码
#             from get_word2vector import get_word2vec
#         """
#         # get sentence embedding
#
#         head_description_list.append(head_description_word_list)
#         # relation_description_list.append(relation_description_word_list)
#         # tail_description_list.append(tail_description_word_list)
#
#         # print("head_description_list",head_description_list)
#
#     get_sentence_init_embedding(pre_word_embedding, word_bag, head_description_list)
#
#     # write_to_file("./FB15K/word_bag.txt",word_list)


def obtain_all_entity_resource(entity_description_path):
    #
    """obtain entity description"""

    entity_description = read_entity_description(entity_description_path)  # read original entity description

    all_entity_description_obj, all_entity_description_word_list, all_word_bag, all_word_bag_dic, pre_trained_word_embedding = set_entity_description_obj(
        entity_description)  # set entity object

    re = {'all_entity_description_obj': all_entity_description_obj,
          'all_entity_description_word_list': all_entity_description_word_list, 'all_word_bag': all_word_bag,
          'all_word_bag_dic': all_word_bag_dic, 'pre_trained_word_embedding': pre_trained_word_embedding}

    return re


def obtain_each_relation_description(head_en_obj, tail_en_obj):
    pass


def get_hrt_description_embedding(_h, _r, _t, entity_res, relation2id):
    """
    :param _h: head index
    :param _r: relation index
    :param _t: tail index
    :param ret: entity resource
    :param relation2id: relation id and name
    :return: the word embedding of _h,_r, and _t,

     entity_res = {'all_entity_obj_list': all_entity_obj_list,
                    'all_entity_description_word_list': all_entity_description_word_list,
                    'all_word_bag': all_word_bag,
                    'all_word_bag_dic': all_word_bag_dic,
                    'pre_trained_word_embedding': pre_trained_word_embedding
                    }

    """
    char = " "

    all_entity_res_obj = entity_res['all_entity_obj_list']
    all_entity_des_word = entity_res['all_entity_description_word_list']
    pre_word_embedding = entity_res['pre_trained_word_embedding']
    word_bag = entity_res['all_word_bag_dic']

    head_index = _h
    tail_index = _t
    relation_index = _r

    # print("relation_index",relation_index)

    head_obj = [all_entity_res_obj[i] for i in head_index]
    tail_obj = [all_entity_res_obj[i] for i in tail_index]

    head_description_list = [" ".join(all_entity_des_word[i]) for i in head_index]  # get head entity description

    tail_description_list = [" ".join(all_entity_des_word[i]) for i in tail_index]  # get tail entity

    relation_description_list = []  # get relation descriptions

    # relation_index = relation_index.cpu().numpy()

    relation_name = relation2id[relation_index, 0]

    # print("relation_name",relation_name)

    # from multiprocessing import Pool
    # pool = Pool(processes=32)
    #
    # rel = pool.map(admin, num_list)
    # print(rel)
    
    for i in range(len(relation_name)):
        # print("combine relation des ", i)

        # head_name = head_obj[i].label
        # tail_name = tail_obj[i].label
        # if '/m/' in head_name:
        #     head_name = head_obj[i].label
        #
        # else:
        #     head_name = ta.clean(head_name)
        #
        # if '/m/' in tail_name:
        #     tail_name = tail_obj[i].label
        #
        # else:
        #     tail_name = ta.clean(tail_name)

        rel_des = str(relation_name[i]) + ', ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
            i].label + ';' \
                  + head_obj[i].get_random_neighbour() + ';' + tail_obj[i].get_random_neighbour()
        # print("rel_des ", rel_des)
        relation_description_list.append(rel_des)




    # print(relation_description_list)
    # import time
    # time.sleep(20)

    """
    obtain entity and relation description represented by word
    """
    # head_description_word_list = entity_text_process(head_description)
    # relation_description_word_list = relation_text_process(relation_description)
    # tail_description_word_list = entity_text_process(tail_description)
    #
    # head_description_word_list = head_obj.get_entity_description()

    relation_description_word_list = relation_text_process(relation_description_list)

    # tail_description_word_list = tail_obj.get_entity_description()

    """ create word-bag , I have obtain word list , and obtain each word embedding using glove"""

    """ next , all words become vector using World2vector 
        将获取的head_description_word_list，relation_description_word_list，tail_description_word_list
        通过word词模型，变成向量,然后使用LSTM进行编码
        from get_word2vector import get_word2vec 
    """
    # get sentence embedding

    # head_description_list.append(head_description_word_list)
    # relation_description_list.append(relation_description_word_list)
    # tail_description_list.append(tail_description_word_list)
    # print("head_description_list",head_description_list)

    # h_des_init_embedding = get_sentence_init_embedding(pre_word_embedding, word_bag, head_description_list)
    # r_des_init_embedding = get_sentence_init_embedding(pre_word_embedding, word_bag, relation_description_word_list)
    # t_des_init_embedding = get_sentence_init_embedding(pre_word_embedding, word_bag, tail_description_list)

    # Thread
    # print("head_description_list:\n ", head_description_list)
    # print("relation_description_word_list:\n ", relation_description_word_list)
    # print("tail_description_list:\n ", tail_description_list)
    # print("------------------")

    # time.sleep(2)

    # thread_list = []
    # h_des_e = MyThread(get_sentence_init_embedding, args=(pre_word_embedding, word_bag, head_description_list,))
    # r_des_e = MyThread(get_sentence_init_embedding,
    #                    args=(pre_word_embedding, word_bag, relation_description_word_list,))
    # t_des_e = MyThread(get_sentence_init_embedding, args=(pre_word_embedding, word_bag, tail_description_list,))
    #
    # thread_list.append(h_des_e)
    # thread_list.append(r_des_e)
    # thread_list.append(t_des_e)
    #
    # for t in thread_list:
    #     t.start()
    # for t in thread_list:
    #     t.join()
    #
    # h_des_init_embedding = thread_list[0].get_result()
    # r_des_init_embedding = thread_list[1].get_result()
    # t_des_init_embedding = thread_list[2].get_result()

    # print(h_des_init_embedding.shape)

    # return h_des_init_embedding, r_des_init_embedding, t_des_init_embedding

    return head_description_list, relation_description_word_list,tail_description_list

if __name__ == "__main__":
    entity_description_path = "../benchmarks/FB15K/all_entity_description_6.txt"

    ret = obtain_all_entity_resource(entity_description_path)

    # ret = {'all_entity_description_obj': all_entity_description_obj,
    #       'all_entity_description_word_list': all_entity_description_word_list, 'all_word_bag': all_word_bag,
    #       'all_word_bag_dic': all_word_bag_dic, 'pre_trained_word_embedding': pre_trained_word_embedding}

    # number_of_entity = len(entity_description_obj)
    # print(number_of_entity)
    # for i in range(number_of_entity):
    #     tmp_en = entity_description_obj[i]
    #     entity_str = tmp_en.id + '\t' + tmp_en.symb + '\t' + tmp_en.label + '\t' + tmp_en.description
    #     print(entity_str)
    #     entity_des = tmp_en.get_entity_description()
    #     print("entity_des :\n ",entity_des)
    #
    #     print("all_entity_description_list :\n",all_entity_description_list[i])
    #     import time
    #     time.sleep(2)

    # word_bag_path = "./FB15K/word_bag.txt"
    # word_bag, pre_word_embedding = get_word2vec(word_bag_path)

    """
    Obtain entity2id ,relation2id, and train2id. (25 Mar)
    """
    # entity2id_path = "../benchmarks/FB15K/entity2id.txt"
    relation2id_path = "../benchmarks/FB15K/relation2id.txt"
    # train_id_path = "../benchmarks/FB15K/train2id.txt"

    # entity2id = read_ent_rel_2id(entity2id_path)
    relation2id = read_ent_rel_2id(relation2id_path)

    # train2id = read_data2id(train_id_path)

    # entity_description_obj = ret['all_entity_description_obj']
    # word_bag_dic = ret['all_word_bag_dic']
    # pre_word_embedding = ret['pre_trained_word_embedding']
    # get_triples_description(train2id, relation2id, entity_description_obj, word_bag_dic, pre_word_embedding)

    _h = [0, 2, 4, 6, 8, 10]
    _r = [0, 1, 2, 3, 4, 5]
    _t = [1, 3, 5, 7, 9, 11]

    h_des_init_embedding, r_des_init_embedding, t_des_init_embedding = get_hrt_description_embedding(_h, _r, _t, ret,
                                                                                                     relation2id)
