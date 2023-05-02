# define a class,ERDes,to finish the task of obtaining entity and relation description

# coding:utf-8
import torch
from torch.autograd import Variable
import os
import numpy as np

use_gpu = False

def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


from create_description.utilities_get_entity_description import read_all_triples, read_entity2obj, read_entity2id, \
    obtain_entity_res, read_ent_rel_2id
from create_description.utilities_get_hrt_text import get_hrt_description_embedding
from create_description.utilities import read_data2id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    use_gpu = True


def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


def write_to_file_entity_obj(out_path, all_data):
    ls = os.linesep
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for x in all_data:
            # _str = str(x.id) + '\t' + str(x.symbol) + '\t' + str(x.label) + '\t' + str(x.mention) + '\t' + str(
            #     x.neighbours) + '\n'

            _str = str(x.id) + '\t' + str(x.symbol) + '\t' + char.join(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')


def write_triple_descriptions(out_path, head_des, rel_des, tail_des):
    num_triples = len(head_des)
    ls = os.linesep
    head_len = []
    rel_len = []
    tail_len = []
    char = " "

    try:
        fobj = open(out_path, 'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for i in range(num_triples):
            # print(i)
            head = head_des[i]
            rel = rel_des[i]
            tail = tail_des[i]
            head_len.append(len(head))
            rel_len.append(len(rel))
            tail_len.append(len(tail))

            _str = str(i) + '\t' + char.join(head) + '\t' + char.join(rel) + '\t' + char.join(tail) + '\n'
            #
            # _str = str(x.id) + '\t' + str(x.entity_des) + '\n'
            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')
    print("head len ", np.mean(head_len))
    print("rel len ", np.mean(rel_len))
    print("tail len ", np.mean(tail_len))


class ERDes(object):

    def __init__(self, _Paras):
        self.num_neighbours = _Paras['num_neighbours']
        self.num_step = _Paras['num_step']
        self.word_dim = _Paras['word_dim']
        self.all_triples_path = _Paras['all_triples_path']
        self.entity2Obj_path = _Paras['entity2Obj_path']
        self.entity2id_path = _Paras['entity2id_path']

        self.training_entity2id_path = _Paras['training_entity2id_path']
        self.training_relation2id_path = _Paras['training_relation2id_path']

        self.relation2id_path = _Paras['relation2id_path']

        self.entity_des_path = _Paras['entity_des_path']

        self.entity_res = None
        self.entity_symbol_set = []
        self.entity_index_set = []
        self.training_entity_index_set = []
        self.training_relation_index_set = []


        self.relation2id = []  # ['/location/country/form_of_government' '0']

    def get_entity_des(self):
        print("get entity des BEGIN...")

        # get entity2Obj
        X, relation_set, entity_set, entityPair_set = read_all_triples(self.all_triples_path)
        sub_x_obj = read_entity2obj(
            self.entity2Obj_path)  # 14515 entities have label and des , and about 436 has not desc...

        relation_set = list(set(relation_set))  # all relation

        entity_set = list(set(entity_set))  # all entity

        # obtain_ entity id
        entity_id_read_file = read_entity2id(self.entity2id_path)
        self.entity_symbol_set = entity_id_read_file[:, 0].tolist()
        self.entity_index_set = entity_id_read_file[:, 1].tolist()

        # obtain training entity id
        training_id_read_file = read_entity2id(self.training_entity2id_path)
        self.training_entity_index_set = training_id_read_file[:,1].tolist()

        training_rel_id_read_file = read_ent_rel_2id(self.training_relation2id_path)
        self.training_relation_index_set = training_rel_id_read_file[:,1].tolist()

        # entity2name:
        entity2name = {}  # '/m/03_48k': 'Fred Ward'
        for i in range(len(sub_x_obj)):
            entity2name[sub_x_obj[i][1]] = sub_x_obj[i][2]

        all_entity_obj_list, all_entity_description_word_list, all_word_bag, all_word_bag_dic, pre_trained_word_embedding = \
            obtain_entity_res(
                X, sub_x_obj, entity2name, self.entity_symbol_set, self.num_neighbours, self.num_step)

        self.entity_res = {'all_entity_obj_list': all_entity_obj_list,
                           'all_entity_description_word_list': all_entity_description_word_list,
                           'all_word_bag': all_word_bag,
                           'all_word_bag_dic': all_word_bag_dic,
                           'pre_trained_word_embedding':pre_trained_word_embedding
                           }

        self.relation2id = read_ent_rel_2id(self.relation2id_path)  # ['/location/country/form_of_government' '0']
        print("get entity des OVER ... \n")

    def get_relation_des(self):
        pass

    def get_triple_des(self, _h, _r, _t):
        # print("get triple des begin ... ")
        h_des, r_des, t_des = get_hrt_description_embedding(_h, _r, _t, self.entity_res, self.relation2id)
        # print("finished get triple des ... ")

        return h_des, r_des, t_des

    def er_des_print(self):
        print(self.entity2id_path)


def obtain_train_triple_des(file_path, en_rel_des):
    print("obtain_train_triple_des ... \n")
    train_data_set_path = file_path + 'train2id.txt'
    train = read_data2id(train_data_set_path)
    h = train[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = train[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = train[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]


    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'train_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_valid_triple_des(file_path, en_rel_des):
    print("obtain_valid_triple_des ... \n")
    valid_data_set_path = file_path + 'valid2id.txt'
    valid = read_data2id(valid_data_set_path)
    h = valid[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = valid[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = valid[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]
    # _h = [0, 2, 4, 6, 8, 10]
    # _r = [0, 1, 2, 3, 4, 5]
    # _t = [1, 3, 5, 7, 9, 11]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'valid_triple_des_4num_2step.txt', h_des, r_des, t_des)


def obtain_test_triple_des(file_path, en_rel_des):
    print("obtain_test_triple_des ... \n")
    test_data_set_path = file_path + 'test2id.txt'
    test = read_data2id(test_data_set_path)
    h = test[:, 0].tolist()
    _h = [int(h[i]) for i in range(len(h))]
    t = test[:, 1].tolist()
    _t = [int(t[i]) for i in range(len(t))]
    r = test[:, 2].tolist()
    _r = [int(r[i]) for i in range(len(r))]
    # _h = [0, 2, 4, 6, 8, 10]
    # _r = [0, 1, 2, 3, 4, 5]
    # _t = [1, 3, 5, 7, 9, 11]

    h_des, r_des, t_des = en_rel_des.get_triple_des(_h, _r, _t)

    write_triple_descriptions(file_path + 'test_triple_des_4num_2step.txt', h_des, r_des, t_des)


if __name__ == "__main__":

    file_path = '../benchmarks/FB15K237/'
    Paras = {
        'num_neighbours': 4,
        'num_step': 2,
        'word_dim': 100,
        'all_triples_path': file_path + 'train.tsv',
        'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
        'entity2id_path': file_path + 'entity2id.txt',
        'relation2id_path': file_path + 'relation2id.txt',
        'entity_des_path': file_path + 'entity2new_des_4nums_2step.txt',
    }
    en_rel_des = ERDes(_Paras=Paras)
    en_rel_des.get_entity_des()

    # train
    obtain_train_triple_des(file_path, en_rel_des)
    # valid
    obtain_valid_triple_des(file_path, en_rel_des)
    # test
    obtain_test_triple_des(file_path, en_rel_des)
