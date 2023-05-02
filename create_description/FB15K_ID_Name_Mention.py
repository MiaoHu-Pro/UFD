
# this demo is to get the structure of "index + ID + Name + Mention" for FB15K.

# 1.read entity2id,
# 2.read entity2text that is entity name,
# 3.read entity2Obj, 将确实的实体名进行补全/
import os

import numpy as np
import pandas as pd


def read_entity2id(data_id_paht):
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
    data_id = np.array(data_id)
    return data_id

def read_entity2obj(entity_obj_path):
    """
    14344(index) 	/m/0wsr(symbol) 	 Atlanta Falcons(label)	 American football team (description)
    :param entity_obj_path:
    :return:
    """
    f = open(entity_obj_path)

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
    X = np.array(x_obj)

    return X

def write_ID_Name_Mention(out_path,data):

    ls = os.linesep
    num = len(data)

    try:
        fobj = open(out_path,  'w')
    except IOError as err:
        print('file open error: {0}'.format(err))

    else:
        for j in range(num):
            #
            _str = str(j) + '\t' + data[j][0] + '\t' + data[j][1] + '\t' + data[j][2] + '\n'

            fobj.writelines('%s' % _str)

        fobj.close()

    print('WRITE FILE DONE!')

if __name__ == "__main__":

    entity2id_path = "../data/FB15K/entity2id.txt"
    entity2name_path = "../data/FB15K/entity2text.txt"
    entity2Obj_path = "../data/FB15K/entity2Obj.txt"

    entity2id = read_entity2id(entity2id_path)
    entity2name = read_entity2obj(entity2name_path)
    entity2obj = read_entity2obj(entity2Obj_path)

    print(entity2id.shape)
    print(entity2name.shape)
    print(entity2obj.shape)

    row, column = entity2name.shape

    entity2obj_id = entity2obj[:,1].tolist()

    id_name_mention = []
    for i in range(row):
        _id_name_mention = []
        Id = entity2name[i][0]
        Name = entity2name[i][1]
        Mention = None
        if Id in entity2obj_id:
            index = entity2obj_id.index(Id)
            Mention = entity2obj[index][3]
        else:
            Mention = "There is no mention."

        _id_name_mention.append(Id)
        _id_name_mention.append(Name)
        _id_name_mention.append(Mention)

        id_name_mention.append(_id_name_mention)

    print(id_name_mention[10])
    write_ID_Name_Mention("../data/FB15K/ID_Name_Mention.txt",id_name_mention)









