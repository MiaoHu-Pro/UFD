
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

    entity2id_path = "../data/WN18/entity2id.txt"
    entity2tex_path = "../data/WN18/entity2text.txt"
    entity2Obj_path = "../data/WN18/entity2Obj.txt"

    entity2name = read_entity2obj(entity2tex_path)

    write_ID_Name_Mention("../data/WN18/ID_Name_Mention.txt", entity2name)









