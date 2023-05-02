from Config import Config
from ConvKB import ConvKB
from BERT_old import Bert
import json
import os
import numpy as np
import time
from argparse import ArgumentParser
from create_description.ERDse import ERDes


# 设置参数
"""
python train_Bert.py --dataset FB15K237 --hidden_size 100 --num_of_filters 128 --neg_num 10 --valid_step 50 
--nbatches 100 --num_epochs 300 
--learning_rate 0.01 --lmbda 0.1 --model_name FB15K237_lda-0.1_nneg-10_nfilters-128_lr-0.01 --mode train

"""
parser = ArgumentParser("ConvKB")
parser.add_argument("--dataset", default="FB15K237", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--nbatches", default=100, type=int, help="Number of batches")
parser.add_argument("--num_epochs", default=1, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='FB15K237', help="")
parser.add_argument('--neg_num', default=2, type=int, help='')
parser.add_argument('--hidden_size', type=int, default=50, help='')
parser.add_argument('--num_of_filters', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--save_steps', type=int, default=1000, help='')
parser.add_argument('--valid_steps', type=int, default=50, help='')
parser.add_argument("--lmbda", default=0.2, type=float, help="")
parser.add_argument("--lmbda2", default=0.01, type=float, help="")
parser.add_argument("--mode", choices=["train", "predict"], default="train", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--test_file", default="", type=str)
parser.add_argument("--optim", default='adagrad', help="")
parser.add_argument('--use_init', default=1, type=int, help='')
parser.add_argument('--kernel_size', default=1, type=int, help='')

parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

args = parser.parse_args()

print(args)

out_dir = os.path.abspath(os.path.join("../../runs_pytorch_ConvKB/")) # 输出模型参数
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
result_dir = os.path.abspath(os.path.join(checkpoint_dir, args.model_name))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 加数据类
file_path = '../data/FB15K237/'
Paras = {
    'num_neighbours': 4,
    'num_step': 1,
    'word_dim': 100,
    'all_triples_path': file_path + 'train.tsv',
    'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
    'entity2id_path': file_path + 'entity2id.txt',
    'relation2id_path': file_path + 'relation2id.txt',
    'entity_des_path': file_path + 'entity2new_des_4nums_1step.txt',
}
en_rel_des = ERDes(_Paras=Paras)
en_rel_des.get_entity_des() # 获取实体描述


# 加载配置类
con = Config()
in_path = "./data/" + args.dataset + "/" # 加载数据
con.set_in_path(in_path)
test_file_path = ""
if args.test_file != "":
    test_file_path = in_path + args.test_file
con.set_test_file_path(test_file_path) # 测试数据
con.set_work_threads(8)
con.set_train_times(args.num_epochs)
con.set_nbatches(args.nbatches)
con.set_alpha(args.learning_rate)  #
con.set_bern(1)
con.set_dimension(args.hidden_size)# 纬度，
con.set_lmbda(args.lmbda)
con.set_lmbda_two(0.01)
con.set_margin(1.0) #
con.set_ent_neg_rate(args.neg_num)
con.set_opt_method(args.optim)
con.set_save_steps(args.save_steps)
con.set_valid_steps(args.valid_steps)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(checkpoint_dir)
con.set_result_dir(result_dir)


con.set_local_rank(args.local_rank)
con.set_cache_dir(args.cache_dir)

con.set_bert_model(args.bert_model)
con.set_do_lower_case(args.do_lower_case)
con.set_max_seq_length(args.max_seq_length)

con.set_en_rel_obj(en_rel_des)
# set knowledge graph completion
con.set_test_link(True)
con.set_test_triple(True)

con.init() #初始化

def get_term_id(filename):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity

def get_init_embeddings(relinit, entinit):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


if args.mode == "train":

    # if args.use_init:
    #     hidden_size = "100"  # for FB15K-237
    #     con.set_dimension(100)
    #     if args.dataset == "WN18RR":
    #         hidden_size = "50"
    #         con.set_dimension(50)
    #     #初始化数据
    #     init_entity_embs, init_relation_embs = get_init_embeddings(
    #         "./data/" + args.dataset + "/relation2vec"+hidden_size+".init",
    #         "./data/" + args.dataset + "/entity2vec"+hidden_size+".init")
    #
    #     #实体与下标的对应
    #     e2id, id2e = get_term_id(filename="./data/" + args.dataset + "/entity2id.txt")
    #     e2id50, id2e50 = get_term_id(filename="./data/" + args.dataset + "/entity2id_"+hidden_size+"init.txt")
    #     assert len(e2id) == len(e2id50)
    #
    #     #  e2id '01527194': 40942
    #
    #     # id2e40942: '01527194'
    #
    #
    #     entity_embs = np.empty([len(e2id), con.hidden_size]).astype(np.float32) #设置一个实体嵌入矩阵
    #
    #     for i in range(len(e2id)):
    #         _word = id2e[i]
    #
    #         id = e2id50[_word]
    #
    #         entity_embs[i] = init_entity_embs[id]
    #
    #
    #     r2id, id2r = get_term_id(filename="./data/" + args.dataset + "/relation2id.txt")
    #     r2id50, id2r50 = get_term_id(filename="./data/" + args.dataset + "/relation2id_"+hidden_size+"init.txt")
    #     assert len(r2id) == len(r2id50)
    #
    #     rel_embs = np.empty([len(r2id), con.hidden_size]).astype(np.float32) #设置一个关系嵌入矩阵
    #     for i in range(len(r2id)):
    #         _rel = id2r[i]
    #         id = r2id50[_rel]
    #         rel_embs[i] = init_relation_embs[id]
    #
    #     con.set_init_embeddings(entity_embs, rel_embs) #初始化实体嵌入矩阵和关系嵌入矩阵 将entity2vec50.init ,relation2vec50.init 导入初始化举证

    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout, kernel_size=args.kernel_size)
    # con.set_config_BERT(num_of_filters=args.num_of_filters, drop_prob=args.dropout, kernel_size=args.kernel_size)

    con.set_train_model(Bert)

    # print("print parameters :\n")
    # out = con.get_parameters()
    # print(out.keys())
    # time.sleep(100)

    con.training_model()

else:
    if args.use_init:
        hidden_size = "100"  # for FB15K-237

        con.set_dimension(100)
        if args.dataset == "WN18RR":
            hidden_size = "50"
            con.set_dimension(50)

    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout, kernel_size=args.kernel_size)
    con.set_test_model(Bert, args.checkpoint_path)
    con.test()

"""
python train_Bert.py --dataset FB15K237 --hidden_size 100 --num_of_filters 128 --neg_num 10 --valid_step 50 
--nbatches 100 --num_epochs 300 
--learning_rate 0.01 --lmbda 0.1 --model_name FB15K237_lda-0.1_nneg-10_nfilters-128_lr-0.01 --mode train

"""
