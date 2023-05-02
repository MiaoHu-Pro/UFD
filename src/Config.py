# coding:utf-8
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from numpy import unicode
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from loss.BaseModule import BaseModule
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn import metrics

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


torch.cuda.empty_cache()

use_gpu=False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

if torch.cuda.is_available():
    use_gpu = True

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        # print("lines",lines[0])

        # time.sleep(2)

        ent2text = {}
        with open(os.path.join(data_dir, "entity2new_des_4nums_2step.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        # with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
        #     rel_lines = f.readlines()
        #     for line in rel_lines:
        #         temp = line.strip().split('\t')
        #         rel2text[temp[0]] = temp[1]

        if set_type == "train":
            with open(os.path.join(data_dir, "train_triple_des_4num_2step.txt"), 'r') as f:
                rel_lines = f.readlines()
                for line in rel_lines:
                    temp = line.strip().split('\t')
                    rel2text[temp[0]] = temp[2]

        elif set_type == "dev":
            with open(os.path.join(data_dir, "valid_triple_des_4num_2step.txt"), 'r') as f:
                rel_lines = f.readlines()
                for line in rel_lines:
                    temp = line.strip().split('\t')
                    rel2text[temp[0]] = temp[2]

        else:
            with open(os.path.join(data_dir, "train_triple_des_4num_2step.txt"), 'r') as f:
                rel_lines = f.readlines()
                for line in rel_lines:
                    temp = line.strip().split('\t')
                    rel2text[temp[0]] = temp[2]

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):

            # 导入数据 ：
            # head_ent_text : Robert Ryan
            # tail_ent_text : Actor-GB
            # relation_text : people person profession
            # 获取 entity_des
            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            # relation_text = rel2text[line[1]]
            # 读取创建好的triple des,将关系描述分割出来
            relation_text = rel2text[str(i)]

            # print("head_ent_text", line[0],"\t",head_ent_text)
            # print("tail_ent_text", line[2],"\t",tail_ent_text)
            # print("relation_text", line[1],"\t",relation_text)
            # print("==================\n")

            if set_type == "dev" or set_type == "test":

                # triple_label = line[3]
                # if triple_label == "1":
                #     label = "1"
                # else:
                #     label = "0"
                label = "1"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))
                else:
                    # corrupting tail
                    tmp_tail = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))
        return examples

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        """
        print("total length : ", total_length)
        print("--len(tokens_a)",len(tokens_a))
        print("--len(tokens_b)",len(tokens_b))
        print("--len(tokens_c)",len(tokens_c))
        """
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    # print("convert examples to features ... ")

    features = []
    for (ex_index, example) in enumerate(examples):

        # print("ex_index: ",ex_index)

        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a) # TypeError: ord() expected a character, but string of length 4 found


        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            #_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)

        # print("len(tokens_a)", len(tokens_a))
        # print("len(tokens_b)", len(tokens_b))
        # print("len(tokens_c)", len(tokens_c))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # print("len(input_ids) : ",len(input_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # print("label_id :", label_id)

        # if ex_index < 5 and print_info:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

# ------------------------


class MyDataParallel(nn.DataParallel):
    def _getattr__(self, name):
        return getattr(self.module, name)
  
            
def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


class Config(object):
    def __init__(self):
        # super(BaseModule, self).__init__()
        base_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "release/Base.so")
        )
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        """'sample"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        """'valid"""
        self.lib.getValidHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getValidTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.validHead.argtypes = [ctypes.c_void_p]
        self.lib.validTail.argtypes = [ctypes.c_void_p]
        """test link prediction"""
        self.lib.getHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        """test triple classification"""
        self.lib.getValidBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTestBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getBestThreshold.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.test_triple_classification.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """restype"""
        self.lib.getValidHit10.restype = ctypes.c_float

        # for triple classification
        self.lib.test_triple_classification.restype = ctypes.c_float
        """set essential parameters"""

        self.in_path = "./"
        self.batch_size = 100
        self.bern = 0
        self.work_threads = 8
        self.hidden_size = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.margin = 1.0
        self.valid_steps = 5
        self.save_steps = 5
        self.opt_method = "SGD"
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0.0
        self.lmbda_two = 0.0
        self.alpah = 0.001
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 1
        self.test_link = True
        self.test_triple = False
        self.model = None
        self.trainModel = None
        self.testModel = None
        self.pretrain_model = None
        self.ent_dropout = 0
        self.rel_dropout = 0
        self.use_init_embeddings = False
        self.test_file_path = None

        self.max_seq_length = 512
        self.en_rel_des_obj = None
        self.num_labels = 2

    def init(self):
        self.lib.setInPath(
            ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2)
        )

        self.lib.setTestFilePath(
            ctypes.create_string_buffer(self.test_file_path.encode(), len(self.test_file_path) * 2)
        )

        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.lib.importTestFiles()
        self.lib.importTypeFiles()

        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.testTotal = self.lib.getTestTotal()
        self.validTotal = self.lib.getValidTotal()

        self.batch_size = int(self.trainTotal / self.nbatches)
        self.batch_seq_size = self.batch_size * (
            1 + self.negative_ent + self.negative_rel
        )

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)

        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

        self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_r = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_h_addr = self.valid_h.__array_interface__["data"][0]
        self.valid_t_addr = self.valid_t.__array_interface__["data"][0]
        self.valid_r_addr = self.valid_r.__array_interface__["data"][0]

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)

        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__["data"][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__["data"][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__["data"][0]
        self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__["data"][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__["data"][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__["data"][0]

        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]
        self.relThresh = np.zeros(self.relTotal, dtype=np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__["data"][0]


        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        self.label_list = self.get_labels()
        self.num_labels = len(self.label_list)

        self.cache_dir = self.cache_dir if self.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(self.local_rank))

    def set_local_rank(self, local_rank):
        self.local_rank = local_rank

    def set_cache_dir(self, cache_dir):
        self.cache_dir = cache_dir

    def get_labels(self):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def set_bert_model(self, bert_model):
        self.bert_model = bert_model

    def set_do_lower_case(self, do_lower_case):

        self.do_lower_case = do_lower_case

    def set_en_rel_obj(self, en_rel_des):

        self.en_rel_des_obj = en_rel_des

    def set_max_seq_length(self, max_seq_length):

        self.max_seq_length = max_seq_length

    def set_test_link(self, test_link):
        self.test_link = test_link

    def set_test_triple(self, test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_test_file_path(self, test_file_path):
        self.test_file_path = test_file_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda
        
    def set_lmbda_two(self, lmbda_two):
        self.lmbda_two = lmbda_two

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_bern(self, bern):
        self.bern = bern

    def set_init_embeddings(self, entity_embs, rel_embs):
        self.use_init_embeddings = True
        self.init_ent_embs = torch.from_numpy(entity_embs).to(device)  # 初始化的entity embedding
        self.init_rel_embs = torch.from_numpy(rel_embs).to(device)     # 初始化的relation embedding

    def set_config_CNN(self, num_of_filters, drop_prob, kernel_size=1):
        self.out_channels = num_of_filters
        self.convkb_drop_prob = drop_prob
        self.kernel_size = kernel_size

    def set_config_BERT(self):
        pass

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        self.rel_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_ent_dropout(self, ent_dropout):
        self.ent_dropout = ent_dropout

    def set_rel_dropout(self, rel_dropout):
        self.rel_dropout = rel_dropout
        
    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def get_batch_size(self):
        return self.batch_size

    def get_parameters_best_model(self, param_dict, mode="numpy"):

        for param in param_dict:
            param_dict[param] = param_dict[param].cpu()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = param_dict[param].numpy()
            elif mode == "list":
                res[param] = param_dict[param].numpy().tolist()
            else:
                res[param] = param_dict[param]
        return res

    def get_parameters(self):
        # print("====get_parameters=====")
        out = self.trainModel.get_parameters(mode = "list")
        return out

    def save_embedding_matrix(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".json")
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters_best_model(best_model, "list")))
        f.close()


    def set_train_model(self, model):
        print("Initializing training model...")

        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        self.model = model
        self.trainModel = self.model(config=self) # ConvKB -->   forward pass
        #self.trainModel = nn.DataParallel(self.trainModel, device_ids=[2,3,4])

        self.trainModel.to(device)

        # if n_gpu > 1:
        #     self.trainModel = torch.nn.DataParallel(self.trainModel)

        # self.trainModel.bert_model.to(device)

        # print("self.trainModel.parameters()", self.trainModel.parameters().key())


        #  加载优化器
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.trainModel.parameters(), # 待优化参数的iterable或者是定义了参数组的dict
                lr=self.alpha,                # 学习率
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,# 可选，权重衰减，L2乘法，默认0
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing")

    def set_test_model(self, model, path=None):
        print("Initializing test model...")
        self.model = model   # 加载模型
        self.testModel = self.model(config=self) #初始化模型
        if path == None:
            path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        self.testModel.load_state_dict(torch.load(path)) #加载参数 读取  .ckpt
        self.testModel.to(device)
        self.testModel.eval()
        print("Finish initializing")

    def sampling(self):

        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
        )

    def save_checkpoint(self, model, epoch):
        path = os.path.join(
            self.checkpoint_dir, self.model.__name__ + "-" + str(epoch) + ".ckpt"
        )
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        torch.save(best_model, path)

    @property
    def train_one_step(self):

        set_type = "train"

        self.trainModel.train()  # ConvKB  --> forward :  forward pass
        self.trainModel.batch_h = to_var(self.batch_h)
        self.trainModel.batch_t = to_var(self.batch_t)
        self.trainModel.batch_r = to_var(self.batch_r)
        self.trainModel.batch_y = to_var(self.batch_y)

        # print(self.trainModel.batch_y)

        num_batch = len(self.batch_y)
        # print("num_batch: ", num_batch)

        # ERDse - >h_des, r_dex , t_des =  get_triple_des(self.trainModel.batch_h,self.trainModel..batch_r,self.trainModel..batch_t)
        h_des, r_dex, t_des = self.en_rel_des_obj.get_triple_des(self.batch_h, self.batch_r, self.batch_t)
        # train_examples : to package input obj - >  InputExample()
        batch_examples = []
        for i in range(num_batch):
            guid = "%s-%s" % (set_type, i)
            # str(self.batch_y[i]))

            if self.batch_y[i] == 1:
                label = "1"
            else:
                label = "0"
            batch_examples.append(InputExample(guid=guid, text_a=h_des[i], text_b=r_dex[i], text_c=t_des[i], label=label))

        # train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        batch_train_features = convert_examples_to_features(batch_examples, self.label_list,
                                                            self.max_seq_length, self.tokenizer)

        self.trainModel.batch_input_ids = (torch.tensor([f.input_ids for f in batch_train_features], dtype=torch.long)).to(device)
        self.trainModel.batch_input_mask = (torch.tensor([f.input_mask for f in batch_train_features], dtype=torch.long)).to(device)
        self.trainModel.batch_segment_ids = (torch.tensor([f.segment_ids for f in batch_train_features], dtype=torch.long)).to(device)
        self.trainModel.batch_label_ids = (torch.tensor([f.label_id for f in batch_train_features], dtype=torch.long)).to(device)

        print("self.trainModel.batch_label_ids",self.trainModel.batch_label_ids)
        # print(self.trainModel.batch_input_ids, self.trainModel.batch_input_ids.shape)
        # print(self.trainModel.batch_input_mask, self.trainModel.batch_input_mask.shape)
        # print(self.trainModel.batch_segment_ids, self.trainModel.batch_segment_ids.shape)
        # print(self.trainModel.batch_label_ids,self.trainModel.batch_label_ids.shape)

        # time.sleep(10)

        #
        # loss = model(input_ids, segment_ids, input_mask, labels=None)
        self.optimizer.zero_grad() # 梯度清空
        # trainModel就是 ConvKB，调用ConvKB --—> forward
        loss = self.trainModel() # 计算损失 什么样的损失函数？？？？？？ 得到损失，反向传播
        print("loss backward ... ")
        loss.backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(self.trainModel.parameters(), 0.5)
        self.optimizer.step() # 更新参数

        return loss.item()

    def test_one_step(self, model, test_h, test_t, test_r):
        model.eval()
        set_type = "testing ..."
        with torch.no_grad():

            # model.batch_h = to_var(test_h)
            # model.batch_t = to_var(test_t)
            # model.batch_r = to_var(test_r)

            model.batch_h = to_var(self.batch_h)
            model.batch_t = to_var(self.batch_t)
            model.batch_r = to_var(self.batch_r)
            model.batch_y = to_var(self.batch_y)

            # print(self.trainModel.batch_y)

            num_batch = len(self.batch_y)
            print("num_batch: ", num_batch)

            # ERDse - >h_des, r_dex , t_des =  get_triple_des(self.trainModel.batch_h,self.trainModel..batch_r,self.trainModel..batch_t)
            h_des, r_dex, t_des = self.en_rel_des_obj.get_triple_des(test_h, test_r, test_t)
            # train_examples : to package input obj - >  InputExample()
            batch_examples = []
            for i in range(num_batch):
                guid = "%s-%s" % (set_type, i)
                # str(self.batch_y[i]))

                label = "1"
                batch_examples.append(InputExample(guid=guid, text_a=h_des[i], text_b=r_dex[i], text_c=t_des[i], label=label))

            # train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
            batch_train_features = convert_examples_to_features(batch_examples, self.label_list,
                                                                self.max_seq_length, self.tokenizer)

            model.batch_input_ids = (torch.tensor([f.input_ids for f in batch_train_features], dtype=torch.long)).to(device)
            model.batch_input_mask = (torch.tensor([f.input_mask for f in batch_train_features], dtype=torch.long)).to(device)
            model.batch_segment_ids = (torch.tensor([f.segment_ids for f in batch_train_features], dtype=torch.long)).to(device)
            model.batch_label_ids = (torch.tensor([f.label_id for f in batch_train_features], dtype=torch.long)).to(device)

        return model.predict()

    def valid(self, model):
        self.lib.validInit()
        for i in range(self.validTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getValidHeadBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)

            self.lib.validHead(res.__array_interface__["data"][0])

            self.lib.getValidTailBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)
            self.lib.validTail(res.__array_interface__["data"][0])
        return self.lib.getValidHit10()


    def training_model(self):

        # print("---------check--------------------")

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit10 = 0.0
        best_model = None
        bad_counts = 0
        training_range = tqdm(range(self.train_times))
        for epoch in training_range: # 训练n次
            res = 0.0
            for batch in range(self.nbatches):
                print("Epoch -> batch: ", epoch, batch)
                self.sampling() # 采样

                loss = self.train_one_step
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            # print("Epoch %d | loss: %f" % (epoch, res))
            if (epoch + 1) % self.save_steps == 0:
                training_range.set_description("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)

            if (epoch + 1) % self.valid_steps == 0:
                training_range.set_description("Epoch %d has finished | loss: %f, validating..." % (epoch, res))
                hit10 = self.valid(self.trainModel)
                if hit10 > best_hit10:
                    best_hit10 = hit10
                    best_epoch = epoch
                    best_model = self.trainModel.state_dict()
                    print("Best model | hit@10 of valid set is %f" % (best_hit10))
                    bad_counts = 0
                else:
                    print("Hit@10 of valid set is %f | bad count is %d" % (hit10, bad_counts))
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print("Early stopping at epoch %d" % (epoch))
                    break

        if best_model == None:

            # print("--------best_model---------------")

            best_model = self.trainModel.state_dict()   # 保存参数 将每一层与它的对应参数建立映射关系
            # print("--------self.trainModel.state_dict()---------------")
            best_epoch = self.train_times - 1
            # print("--------best_model---------------")
            best_hit10 = self.valid(self.trainModel)


        print("Best epoch is %d | hit@10 of valid set is %f" % (best_epoch, best_hit10))
        print("Store checkpoint of best result at epoch %d..." % (best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)
        print("Finish storing")
        print("Testing...")
        self.set_test_model(self.model)
        self.test()
        print("Finish test")

        print("---------check--------------------")

        return best_model

    def valid_triple_classification(self, model):
        self.lib.getValidBatch(
            self.valid_pos_h_addr,
            self.valid_pos_t_addr,
            self.valid_pos_r_addr,
            self.valid_neg_h_addr,
            self.valid_neg_t_addr,
            self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            model, self.valid_pos_h, self.valid_pos_t, self.valid_pos_r
        )
        res_neg = self.test_one_step(
            model, self.valid_neg_h, self.valid_neg_t, self.valid_neg_r
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        return self.lib.test_triple_classification(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

    def training_triple_classification(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_acc = 0.0
        best_model = None
        bad_counts = 0
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for batch in range(self.nbatches):
                self.sampling()
                loss = self.train_one_step
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            if (epoch + 1) % self.save_steps == 0:
                training_range.set_description("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)
            if (epoch + 1) % self.valid_steps == 0:
                training_range.set_description("Epoch %d has finished | loss: %f, validating..." % (epoch, res))
                acc = self.valid_triple_classification(self.trainModel)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_model = self.trainModel.state_dict()
                    print("Best model | Acc of valid set is %f" % (best_acc))
                    bad_counts = 0
                else:
                    print("Acc of valid set is %f | bad count is %d" % (acc, bad_counts))
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print("Early stopping at epoch %d" % (epoch))
                    break
        if best_model == None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            best_acc = self.valid_triple_classification(self.trainModel)
        print("Best epoch is %d | Acc of valid set is %f" % (best_epoch, best_acc))
        print("Store checkpoint of best result at epoch %d..." % (best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)
        print("Finish storing")
        print("Testing...")
        self.set_test_model(self.model)
        self.test()
        print("Finish test")
        return best_model

    def link_prediction(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testHead(res.__array_interface__["data"][0])

            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testTail(res.__array_interface__["data"][0])
        self.lib.test_link_prediction()

    def triple_classification(self):
        self.lib.getValidBatch(
            self.valid_pos_h_addr,
            self.valid_pos_t_addr,
            self.valid_pos_r_addr,
            self.valid_neg_h_addr,
            self.valid_neg_t_addr,
            self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.valid_pos_h, self.valid_pos_t, self.valid_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.valid_neg_h, self.valid_neg_t, self.valid_neg_r
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel, self.test_pos_h, self.test_pos_t, self.test_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, self.test_neg_h, self.test_neg_t, self.test_neg_r
        )
        self.lib.test_triple_classification(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

    def test(self):
        if self.test_link:
            self.link_prediction()
        if self.test_triple:
            self.triple_classification()
