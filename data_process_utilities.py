import logging
import os
import random
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from create_description.ERDse import ERDes

logger = logging.getLogger(__name__)


class EntDes(object):

    def __init__(self, _processor, _label_list, _entity_list, _task_name):
        self.processor = _processor
        self.label_list = _label_list
        self.entity_list = _entity_list
        self.task_name = _task_name


class TrainSrc(object):
    def __init__(self, _train_features, _num_train_optimization_steps):
        self.train_features = _train_features
        self.num_train_optimization_steps = _num_train_optimization_steps

class TrainExam(object):
    def __init__(self, _train_example):
        self.train_example = _train_example

class UnseenTestSrc(object):
    def __init__(self, _test_features):
        self.test_features = _test_features


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

    def get_train_examples(self, data_dir, negative):
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
        """Reads a tab separated value file.
        reading train.tsv will be changed with reading train2id.txt, obtain index
        """
        data = pd.read_csv(input_file)  #
        data = np.array(data)
        data_id = []
        for i in range(len(data)):
            _tmp = data[i][0]
            tmp = _tmp.split(' ')
            if tmp:
                id_list = []
                for s in tmp:
                    id_list.append(s.strip())
                data_id.append(id_list)

        return data_id

class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()
        self.en_rel_des = None

    def get_entity_res(self, file_path, num_neighbours, num_step):
        Paras = {
            'num_neighbours': num_neighbours,
            'num_step': num_step,
            'word_dim': 100,
            'all_triples_path': file_path + '/train.tsv',
            'entity2Obj_path': file_path + '/ID_Name_Mention.txt',
            'entity2id_path': file_path + '/entity2id.txt',
            'training_entity2id_path': file_path + '/training_entity2id.txt',
            'training_relation2id_path': file_path + '/training_relation2id.txt',
            'relation2id_path': file_path + '/new_relation2id.txt',
            'entity_des_path': file_path + '/entity2new_des_' + str(num_neighbours) + 'nums_' + str(
                num_step) + 'step.txt', }

        self.ent_res = ERDes(_Paras=Paras)
        self.ent_res.get_entity_des()  # 获取实体描述

    def get_train_examples(self, data_dir, negative):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, "train2id.txt")), "train", data_dir, negative)

    def get_dev_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, "valid2id.txt")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)
        print("test2id.txt")
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, "test2id.txt")), "test", data_dir)

    def get_other_graph_test_examples(self, data_dir, test_data ,negative):
        """See base class."""

        print("test2id.txt")
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, test_data)), "unseen_triple_test", data_dir,
            negative)

    def get_every_test_examples(self, data_dir, test_data ,negative):
        """See base class."""

        print("test2id.txt")
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, test_data)), "unseen_triple_test", data_dir,
            negative)


    def get_unseen_test_examples(self, data_dir,negative):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)
        print("test2id.txt")
        return self.to_create_examples(
            self._read_tsv(os.path.join(data_dir, "test2id_unseen_relation.txt")), "unseen_triple_test", data_dir,negative)


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
        return list(self.ent_res.entity_index_set)

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train2id.txt"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "valid2id.txt"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test2id.txt"))

    def to_create_examples(self, lines, set_type, data_dir, negative=1):
        """Creates examples for the training and dev sets."""
        # entity to text
        # entities = list(self.ent_res.entity_index_set)

        # training_entity_set
        entities = list(self.ent_res.training_entity_index_set)
        relations = list(self.ent_res.training_relation_index_set)
        # print("entities", entities)

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        triples = []
        labels = []

        """
        input sample:
        a triple [IDs, head_text, relation_text, tail_text, label ]
        """
        for (i, line) in enumerate(lines):
            # if i % 5000 == 0:
            #     print(i)

            head_ent_index = line[0]
            tail_ent_index = line[1]
            relation_index = line[2]

            if set_type == "dev" or set_type == "test":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

            elif set_type == "unseen_other_graph_rel_test":
                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)


            elif set_type == "unseen_triple_test":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(negative):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)

                            # tmp_ent_list.remove(line[0])

                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break

                        label = 0
                        triples.append([i, int(tmp_head), int(relation_index), int(tail_ent_index)])
                        labels.append(label)
                else:
                    # corrupting tail
                    tmp_tail = ''

                    for j in range(negative):
                        while True:
                            tmp_ent_list = set(entities)

                            # tmp_ent_list.remove(line[1])

                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + line[2]
                            # print("tmp_triple_str",tmp_triple_str)
                            if tmp_triple_str not in lines_str_set:
                                break

                        label = 0
                        triples.append([i, int(head_ent_index), int(relation_index), int(tmp_tail)])
                        labels.append(label)
            elif set_type == "train":

                label = 1
                triples.append([i, int(head_ent_index), int(relation_index), int(tail_ent_index)])
                labels.append(label)

                # rnd = random.random()
                # guid = "%s-%s" % (set_type + "_corrupt", i)

                # creat_negative_triple

                # corrupting head
                tmp_head = ''
                while True:
                    tmp_ent_list = set(entities)
                    # tmp_ent_list.remove(line[0])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_head = random.choice(tmp_ent_list)
                    tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                    if tmp_triple_str not in lines_str_set:
                        break

                label = 0
                triples.append([i, int(tmp_head), int(relation_index), int(tail_ent_index)])
                labels.append(label)

                # corrupting tail
                tmp_tail = ''
                while True:
                    tmp_ent_list = set(entities)
                    # tmp_ent_list.remove(line[1])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_tail = random.choice(tmp_ent_list)
                    tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + line[2]
                    if tmp_triple_str not in lines_str_set:
                        break

                label = 0
                triples.append([i, int(head_ent_index), int(relation_index), int(tmp_tail)])
                labels.append(label)

                # corrupting head, tail
                tmp_head = ''
                tmp_tail = ''

                while True:
                    tmp_ent_list = set(entities)
                    # tmp_ent_list.remove(line[0])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_head = random.choice(tmp_ent_list)

                    tmp_ent_list = set(entities)
                    # tmp_ent_list.remove(line[1])
                    tmp_ent_list = list(tmp_ent_list)
                    tmp_tail = random.choice(tmp_ent_list)

                    # tep_rel_list = set(relations)
                    # tep_rel_list.remove(line[2])
                    # tep_rel_list = list(tep_rel_list)
                    # tmp_rel = random.choice(tep_rel_list)

                    tmp_triple_str = tmp_head + '\t' + tmp_tail + '\t' + line[2]
                    if tmp_triple_str not in lines_str_set:
                        break

                label = 0
                triples.append([i, int(tmp_head), int(relation_index), int(tmp_tail)])
                labels.append(label)

                for j in range(negative):
                    # # corrupting rel
                    tmp_rel = ''
                    while True:
                        # tmp_ent_list = set(entities)
                        tep_rel_list = set(relations)
                        # tep_rel_list.remove(line[2])
                        tep_rel_list = list(tep_rel_list)

                        tmp_rel = random.choice(tep_rel_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_rel
                        if tmp_triple_str not in lines_str_set:
                            break

                    label = 0
                    triples.append([i, int(head_ent_index), int(tmp_rel), int(tail_ent_index)])
                    labels.append(label)

                    # corrupting head, rel, tail
                    tmp_head = ''
                    tmp_tail = ''
                    tmp_rel = ''
                    while True:
                        tmp_ent_list = set(entities)
                        # tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)

                        tmp_ent_list = set(entities)
                        # tmp_ent_list.remove(line[1])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)

                        tep_rel_list = set(relations)
                        # tep_rel_list.remove(line[2])
                        tep_rel_list = list(tep_rel_list)
                        tmp_rel = random.choice(tep_rel_list)

                        tmp_triple_str = tmp_head + '\t' + tmp_tail + '\t' + tmp_rel
                        if tmp_triple_str not in lines_str_set:
                            break

                    label = 0
                    triples.append([i, int(tmp_head), int(tmp_rel), int(tmp_tail)])
                    labels.append(label)

                # if rnd <= 0.5:
                #     # corrupting head
                #     for j in range(negative):
                #         tmp_head = ''
                #         while True:
                #             tmp_ent_list = set(entities)
                #             tmp_ent_list.remove(line[0])
                #             tmp_ent_list = list(tmp_ent_list)
                #             tmp_head = random.choice(tmp_ent_list)
                #             tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                #             if tmp_triple_str not in lines_str_set:
                #                 break
                #
                #         label = 0
                #         triples.append([i, int(tmp_head), int(relation_index), int(tail_ent_index)])
                #         labels.append(label)
                # else:
                #     # corrupting tail
                #     tmp_tail = ''
                #
                #     for j in range(negative):
                #         while True:
                #             tmp_ent_list = set(entities)
                #             tmp_ent_list.remove(line[1])
                #             tmp_ent_list = list(tmp_ent_list)
                #             tmp_tail = random.choice(tmp_ent_list)
                #             tmp_triple_str = line[0] + '\t' + tmp_tail + '\t' + line[2]
                #             # print("tmp_triple_str",tmp_triple_str)
                #             if tmp_triple_str not in lines_str_set:
                #                 break
                #
                #         label = 0
                #         triples.append([i, int(head_ent_index), int(relation_index), int(tmp_tail)])
                #         labels.append(label)

        triple_data = TensorDataset(torch.tensor(triples), torch.tensor(labels, dtype=torch.long))

        triple_dataloader = DataLoader(triple_data, batch_size=2048)
        print("begin to obtain triple descriptions ... ")
        for step, batch in enumerate(tqdm(triple_dataloader, desc="\n obtain " + set_type + " triple description ")):

            temp_triple, temp_lab = batch
            guid = temp_triple[:, 0]
            head_index = temp_triple[:, 1]

            tail_index = temp_triple[:, 3]

            relation_index = temp_triple[:, 2]

            text_a, text_b, tail_c = self.ent_res.get_triple_des([int(i) for i in head_index],
                                                                 [int(i) for i in relation_index],
                                                                 [int(i) for i in tail_index])


            for i in range(len(temp_lab)):
                examples.append(InputExample(guid=guid[i], text_a=text_a[i], text_b=text_b[i], text_c=tail_c[i],
                                             label=str(temp_lab[i].cpu().detach().numpy().tolist())))
                # if i % 1000 == 0:
                #     print("example: ", i, guid[i], text_a[i], text_b[i], tail_c[i], str(temp_lab[i].cpu().detach().numpy().tolist()))
        print("len(examples)", len(examples))
        print("end to obtain triple descriptions ... ")
        return examples


def examples_to_features(examples, label_map, max_seq_length, tokenizer, i):
    features = []
    ex_index = i

    if ex_index % 10000 == 0:
        # logger.info("Writing example %d of %d" % (ex_index, len(features)))
        print("Writing example %d of %d" % (ex_index, len(examples)))

    example = examples[ex_index]
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    tokens_c = None

    if example.text_b and example.text_c:
        tokens_b = tokenizer.tokenize(example.text_b)
        tokens_c = tokenizer.tokenize(example.text_c)
        # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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

        # input data : tokens = "[CLS]" + tokens_a + "[SEP]" + tokens_b + "[SEP]" + tokens_c + "[SEP]"

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
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

    # if ex_index < 1 and print_info:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #         [str(x) for x in tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(
        InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label_id=label_id))


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    # pool = Pool(processes=6)
    # index_list = [i for i in range(len(examples))]
    # pfunc = partial(examples_to_features, examples, label_map, max_seq_length, tokenizer)
    # features = pool.map(pfunc, index_list)
    # pool.close()
    # pool.join()

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            # Modifies `tokens_a`, `tokens_b` and `tokens_c`in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
            # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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

            # input data : tokens = "[CLS]" + tokens_a + "[SEP]" + tokens_b + "[SEP]" + tokens_c + "[SEP]"

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
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

        if ex_index < 2 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        #
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    # print("len(features)",len(features))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    global index_a, index_b, index_c
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)

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

        # if total_length <= max_length:
        #
        #     break
        # else:
        #     try:
        #
        #         if len(tokens_a) > len(tokens_c):
        #             index_a = [i for i in range(len(tokens_a)) if tokens_a[i] == '.']
        #             if len(index_a) > 2:
        #                 del tokens_a[index_a[-2]:-1]
        #             else:
        #                 index_b = [i for i in range(len(tokens_b)) if tokens_b[i] == '.']
        #                 del tokens_b[index_b[-2]:-1]
        #         else:
        #             index_c = [i for i in range(len(tokens_c)) if tokens_c[i] == '.']
        #             del tokens_c[index_c[-2]:-1]
        #     except:
        #         print("failed to deal with ", index_a, tokens_a, index_b, tokens_b, index_c, tokens_c)
#

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
