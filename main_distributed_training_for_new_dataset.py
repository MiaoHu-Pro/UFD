from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
import numpy as np
import torch

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn import CrossEntropyLoss, NLLLoss
from sklearn import metrics
from data_process_utilities import KGProcessor, convert_examples_to_features, compute_metrics, EntDes, TrainSrc, \
    TrainExam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--pre_process_data",
                        default=None,
                        type=str,
                        required=True,
                        help="the path of pre_process training data .")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_link_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--load_pre_data",
                        action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--negative',
                        type=int,
                        default=3,
                        help="how many negative entities")

    parser.add_argument('--neighbours',
                        type=int,
                        default=5,
                        help="random seed for initialization")
    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help="random seed for initialization")

    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

    else:
        # DDP
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # set seed
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_features = None
    num_train_optimization_steps = 0

    if args.load_pre_data:

        print("read Ent src ... ")
        with open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
                args.neighbours) + "_neighbours_data_src.pkl", 'rb') as file:
            data_src = pickle.loads(file.read())
        # load data
        processor = data_src.processor
        label_list = data_src.label_list
        entity_list = data_src.entity_list
        task_name = data_src.task_name
        num_labels = len(label_list)
        # if args.do_train:
        #
        #     print("read Training src ... ")
        #     with open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
        #             args.neighbours) + "_neighbours_train_src.pkl", 'rb') as file:
        #         train_src = pickle.loads(file.read())
        #     load data
        # train_features = train_src.train_features

        # num_train_optimization_steps = int(
        #     len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        # if args.local_rank != -1:
        #     num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        #
        # num_train_optimization_steps = num_train_optimization_steps
    else:

        processors = {
            "kg": KGProcessor,
        }

        task_name = args.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % task_name)

        processor = processors[task_name]()
        label_list = processor.get_labels(args.data_dir)
        num_labels = len(label_list)  # num_labels = 2

        # create new des for entity
        processor.get_entity_res(file_path=args.data_dir, num_neighbours=args.neighbours, num_step=args.step)

        # read entities.txt, obtain entities
        entity_list = processor.get_entities(args.data_dir)

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            print("save EntDes ...")
            data_src = EntDes(_processor=processor, _label_list=label_list, _entity_list=entity_list,
                              _task_name=task_name)
            # save
            out_put = open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
                args.neighbours) + "_neighbours_data_src.pkl", 'wb')
            my_data_src = pickle.dumps(data_src)
            out_put.write(my_data_src)
            out_put.close()

    if args.do_train:
        print("construct train data features  begin ... ")
        if not os.path.exists(args.pre_process_data) and args.local_rank in [-1, 0]:
            os.makedirs(args.pre_process_data)

        # get training data set
        train_examples = processor.get_train_examples(args.data_dir, args.negative)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            print("save TrainSrc ... ")
            train_src = TrainSrc(_train_features=train_features,
                                 _num_train_optimization_steps=num_train_optimization_steps)
            out_put = open(args.pre_process_data + "/" + args.data_dir.split('/')[-1] + "_for_test_" + str(
                args.neighbours) + "_neighbours_train_src.pkl", 'wb')
            src = pickle.dumps(train_src)
            out_put.write(src)
            out_put.close()
        print("construct train data features finish ... ")

    # Prepare model
    # cache_dir: Where do you want to store the pre-trained models downloaded from s3
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    # args.bert_model: bert-base-cased
    from bert_model.BERT import MyBertForTokenHiddenState
    model = MyBertForTokenHiddenState.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels,
                                                      max_seq_length=args.max_seq_length)
    model.to(device)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    elif n_gpu > 1:

        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # Initial parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_data = None
    if args.do_train:

        # Loss Fun
        loss_fct = CrossEntropyLoss().to(args.local_rank)
        # loss_fct = NLLLoss().to(args.local_rank)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="\n No. " + str(e) + " epoch training ")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                logits = model(input_ids, segment_ids, input_mask, labels=None)

                # for CrossEntropyLoss loss function
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                # for NLLLoss loss function
                # loss = loss_fct(torch.log(logits.view(-1, num_labels)), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            print("Training loss: ", tr_loss, nb_tr_examples)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # torch.distributed.get_rank() == 0 Save only once on process 0, avoiding saving duplicates
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_link_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Load a trained model and vocabulary that you have fine-tuned
        model = MyBertForTokenHiddenState.from_pretrained(args.output_dir, num_labels=num_labels,
                                                          max_seq_length=args.max_seq_length)

        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        train_triples = processor.get_train_triples(args.data_dir)
        dev_triples = processor.get_dev_triples(args.data_dir)
        test_triples = processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        model.to(device)

        test_data = ["test2id.txt", "unseen_relation_inverse_test2id.txt",
                     "unseen_relation_others_test2id.txt",
                     "unseen_both_entity_test2id.txt", "unseen_head_entity_test2id.txt",
                     "unseen_tail_entity_test2id.txt",
                     "unseen_facts_inverse_test2id.txt", "unseen_facts_others_test2id.txt", "train"]

        # get test data
        for _data_test in test_data:

            # get test data
            # eval_examples = processor.get_test_examples(args.data_dir)
            if args.do_train and _data_test == "train":

                eval_data = train_data
            elif args.data_dir.split('/')[-1] == "WN18RR" and _data_test== "test2id.txt":

                eval_examples = processor.get_other_graph_test_examples(args.data_dir, _data_test, args.negative)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer)

                logger.info("***** Running Prediction *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


            else:
                if _data_test == "train":
                    continue

                eval_examples = processor.get_every_test_examples(args.data_dir, _data_test, args.negative)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer)

                logger.info("***** Running Prediction *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            # eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            loss_fct = CrossEntropyLoss()
            # loss_fct = NLLLoss()

            for batch in tqdm(eval_dataloader, desc="\nTesting"):

                model.eval()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))  # CrossEntropyLoss()

                # tmp_eval_loss = loss_fct(torch.log(logits.view(-1, num_labels)), label_ids.view(-1)) # loss_fct = NLLLoss()

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]

            all_label_ids = all_label_ids.numpy()

            preds = np.argmax(preds, axis=1)

            result = compute_metrics(task_name, preds, all_label_ids)

            print("The result of Triple Classification : \n")
            acc = metrics.accuracy_score(all_label_ids, preds)
            recall = metrics.recall_score(all_label_ids, preds)
            precision = metrics.precision_score(all_label_ids, preds)
            F1 = metrics.f1_score(all_label_ids, preds)

            tn, fp, fn, tp = metrics.confusion_matrix(all_label_ids, preds).ravel()

            print("================== " + args.data_dir.split('/')[
                -1] + " ==  " + _data_test + "  ====================\n\n")
            print("acc is : ", acc)
            print("recall is : ", recall)
            print("precision is : ", precision)
            print("F1 is : ", F1)
            print("tn, fp, fn, tp : ", tn, fp, fn, tp)

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['acc'] = acc
            result['recall'] = recall
            result['precision'] = precision
            result['F1'] = F1

            result['tn_fp_fn_tp'] = (tn, fp, fn, tp)

            output_eval_file = os.path.join(args.output_dir, "result_of_triple_classification.txt")
            with open(output_eval_file, "a+") as writer:
                writer.write("==================  " + args.data_dir.split('/')[
                    -1] + " ==  " + _data_test + " ====================\n")
                logger.info("***** Test results *****")

                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            write_true_ids_and_preds(args, _data_test, all_label_ids, preds)

        # run link prediction
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])
        count = 0
        for test_triple in test_triples:
            count += 1
            print(count, "\n")
            head = test_triple[0]
            tail = test_triple[1]
            relation = test_triple[2]

            head_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, tail, relation]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        head_corrupt_list.append(tmp_triple)

            #
            tmp_examples = processor.to_create_examples(head_corrupt_list, "test", args.data_dir)
            tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer,
                                                        print_info=False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for temp data
            # eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            preds = []
            for batch in tqdm(eval_dataloader, desc="Testing"):
                model.eval()

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # input_ids = input_ids.to(device)
                # input_mask = input_mask.to(device)
                # segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)

            preds = preds[0]
            # get the dimension corresponding to current label 1
            # print(preds, preds.shape)
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            # print(rel_values, rel_values.shape)
            _, argsort1 = torch.sort(rel_values, descending=True)
            # print(max_values)
            # print(argsort1)
            argsort1 = argsort1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]
            print('left: ', rank1)
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            if rank1 < 10:
                top_ten_hit_count += 1

            # head entity
            tail_corrupt_list = [test_triple]
            for corrupt_ent in entity_list:
                if corrupt_ent != tail:
                    tmp_triple = [head, corrupt_ent, relation]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        tail_corrupt_list.append(tmp_triple)

            tmp_examples = processor.to_create_examples(tail_corrupt_list, "test", args.data_dir)
            # print(len(tmp_examples))
            tmp_features = convert_examples_to_features(tmp_examples, label_list, args.max_seq_length, tokenizer,
                                                        print_info=False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for temp data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            preds = []

            for batch in tqdm(eval_dataloader, desc="Testing"):

                model.eval()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # input_ids = input_ids.to(device)
                # input_mask = input_mask.to(device)
                # segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)

                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)

            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)
            print('right: ', rank2)
            print('mean rank until now: ', np.mean(ranks))
            if rank2 < 10:
                top_ten_hit_count += 1
            print("hit@10 until now: ", top_ten_hit_count * 1.0 / len(ranks))

            file_prefix = str(args.data_dir[7:]) + "_" + str(args.train_batch_size) + "_" + str(
                args.learning_rate) + "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs)
            # file_prefix = str(args.data_dir[7:])
            f = open(file_prefix + '_ranks.txt', 'a')
            f.write(str(rank1) + '\t' + str(rank2) + '\n')
            f.close()
            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        for i in [0, 2, 9]:
            logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))


def write_true_ids_and_preds(args, _data_test, all_label_ids, preds):
    output_eval_file = os.path.join(args.output_dir, _data_test + "_label_result_ture_pre.txt")
    f = open(output_eval_file, 'a')
    length = len(all_label_ids)
    for i in range(length):
        f.write(str(all_label_ids[i]) + '\t' + str(preds[i]) + '\n')

    f.close()


if __name__ == "__main__":
    main()

# new_dataset
# python  main_distributed_training_for_new_dataset.py   --task_name kg --do_train --load_pre_data  --do_link_predict --negative 2  --data_dir ./data/new_dataset --pre_process_data ./pre_process_data  --bert_model bert-base-cased  --max_seq_length 100 --train_batch_size 32   --learning_rate 5e-5  --num_train_epochs 3.0  --output_dir ./output_new_dataset_distributed_liner_layer_for_des_e3_5neighbours_2negative_gpus_2_train_batch_32_gradient_4/  --gradient_accumulation_steps 4  --eval_batch_size 32  --neighbours 5
# FB15K237
# python  main_distributed_training_for_new_dataset.py   --task_name kg --do_train --load_pre_data  --do_link_predict --negative 2  --data_dir ./data/FB15K237 --pre_process_data ./pre_process_data  --bert_model bert-base-cased  --max_seq_length 100 --train_batch_size 32   --learning_rate 5e-5  --num_train_epochs 3.0  --output_dir ./output_FB15K237_dataset_distributed_liner_layer_for_des_e3_5neighbours_2negative_gpus_2_train_batch_32_gradient_4/  --gradient_accumulation_steps 4  --eval_batch_size 32  --neighbours 0
