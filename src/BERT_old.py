import torch
import torch.nn as nn
from loss.MarginLoss import MarginLoss
from src.Model import Model

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()

# import gc
# del variables #delete unnecessary variables
# gc.collect()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()


class MyBertForTokenHiddenState(BertPreTrainedModel):
    print("MyBertForTokenHiddenstate .. ")
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(MyBertForTokenHiddenState, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # logits = self.classifier(sequence_output)
        # return loss
        # else:

        return sequence_output, pooled_output


class Bert(Model):

    def __init__(self, config):
        super(Bert, self).__init__(config)

        self.bert_model = MyBertForTokenHiddenState.from_pretrained(self.config.bert_model,
                                                                    cache_dir=self.config.cache_dir,
                                                                    num_labels=self.config.num_labels)

        self.active_layer = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.trans_layer = nn.Linear(self.config.num_labels, 1)
        self.score_layer = nn.Linear(self.config.max_seq_length, 1)

        self.batch_size = self.config.batch_size
        self.loss = MarginLoss(margin=2)
        self.criterion = nn.Softplus()  #
        self.init_parameters()



    def init_parameters(self):

        nn.init.xavier_uniform_(self.score_layer.weight.data)

        # param_optimizer = list(self.bert_model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    def _calc(self, triple_matrix):  # 论文中的评分函数

        logics = self.active_layer(self.bert_model.classifier(triple_matrix))
        # print("logics : ", logics.shape, logics)

        _score = self.trans_layer(logics)  # 44 * 300 *1,
        # print("_score : ", _score.shape, _score)

        _score = torch.reshape(_score, (-1, _score.shape[1]))
        # print("_score : ", _score.shape, _score)
        score = self.score_layer(self.active_layer(_score)).view(-1)  # 44 *1

        return -score

    def loss_fun(self, score, labels, regul):  # 论文中的损失函数
        return torch.mean(self.criterion(score * labels)) + self.config.lmbda * regul

    def forward(self):
        print("training BERT ... ")
        """
        input_ids, segment_ids, input_mask, labels
        #define a new function to compute loss values for both output_modes
        sequence_output,pooled_output = model(input_ids, segment_ids, input_mask, labels=None)
        print('\n',logits, logits.shape,'\n')

        # 设计评分函数，计算loss

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        """

        # h = self.ent_embeddings(self.batch_h)  # batch_h 对应下标，ent_embeddings，是一个字典 （"index"：描述）
        # r = self.rel_embeddings(self.batch_r)
        # t = self.ent_embeddings(self.batch_t)

        # input_ids, segment_ids, input_mask, labels
        # print(self.batch_input_ids)
        # print(self.batch_input_mask)
        # print(self.batch_segment_ids)
        # print(self.batch_label_ids)

        input_ids = self.batch_input_ids
        segment_ids = self.batch_segment_ids

        input_mask = self.batch_input_mask

        labels = self.batch_label_ids

        # print("forward()+", labels[:self.batch_size], "len: ", len(labels[:self.batch_size]))
        # print("forward() -", labels[self.batch_size:])

        sequence_output, pooled_output = self.bert_model(input_ids, segment_ids, input_mask, labels=None)

        # print("sequence_output.shape)", sequence_output.shape)
        # print("pooled_output.shape", pooled_output.shape)
        # print("\n")
        # print("sequence_output[0])", sequence_output[0].shape)
        #
        # print("sequence_output[0])", sequence_output[0])
        # print("sequence_output[1])", sequence_output[1])
        # print("sequence_output[2])", sequence_output[2])
        # print("sequence_output[3])", sequence_output[3])

        # print("sequence_output[1][0])", sequence_output[1][0])
        # print("sequence_output[1][0])", sequence_output[2][0])
        # print("sequence_output[1][-1])", sequence_output[1][-1])
        # print("sequence_output[1][-1])", sequence_output[2][-1])

        # print("pooled_output", pooled_output)
        #
        # time.sleep(10)

        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(pooled_output.view(-1, self.num_labels), labels.view(-1))

        score = self._calc(sequence_output)

        # print("score : ", score)
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        # print("p_score : ", p_score, "len(p_score) ", len(p_score))
        # print("n_score : ", n_score)

        # MarginLoss
        score_loss = self.loss(p_score, n_score)

        print("score_loss", score_loss)

        # loss_res = self.loss(p_score, n_score)
        # print("score: ",len(score),score)

        # regularization
        # l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        l2_reg = 0
        for W in self.bert_model.parameters():
            l2_reg = l2_reg + W.norm(2)

        for W in self.trans_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        for W in self.score_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        # ConvKB loss function
        # all_loss = self.loss_fun(score, labels, l2_reg)

        # all_loss = score_loss + l2_reg
        # there is no regulation,....
        all_loss = score_loss

        # print("l2_reg", l2_reg)
        print("all_loss", all_loss)

        return all_loss

    def predict(self):

        input_ids = self.batch_input_ids
        segment_ids = self.batch_segment_ids

        input_mask = self.batch_input_mask

        labels = self.batch_label_ids

        # print("forward ",labels)
        sequence_output, pooled_output = self.bert_model(input_ids, segment_ids, input_mask, labels=None)

        score = self._calc(sequence_output)

        return score.cpu().data.numpy()
