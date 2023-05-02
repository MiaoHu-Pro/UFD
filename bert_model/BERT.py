import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

torch.cuda.empty_cache()

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

    def __init__(self, config, num_labels, max_seq_length):
        super(MyBertForTokenHiddenState, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.softmax_func = nn.Softmax(dim=1)
        self.apply(self.init_bert_weights)

        self.max_seq_length = max_seq_length

        print("__init__")
        print("max_seq_length : ",self.max_seq_length)

        if self.max_seq_length == 100:
            self.hidden_dim = 40

        elif self.max_seq_length == 200:
            self.hidden_dim = 80

        elif self.max_seq_length == 300:
            self.hidden_dim = 160

        elif self.max_seq_length == 400:
            self.hidden_dim = 240

        elif self.max_seq_length == 500:
            self.hidden_dim = 320

        print("self.hidden_dim : ", self.hidden_dim)

        # --------------------------------------------
        self.conv1 = nn.Sequential(  # 输入大小 (1,28,28)
            nn.Conv2d(
                in_channels=1,  # 说明是灰度图
                out_channels=3,  # 要得到多少个特征图
                kernel_size=(20, 48),  # 卷积核的大小
                stride=1,  # 步长
                padding=2),  # 边缘填充的大小
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)  # 池化操作 (2 * 2) 输出结果为: (16,14,14)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 2, (10, 24)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(3))  # 输出 (32, 7, 7)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 1, (5, 12)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 输出 (32, 7, 7)

        self.out = nn.Linear(self.hidden_dim, num_labels)  # 全连接输入分类 300- 160 , 200-80

        # self.out = nn.Linear(80, 1)  # 全连接输入分类 200 - 80
        # self.out = nn.Linear(240, 1) # 全连接输入分类 400 - 240
        # self.out = nn.Linear(340, 1)# 全连接输入分类 512 - 340
        # self.init_parameters()

    def _calc(self, triple_matrix):  # 论文中的评分函数

        n, r, c = triple_matrix.shape

        triple_matrix = torch.reshape(triple_matrix, (n, -1, r, c))
        # print("triple_matrix.shape",triple_matrix.shape)
        # print(triple_matrix.shape)
        x = self.conv1(triple_matrix)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten操作,结果为 (batch_size, 32*7*7)
        # print(x.shape)
        score = self.out(x)

        return score

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        # Con layer
        # sequence_output = self.dropout(sequence_output)
        # score = self._calc(sequence_output)

        # other score fun ..( Linear layer  )
        pooled_output = self.dropout(pooled_output)
        score = self.classifier(pooled_output)

        # score = self.softmax_func(score)

        return score

