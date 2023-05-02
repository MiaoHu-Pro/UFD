import torch
import torch.nn as nn
from src.Model import Model

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class ConvKB(Model):

    def __init__(self, config):
        super(ConvKB, self).__init__(config)

        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)  # 定义entity embedding矩阵 h ,r,t 的纬度 hidden_size = 50或100
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)  # 定义relation embedding矩阵

        self.conv1_bn = nn.BatchNorm2d(1)  # 归一化操作
        self.conv_layer = nn.Conv2d(1, # number of input channels
                                    self.config.out_channels,# number of output channels 128,卷积的个数
                                    (self.config.kernel_size, 3)# size of the kernel 1*3 : 一行三列
                                    )  # kernel size x 3 # 定义卷积层 out_channels = num_of_filters

        self.conv2_bn = nn.BatchNorm2d(self.config.out_channels)
        self.dropout = nn.Dropout(self.config.convkb_drop_prob)  # dropout操作
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear((self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels, 1, bias=False)


        self.criterion = nn.Softplus()  #
        self.init_parameters()

    def init_parameters(self):
        if self.config.use_init_embeddings == False:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        else:
            self.ent_embeddings.weight.data = self.config.init_ent_embs
            self.rel_embeddings.weight.data = self.config.init_rel_embs

        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, h, r, t): #论文中的评分函数

        h = h.unsqueeze(1) # bs x 1 x dim 9548 * 1 * 50
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)


        # print("h,: " , h.shape )

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim 输入
        # print("conv_input,: " , conv_input.shape )
        conv_input = conv_input.transpose(1, 2)
        # print("conv_input,: " , conv_input.shape )
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        # print("conv_input,: " , conv_input.shape )
        conv_input = self.conv1_bn(conv_input)  # 归一化
        # print(conv_input.shape) # ([9548, 1, 50, 3])
        out_conv = self.conv_layer(conv_input) #  卷积
        # print("out_conv_layer : ", out_conv.shape)
        out_conv = self.conv2_bn(out_conv) #  归一化
        # print("conv_conv2_bn : ", out_conv.shape)
        out_conv = self.non_linearity(out_conv) #  非线性变换 ReLU
        # print("non_linearity : ", out_conv.shape)
        out_conv = out_conv.view(-1, (self.config.hidden_size - self.config.kernel_size + 1) * self.config.out_channels)# 将结果拉伸为一个向量
        # print("out_conv.view : ", out_conv.shape)
        input_fc = self.dropout(out_conv)  #随机droput
        #  print("input_fc : ", input_fc.shape) # l卷积之后做拉伸，
        score = self.fc_layer(input_fc).view(-1)  #全链接层

        return -score

    def loss(self, score, regul): #论文中的损失函数
        return torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul

    def forward(self):

        """
        input_ids, segment_ids, input_mask, labels

        #define a new function to compute loss values for both output_modes
        sequence_output,pooled_output = model(input_ids, segment_ids, input_mask, labels=None)
        print('\n',logits, logits.shape,'\n')

        # 设计评分函数，计算loss

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))



        """


        h = self.ent_embeddings(self.batch_h)  # batch_h 对应下标，ent_embeddings，是一个字典 （"index"：描述）
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)

        score = self._calc(h, r, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        return self.loss(score, l2_reg)

    def predict(self):

        h = self.ent_embeddings(self.batch_h)
        r = self.rel_embeddings(self.batch_r)
        t = self.ent_embeddings(self.batch_t)
        score = self._calc(h, r, t)

        return score.cpu().data.numpy()
