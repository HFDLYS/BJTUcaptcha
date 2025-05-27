import math
import paddle
import paddle.nn as nn

class PositionalEncoding(nn.Layer):
    def __init__(self,d_model,max_len=100):
        super().__init__()
        pe = paddle.zeros(shape=(max_len,d_model))
        position = paddle.arange(0,max_len,dtype="float32").unsqueeze(axis=1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2) *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        self.register_buffer("pe",pe)  # 避免参数被

    def forward(self,x):
        ## 输出(1,seq,d_model) -- > 1是用来broadcast的
        return x + self.pe[:x.shape[1]].unsqueeze(0)

class Encoder(nn.Layer):
    def __init__(self,in_dim,hidden_dim=256,n_head=8,num_layers=2):
        super().__init__()

        self.in_projection = nn.Linear(in_dim,hidden_dim)
        #请确保 输入的是 (batch, seq, d_model)
        self.pos_encoding = PositionalEncoding(hidden_dim) #最多支持seq=100

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim,n_head,hidden_dim*2,dropout=0.1)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.out_projection = nn.Linear(hidden_dim, in_dim)


    def forward(self,x):
        # x: (batch,seq,d_model)
        x = self.in_projection(x) #(batch,seq,hidden)
        x = self.pos_encoding(x) #(batch,seq,hidden)
        x = self.encoder(x) #(batch,seq,hidden)
        x = self.out_projection(x) #(batch,seq,d_model)
        return x
