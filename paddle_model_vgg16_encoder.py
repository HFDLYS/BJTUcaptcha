import paddle
import paddle.nn as nn
import encoder
from encoder import Encoder


class CRNN(paddle.nn.Layer):
    def __init__(self, n_classes, input_shape=(3, 42, 130)):
        super().__init__()
        self.input_shape = input_shape
        # VGG16
        channels = [64, 128, 256, 512, 512]
        layers = [2, 2, 3, 3, 3]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, (2,1), (2,1)]
        modules = []

        def addmod(in_channels, out_channels, kernel_size):
            modules.append(nn.Conv2D(in_channels, out_channels, kernel_size,
                                     padding=(kernel_size // 2, kernel_size // 2)))
            modules.append(nn.BatchNorm2D(out_channels))
            modules.append(nn.ReLU())

        last_channel = input_shape[0]
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                addmod(last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules.append(nn.MaxPool2D(k_pool))
        modules.append(nn.Dropout(0.25))
        self.cnn = nn.Sequential(*modules)

        # -------- 换成 Encoder ----------
        self.encoder = Encoder(in_dim=channels[-1],
                               hidden_dim=256,
                               n_head=8,
                               num_layers=2)

        self.fc = nn.Linear(in_features=channels[-1], out_features=n_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose([0, 3, 1, 2])  # 将宽度维度转为序列长度
        x = x.reshape([x.shape[0], x.shape[1], -1])  # [batch_size, seq_len, features]
        x = self.encoder(x) # [batch_size, seq_len, features]
        x = self.fc(x)  # (batch,seq,n_classes)
        x = x.transpose([1, 0, 2])  # [seq_len, batch_size, features]
        #print(x)
        return x

    def get_seq_len(self):
        x = paddle.zeros([1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        x = self.cnn(x)
        x = x.transpose([0,3,1,2])
        return x.shape[1]

    def _initialize_weights(self):
        import paddle
        import paddle.nn as nn
        from paddle.nn.initializer import KaimingNormal, XavierNormal, Constant, Orthogonal

        # 定义初始化器
        kaiming_normal = KaimingNormal(nonlinearity='relu')
        xavier_uniform = XavierNormal()
        constant_zero = Constant(value=0.)
        constant_one = Constant(value=1.)

        for name, layer in self.named_sublayers():
            # 跳过没有参数的层（如ReLU、Dropout等）
            if not hasattr(layer, 'parameters'):
                continue

            # 卷积层初始化
            if isinstance(layer, nn.Conv2D):
                kaiming_normal(layer.weight)
                if layer.bias is not None:
                    constant_zero(layer.bias)

            # BatchNorm层初始化
            elif isinstance(layer, nn.BatchNorm2D):
                constant_one(layer.weight)
                constant_zero(layer.bias)

            # LSTM层特殊处理
            elif isinstance(layer, nn.LSTM):
                for param_name, param in layer.named_parameters():
                    # Paddle中LSTM参数命名规则：
                    # - weight_ih_l{k} : 输入到隐藏的权重
                    # - weight_hh_l{k} : 隐藏到隐藏的权重
                    if 'weight_ih' in param_name:
                        xavier_uniform(param)
                    elif 'weight_hh' in param_name:
                        Orthogonal()(param)
                    elif 'bias' in param_name:
                        constant_zero(param)

            # 全连接层初始化
            elif isinstance(layer, nn.Linear):
                kaiming_normal(layer.weight)
                if layer.bias is not None:
                    constant_zero(layer.bias)

