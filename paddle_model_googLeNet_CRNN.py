import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class InceptionModule(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2D(in_channels, out_channels // 2, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(out_channels // 4, out_channels // 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))
        branch2 = F.relu(self.branch2(x))
        return paddle.concat([branch1, branch2], axis=1)


class CRNN(nn.Layer):
    def __init__(self, n_classes, input_shape=(3, 42, 130)):
        super().__init__()
        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            # 第一层
            nn.Conv2D(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),  # 21x65
            # 第二层
            InceptionModule(64, 128),
            nn.MaxPool2D(kernel_size=2),  # 11x33
            # 第三层
            InceptionModule(128, 256),
            nn.MaxPool2D(kernel_size=(2, 1)),  # 6x33
            # 第四层：
            nn.Conv2D(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),  # 3x33
            # 降维
            nn.Conv2D(512, 256, kernel_size=1),
            nn.AdaptiveAvgPool2D((1, 32))
        )


        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            direction='bidirectional'
        )

        self.fc = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(0.3)
        self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose([0, 3, 1, 2])
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose([1, 0, 2])
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                nn.initializer.XavierNormal()(layer.weight)
                if layer.bias is not None:
                    nn.initializer.Constant(0.)(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.initializer.XavierNormal()(layer.weight)
                nn.initializer.Constant(0.)(layer.bias)

    def get_seq_len(self):
        dummy_input = paddle.zeros([1] + list(self.input_shape))
        x = self.cnn(dummy_input)
        return x.shape[-1]

    def count_parameters(self):
        """计算模型参数量"""
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
        return total_params

