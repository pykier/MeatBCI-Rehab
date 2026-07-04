
import os

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

from .utils import (
    Concat,
    Conv2dWithConstraint,
    LayerNormalization,
    LogmLayer,
    PointwiseConv2d,
    R_attention,
    Vec,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# 保存准确率的字典




class BaseModel(nn.Module):
    def __init__(self,  configs, *args,**kwargs):
        """
        初始化模型
        :param configs: 配置字典，包含模型所需的参数
        :param args: 其他参数
        :param kwargs: 其他关键字参数

        Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  SecNet: A second order neural network for MI-EEG.  *Information Processing & Management*, 62(3), 104012.
        """
        class_num = configs.get('class_num', 2)
        channelNum = configs.get('channelNum', 58)
        width = configs.get('width', 300)
        sampleNum = configs.get('sampleNum', configs.get('n_samples', 1000))
        drop_att = configs.get('drop_att', 0.2)
        p = configs.get('p', 3)
        if not isinstance(class_num, int) or class_num <= 0:
            raise ValueError("class_num must be a positive integer")
        if not isinstance(channelNum, int) or channelNum <= 0:
            raise ValueError("channelNum must be a positive integer")
        if not isinstance(width, int) or width <= 0:
            raise ValueError("input_width must be a positive integer")
        if not isinstance(sampleNum, int) or sampleNum <= 0:
            raise ValueError("sampleNum must be a positive integer")

        super(BaseModel, self).__init__()
        self.input_width = width
        self.input_channels = 150
        self.channelNum = channelNum
        self.sampleNum = sampleNum

        try:
            # 创建卷积块
            self.block1 = self.create_block(15)
            self.block2 = self.create_block(95)
            self.block3 = self.create_block(55)
            self.fusion=Concat()
            # 创建其他层
            in_size=self.input_channels * 3 if isinstance(self.fusion,Concat) else self.input_channels
            self.Sconv3 = nn.Sequential(PointwiseConv2d(in_size, 100))
            self.attention_module=R_attention(100,p,drop_att)
            self.log_layer1 = LogmLayer(100, vectorize=False)
            self.vec = Vec(100)
            self.FC = nn.Sequential(nn.Linear(5050, class_num))

            # 初始化参数
            self.apply(self.initParms)
        except Exception as e:
            print(f"An error occurred: {e}")


    def create_block(self, kernel_size):
        """
        创建卷积块
        :param kernel_size: 卷积核大小
        :return: 卷积块
        """
        return nn.Sequential(

            # nn.Conv2d(1, self.input_channels, kernel_size=(1, 1), padding='same',
            #           bias=False, groups=1),

            Conv2dWithConstraint(1, self.input_width, kernel_size=(self.channelNum, 1), padding=0, bias=False,
                                 groups=1),
            PointwiseConv2d(self.input_width, self.input_channels),
            LayerNormalization(self.sampleNum),

            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=(1, kernel_size), padding='same',
                      bias=False, groups=self.input_channels),
            LayerNormalization(self.sampleNum),

        )
    def initParms(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, feature):
        if len(feature.shape)==3:
            feature=feature.unsqueeze(1)
        # print('feature',feature.shape)
        h1=self.block1(feature)
        # print('h1', h1.shape)
        h2=self.block2(feature)
        h3=self.block3(feature)
        h=self.fusion([h1,h2,h3])

        h=self.Sconv3(h)
        # add attention
        # print('h1',h.shape)
        h = self.attention_module(h)
        # print(h.shape)
        # print('h2', h.shape)
        feature=self.log_layer1(h)
        # print('feature', feature.shape)
        h = self.FC(self.vec(feature))
        # print('h3', h.shape)
        return h,feature.flatten(1)


class SECNet(nn.Module):
    """Skorch-compatible SECNet wrapper for rehab MI classification.

    ``BaseModel`` returns both logits and an intermediate feature vector.  The
    training and online prediction pipeline expects a classifier module that
    returns logits only, so this wrapper keeps the original backbone intact and
    exposes the standard output shape.
    """

    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes,
        width=300,
        dropout=0.2,
        p=3,
    ):
        super().__init__()
        self.backbone = BaseModel(
            {
                "class_num": int(n_classes),
                "channelNum": int(n_channels),
                "sampleNum": int(n_samples),
                "width": int(width),
                "drop_att": float(dropout),
                "p": int(p),
            }
        )

    def forward(self, X):
        X = X.float()
        logits, _ = self.backbone(X)
        return logits


def create_secnet_estimator(
    n_channels,
    n_samples,
    n_classes,
    epochs=150,
    early_stopping_patience=80,
    batch_size=16,
    learning_rate=1e-3,
    val_ratio=0.2,
    random_state=42,
    dropout=0.2,
    width=300,
    verbose=1,
):
    module = SECNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        width=width,
        dropout=dropout,
    )
    return NeuralNetClassifier(
        module,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-4,
        lr=float(learning_rate),
        max_epochs=max(1, int(epochs)),
        batch_size=max(1, int(batch_size)),
        train_split=ValidSplit(
            float(val_ratio),
            stratified=True,
            random_state=int(random_state),
        ),
        iterator_train__shuffle=True,
        callbacks=[
            (
                "lr_scheduler",
                LRScheduler(
                    "CosineAnnealingLR",
                    T_max=max(1, int(epochs) - 1),
                ),
            ),
            (
                "estoper",
                EarlyStopping(
                    patience=max(1, int(early_stopping_patience)),
                    load_best=True,
                ),
            ),
        ],
        device="cpu",
        verbose=verbose,
    )





