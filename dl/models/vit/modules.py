import torch as t
from torch import nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb
from torch.functional import F
from ...base_torch_modules.gaussian_noise import GaussianNoise


class ResidualAxialBlock(nn.Module):
    def __init__(self, embedding_dim, num_dimentions, num_heads, droupout) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention = AxialAttention(embedding_dim, num_dimentions, num_heads, 4, -1, True)
        self.do = nn.Dropout(droupout)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.attention(x)
        out = z + x
        out = self.act(out)
        out = self.do(out)
        return out


class AxialClassifier(nn.Module):
    def __init__(
        self, num_classes=7, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.01
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = AxialPositionalEmbedding(self.embedding_dim, (48, 48), -1)
        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, self.embedding_dim, bias=False), nn.ReLU()
        )

        self.attentions = nn.Sequential()
        for i in range(self.num_layers):
            # we need setattr so that lightning can find the module
            self.__setattr__(
                "layer_{}".format(i),
                ResidualAxialBlock(self.embedding_dim, 2, self.num_heads, self.dropout),
            )
            self.attentions.add_module(
                "layer_{}".format(i), self.__getattr__("layer_{}".format(i))
            )

        self.avg_pool_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        self.classifier = nn.Linear(48 * 48, self.num_classes)
        # self.big_classifier = nn.Sequential(
        #     nn.Linear(48 * 48, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.num_classes),
        # )

    def forward(self, x):
        # ipdb.set_trace()
        x = x.permute(0, 2, 3, 1)
        x = self.embedding_encoder(x)
        x = self.embedding(x)
        x = self.attentions(x)

        x = x.max(dim=-1)[0]
        # x = x.permute(0, 3, 1, 2)
        # x = self.avg_pool_classifier(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.softmax(x, dim=1)
