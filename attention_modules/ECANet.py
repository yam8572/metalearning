import torch
from torch import nn
from torch.nn.parameter import Parameter
import math

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # kernel adaptive
        t = int(abs(math.log(channel,2)+1)/2)
        # 取近 t 的奇數
        k_size = t if t % 2 else t+1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("ECA.x=",x.size())
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # print("ECA.y.avgpool=",y.size())
        # Two different branches of ECA module
        """
        y.squeeze(-1)對輸入特徵圖y進行維度調整，將最後一維度（一般是通道維度）壓縮掉
        transpose(-1, -2)進行最後兩個維度進行交換>>將特徵圖的通道維度和空間維度進行交換，以便後續的捲積操作
        做完卷積後再transpose(-1, -2).unsqueeze(-1)維度交換回來並升回原維度
        """
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print("ECA.y.conv=",y.size())
        # Multi-scale information fusion
        y = self.sigmoid(y)
        # print("ECA.y.sigmoid=",y.size())
        output = x * y.expand_as(x)
        # print("ECA.y.output=",output.size())
        
        return output
        