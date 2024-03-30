"""
kernel_size=1意味著卷積核的大小為1x1，
這表示卷積核只與輸入的每個通道的單一像素進行卷積操作。在這種情況下，
nn.Conv2d只是在進行通道之間的線性變換，沒有進行空間上的捲積操作，
因此可以視為全連接層
"""
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1,
                            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1,
                            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("se.x=",x.size())
        # b, c, _ , _= x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        # module_input = x
        y = self.avg_pool(x)
        # print("SEC.y.avgpool=",y.size())
        y = self.fc1(y)
        # print("SEC.y.fc1=",y.size())
        y = self.relu(y)
        # print("SEC.y.relu=",y.size())
        y = self.fc2(y)
        # print("SEC.y.fc2=",y.size())
        y = self.sigmoid(y)
        # print("SEC.y.sigmode=",y.size())
        output = x * y.expand_as(x)
        # print("SEC.output=",output.size())

        return output
        return module_input * x