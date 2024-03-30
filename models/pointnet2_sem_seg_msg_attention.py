import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils_attention import PointNetSetAbstractionMsg,PointNetFeaturePropagation
# from attention_modules.BAM import BAM
# from attention_modules.CBAM import CBAM
# from attention_modules.GAM import GAM, GAMFP
# from attention_modules.ECANet import eca_layer
# from attention_modules.SENet import SELayer
# from attention_modules.scSE import scSE
# from attention_modules.ECA_scSE import ECA_scSE
# from attention_modules.ECA_ResNet import eca_resnet50


class get_model(nn.Module):
    def __init__(self, num_classes, attention_type):
        super(get_model, self).__init__()

        # print(attention_type)
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]], attention_type)
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]], attention_type)
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], attention_type)
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], attention_type)
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256], attention_type)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], attention_type)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], attention_type)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], attention_type)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.att_block_types = attention_type
        # add attention layers
        # if(self.att_block_types=='scSE'):
        #     self.attention = scSE(128)
        # elif(self.att_block_types=='ECA_scSE'):
        #     self.attention = ECA_scSE(128)
        # elif(self.att_block_types=='BAM'):
        #     self.attention = BAM(128)
        # elif(self.att_block_types=='CBAM'):
        #     self.attention = CBAM(128)
        # elif(self.att_block_types=='ECANet'):
        #     self.attention = eca_layer(128)
        # elif(self.att_block_types=='SENet'):
        #     self.attention = eca_layer(128)
        # elif(self.att_block_types=='ECA_ResNet'):
        #     self.attention = eca_resnet50(num_classes=128)
        # elif(self.att_block_types=='GAM'):
        #     self.attention = GAM(128,128)

        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # print("xyz.shape",xyz.shape) # torchsize [nway * kshot, 9, 4096]
        # print("xyz",xyz)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:] # PointNet++ 只有使用 XYZ 去訓練
        # print("l0_xyz.shape",l0_xyz.shape) # torchsize [3 , 9, 4096]
        # print("l0_points.shape",l0_points.shape) # torchsize [nway * kshot, 9, 4096]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.size(), l1_points.size()) #torch.Size([nway * kshot, 3, 1024]) #torch.Size([nway * kshot, 96, 1024])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.size(), l2_points.size()) #torch.Size([nway * kshot, 3, 256]) #torch.Size([nway * kshot, 256, 256])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.size(), l3_points.size()) #torch.Size([nway * kshot, 3, 64]) #torch.Size([nway * kshot, 512, 64])
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.size(), l4_points.size()) #torch.Size([nway * kshot, 3, 16]) #torch.Size([nway * kshot, 1024, 16])

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points.size()) #torch.Size([nway * kshot, 256, 64])
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.size()) #torch.Size([nway * kshot, 256, 256])
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.size()) #torch.Size([nway * kshot, 128, 1024])
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # print(l0_points.size()) #torch.Size([nway * kshot, 128, 1024])

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        # x = self.drop1(F.relu(self.bn1(self.attention(self.conv1(l0_points)))))

        # print(x.size()) #torch.Size([nway * kshot, 128, 1024])
        x = self.conv2(x)
        # print(x.size()) #torch.Size([nway * kshot, 13, 1024])
        x = F.log_softmax(x, dim=1)
        # print(x.size()) #torch.Size([nway * kshot, 13, 1024])
        x = x.permute(0, 2, 1)
        # print(x.size()) #torch.Size([nway * kshot, 1024, 13])
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        # total_loss = F.nll_loss(pred, target, weight=weight)
        total_loss = F.nll_loss(pred, target)
        # total_loss = F.cross_entropy(pred, target)
        
        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))