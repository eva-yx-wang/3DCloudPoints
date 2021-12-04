import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import torchvision.models as models

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        ######## when run forward self.fun will search here !!
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        '''
        B: id of batch, --batch_size
        N: num of in_model_points in each batch, '--npoint'
        npoint: num of far-points in each batch, different in each sa_Layer
        D: initial dimension of features
        out_features: num of feature dimensions, different in each sa_Layer

        '''
        l0_points = xyz  # [B, D, N]
        l0_xyz = xyz[:,:3,:]  # [B, 3, N]

        # lk_xyz:[B, 3, npoint]; lk_points:[B, out_features, npoint]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_points:[2, 64, 1024]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_points:[2, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_points:[2, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # l4_points:[2, 512, 16]

        # l3_points：[B, out_features, npoint]
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # l3_points: [2, 256, 64]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # l2_points:[2, 256, 256]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # l1_points:[2, 128, 1024]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # l0_points:[2, 128, 4096]

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)))) # [B, 128, N]
        x = self.conv2(x)  # [B, num_classes, N]
        x = F.log_softmax(x, dim=1)  #[B, num_classes, N]
        x = x.permute(0, 2, 1)  #[B, N, num_classes]
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))