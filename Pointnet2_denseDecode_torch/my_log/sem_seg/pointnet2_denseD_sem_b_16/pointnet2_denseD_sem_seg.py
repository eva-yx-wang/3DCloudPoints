import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.pointnet2_denseD_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import torchvision.models as models

class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        ######## when run forward self.fun will search here !!
        # change features
        sa_out_D_0 = 8
        sa_npoint = [1024, 256, 64, 16]
        sa_mlp_in_D = [9 + 3, sa_out_D_0*2 + 3, sa_out_D_0*4 + 3, sa_out_D_0*8 + 3]
        sa_mlp_out_D = [[sa_out_D_0, sa_out_D_0, sa_out_D_0*2], 
                        [sa_out_D_0*2, sa_out_D_0*2, sa_out_D_0*4], 
                        [sa_out_D_0*4, sa_out_D_0*4, sa_out_D_0*8], 
                        [sa_out_D_0*8, sa_out_D_0*8, sa_out_D_0*16]]

        fp_out_D_last = 128
        fp_mlp_in_D = [sa_out_D_0*16 + sa_out_D_0*8, 
                        sa_out_D_0*16 + fp_out_D_last*2 + sa_out_D_0*4, 
                        sa_out_D_0*16 + fp_out_D_last*2 + fp_out_D_last*2 + sa_out_D_0*2, 
                        sa_out_D_0*16 + fp_out_D_last*2 + fp_out_D_last*2 + fp_out_D_last]
                        
        fp_mlp_out_D = [[fp_out_D_last*2, fp_out_D_last*2], 
                        [fp_out_D_last*2, fp_out_D_last*2], 
                        [fp_out_D_last*2, fp_out_D_last], 
                        [fp_out_D_last, fp_out_D_last, fp_out_D_last]]
        '''
        fp_mlp_in_D[1]:384 = fp_mlp_out_D[0,-1] + sa_mlp_out_D[, -1]
        '''
        self.sa1 = PointNetSetAbstraction(sa_npoint[0], 0.1, 32, sa_mlp_in_D[0], sa_mlp_out_D[0], False)
        self.sa2 = PointNetSetAbstraction(sa_npoint[1], 0.2, 32, sa_mlp_in_D[1], sa_mlp_out_D[1], False)
        self.sa3 = PointNetSetAbstraction(sa_npoint[2], 0.4, 32, sa_mlp_in_D[2], sa_mlp_out_D[2], False)
        self.sa4 = PointNetSetAbstraction(sa_npoint[3], 0.8, 32, sa_mlp_in_D[3], sa_mlp_out_D[3], False)

        self.fp4 = PointNetFeaturePropagation(fp_mlp_in_D[0], fp_mlp_out_D[0])
        self.fp3 = PointNetFeaturePropagation(fp_mlp_in_D[1], fp_mlp_out_D[1])
        self.fp2 = PointNetFeaturePropagation(fp_mlp_in_D[2], fp_mlp_out_D[2])
        self.fp1 = PointNetFeaturePropagation(fp_mlp_in_D[3], fp_mlp_out_D[3])
        
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
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_points:[B, 64, 10B4]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l2_points:[B, 128, 256]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l3_points:[B, 256, 64]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # l4_points:[B, 512, 16]

        ### dense decoder
        fp_xyz_list = []
        fp_points_list = []

        fp_xyz_list.append(l4_xyz)
        fp_points_list.append(l4_points)
        l3_points = self.fp4(l3_xyz, fp_xyz_list, l3_points, fp_points_list)  # l3_points: [B, 256, 64]
        # self.interpolated_points_list.append(l3_interp_points)

        fp_points_list.append(l3_points)
        fp_xyz_list.append(l3_xyz)
        l2_points = self.fp3(l2_xyz, fp_xyz_list, l2_points, fp_points_list)  # l2_points:[B, 256, 256]
        
        fp_points_list.append(l2_points)
        fp_xyz_list.append(l2_xyz)
        l1_points = self.fp2(l1_xyz, fp_xyz_list, l1_points, fp_points_list)  # l1_points:[B, 128, 1024]
        
        fp_points_list.append(l1_points)
        fp_xyz_list.append(l1_xyz)
        l0_points = self.fp1(l0_xyz, fp_xyz_list, None, fp_points_list)  # l0_points:[B, 128, 4096]
        
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