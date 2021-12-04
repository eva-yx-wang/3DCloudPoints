import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Fun: Calculate distance(source points, target points) in each batch.

    src^T * dst = xn * xm + yn * ym + zn * zm           # 2ab
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;         # a^2
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;         # b^2
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2      # (a-b)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, 3]
        dst: target points, [B, M, 3]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Fun: get xyz of special points in each batch
    Input:
        points: input points data, [B, N, 3]
        idx: id(special points) in each batch, [B, X1, ..., Xn]
    Return:
        new_points: indexed points data, [B, X1, ..., Xn, 3]
    """
    device = points.device
    B = points.shape[0]
    # view_shape: [B, 1]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # repeat_shape: [1, npoint]
    repeat_shape = list(idx.shape)  
    repeat_shape[0] = 1
    # batch_indices: [B, npoint]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) 
    # take [B, npoint, 3] from [B, N, 3]
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    
    Input:
        xyz: in_model_point position data, [B, 3, N]
        npoint: num of farthest points in each batch
    Return:
        centroids: id(farthest points of each batch), [B, npoint]
    """
    device = xyz.device  # gpu
    B, N, C = xyz.shape
    # simutaniously in each batch !!!
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # initialize array saved id(farthest point), [B, npoint]
    distance = torch.ones(B, N).to(device) * 1e10 # distance(each npoint, farthest point) fulled with 10, [B, N]
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # id(farthest points of each point in each batch), abbr:id(farthest), [B]
    batch_indices = torch.arange(B, dtype=torch.long).to(device) # id(batch), [B]
    # update farthest points
    for i in range(npoint):
        centroids[:, i] = farthest # id(farthest)
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # xyz of farthest point, [B,1,3]
        dist = torch.sum((xyz - centroid) ** 2, -1)  # distance_xyz(each batch_point, farthest point), [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]  # update distance when 'dist < distance'
        farthest = torch.max(distance, -1)[1]  # update id of farthest point
    return centroids


    
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Fun: get idx of query points which have far-points within query ball
    Input:
        radius: max distance for farthest points of each batch
        nsample: number of max query points in each batch
        xyz: in_model_point position data, [B, N, 3]
        new_xyz: xyz of farthest points in each batch, [B, npoint, 3]
    Return:
        group_idx: grouped points index, [B, N, S]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # dist(far-point, each point in each batch), calculate agan!!!
    group_idx[sqrdists > radius ** 2] = N  # if far-point is out of query ball, then group_idx==N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # nsample points which have first shortest distance in each batch, [B, npoint, nsample]
    # 若前nsample个group的group_id==N, 则替换为group_id[0]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # if first nsample points have distance > radius, then replace with group_id[0]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: num of farthest points in each batch
        radius: ball query radius
        nsample: max query points in B*N points
        xyz: in_model_point position data, [B, 3, N]
        points: in_model_points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: xyz of farthest point for each point in each batch, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape 
    S = npoint 
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint], id(farthest_points) in each batch 
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3], xyz_of_farthest_point for each sampled point in each batch
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample], id of query points in each batch
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # [B, npoint, nsample, 3]

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint,  nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: in_model_point position data, [B, 3, N]
        points: in_model_points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# interpolate to nearest points
def interpolate_points(xyz1, xyz2_list, points1, points2_list):
    """
    Input:
        xyz1: in_1_points position data, [B, 3, npoint1]
        xyz2_list: in_2_points_list position data, [[B, 3, npoint21], [B, 3, npoint22], ...]
        points1: input points data, [B, in_1_points_features, npoint1]
        points2_list: [ [B, in_2_points_features, npoint2], [B, in_2_points_features, npoint2], ...]
        points2: input points data, [B, in_2_points_features, npoint2]
    Return:
        new_points: upsampled points data, [B, D', N]
    """
    xyz1 = xyz1.permute(0, 2, 1)  # [B, npoint1, 3]
    B, N, C = xyz1.shape
    interpolated_points_list = []

    # interpolate output_points of each FA (FA4-1)
    for xyz2,points2 in zip(reversed(xyz2_list), reversed(points2_list)):
        xyz2 = xyz2.permute(0, 2, 1)  # [B, npoint2, 3]
        points2 = points2.permute(0, 2, 1) # [B, npoint2, in_2_points_features]
        _, S, _ = xyz2.shape

        if S == 1: # not interpolate
            interpolated_points = points2.repeat(1, N, 1)
        else: # interpolate to nearest points
            # xyz_distance()
            dists = square_distance(xyz1, xyz2)  # dists:[B, npoint1, npoint2]
            dists, idx = dists.sort(dim=-1)  # idx:[B, npoint1, npoint2]
            dists, idx = dists[:, :, :3], idx[:, :, :3] # 到npoint1 nearest 3 xyz2 points 

            dist_recip = 1.0 / (dists + 1e-8)  # [B, npoint1, 3]
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # [B, npoint1, 1]
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  #[B, npoint1, in_2_points_features]

        interpolated_points_list.append(interpolated_points)

    return interpolated_points_list

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        '''
        npoint: num of farthest points in each batch
        radius: 0.1
        nsample: max sampled number in local region
        in_channel: 9+3
        mlp: [layer1_d_out, layer2_d_out, layer3_d_out]
        '''
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint 
        self.radius = radius
        self.nsample = nsample  
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  
        for out_channel in mlp:  # construct operation for each layer of mlp
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))  # 1st: Conv2d
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))                # 2ed: BatchNorm2d
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: in_model_points position data, [B, 3, N],
            points: in_model_points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # xyz:[B, N, C]
        xyz = xyz.permute(0, 2, 1)  # permute [B, C, N] to [B, N, C]
        # points:[B, N, D]
        if points is not None:
            points = points.permute(0, 2, 1)  # permute [B, D, N] to [B, N, D]

        # get sampled points which have have required far-points in each batch
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else: # use this !
            # new_xyz:[B, npoint, 3]  new_points:[B, npoint, nsample, 3+D]
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            
        new_points = new_points.permute(0, 3, 2, 1) # [B, 3+D, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i] # batchNorm
            # BN + conv2d + BN + reLu
            new_points =  F.relu(bn(conv(new_points)))  # [B, out_features, nsample, npoint]

        # max pooling of nsample points
        new_points = torch.max(new_points, 2)[0]  # [B, out_features, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, npoint]
        return new_xyz, new_points  # [B, 3, npoint], [B, out_features, npoint]



class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: in_model_point position data, [B, 3, N]
            points: in_model_point data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N ,3]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N ,D]

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        '''
        in_channel: mlp_in_D
        mlp: [layer1_d_out, ...]
        '''
        super(PointNetFeaturePropagation, self).__init__()
        # self.dense_conn = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    # interpolate and concate all prev_decoder_layer features
    def forward(self, xyz1, xyz2_list, points1, points2_list):
        """
        Input:
            xyz1: in_1_points position data, [B, 3, npoint1]
            xyz2_list: in_2_points_list position data, [[B, 3, npoint21], [B, 3, npoint22], ...]
            points1: input points data, [B, in_1_points_features, npoint1]
            points2_list: [ [B, in_2_points_features, npoint2], [B, in_2_points_features, npoint2], ...]
            points2: input points data, [B, in_2_points_features, npoint2]
        Return:
            new_points: upsampled points data, [B, D', N]
        """           
        # interpolate fp_points in different len to points1
        interpolate_points_list = interpolate_points(xyz1, xyz2_list, points1, points2_list)
        # add features
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, npoint1, in_1_points_features]
            # concate points2 features
            new_points = points1
            for points2 in interpolate_points_list:
                new_points = torch.cat([new_points, points2], dim=-1)  # [B, npoint1, in_1_points_features + in_21_points_features + in_22_points_features, ...]
        else:
            # 
            new_points = interpolate_points_list[0]
            for points2 in interpolate_points_list[1:]:
                new_points = torch.cat([new_points, points2], dim=-1)  # [B, npoint1, in_1_points_features + in_21_points_features + in_22_points_features, ...]

        new_points = new_points.permute(0, 2, 1)  # [B, npoint1, in_1_points_features + in_21_points_features + in_22_points_features, ...]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))  # [B, out_features, mlp_out_npoint]
        return new_points  # [B, out_features, mlp_out_npoint]


