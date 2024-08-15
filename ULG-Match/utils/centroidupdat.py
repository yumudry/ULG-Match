import torch
import torch.nn.functional as F
from torch import nn

class CentroidUpdater:
    def __init__(self, num_classes, feature_size, num_parts, device='cpu'):
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.num_parts = num_parts
        self.device = device
        self.reset()

    def reset(self):
        self.feature_sum_g = torch.zeros(self.num_classes, self.feature_size, device=self.device)
        self.feature_count_g = torch.zeros(self.num_classes, device=self.device)
        self.feature_sum_p = torch.zeros(self.num_parts*2, self.num_classes, self.feature_size, device=self.device)
        self.feature_count_p = torch.zeros(self.num_parts*2, self.num_classes, device=self.device)
        self.centroids = torch.zeros(self.num_classes, self.feature_size, device=self.device)

    def update(self, f_g_x, f_g_u_w, targets_x, targets_g_u, mask_g, f_p_x, f_p_u_w, targets_p_u, mask_p):
        self.update_features(f_g_x, f_g_u_w, targets_x, targets_g_u, mask_g, self.feature_sum_g, self.feature_count_g)
        for part in range(self.num_parts*2):
            f_p_part = torch.cat([f_p_x[:, :, part], f_p_u_w[mask_p[:, part] == 1, :, part]], dim=0)
            targets_p_part = torch.cat([targets_x, targets_p_u[mask_p[:, part] == 1, part]], dim=0)
            self.update_features(f_p_part, None, targets_p_part, None, None, self.feature_sum_p[part], self.feature_count_p[part])

    def update_features(self, f_x, f_u_w, targets_x, targets_g_u, mask_g, feature_sum, feature_count):
        for i in range(self.num_classes):
            mask = targets_x == i
            if mask.any():
                feature_sum[i] += f_x[mask].sum(dim=0)
                feature_count[i] += mask.sum()
        if f_u_w is not None and targets_g_u is not None and mask_g is not None:
            mask = mask_g == 1
            targets_u_w = targets_g_u[mask]
            f_u_w_filtered = f_u_w[mask]
            for i in range(self.num_classes):
                mask = targets_u_w == i
                if mask.any():
                    feature_sum[i] += f_u_w_filtered[mask].sum(dim=0)
                    feature_count[i] += mask.sum()

    def compute_centroids(self):
        valid_g = self.feature_count_g > 0
        if valid_g.any():
            self.centroids[valid_g] = self.feature_sum_g[valid_g] / self.feature_count_g[valid_g].unsqueeze(1)
            self.centroids[valid_g] = F.normalize(self.centroids[valid_g], p=2, dim=1)
        
        for part in range(self.num_parts * 2):
            valid_p = self.feature_count_p[part] > 0
            if valid_p.any():
                self.centroids[valid_p] += (self.feature_sum_p[part][valid_p] / self.feature_count_p[part][valid_p].unsqueeze(1))
        
        self.centroids = F.normalize(self.centroids, p=2, dim=1)


# class CentroidUpdater:
#     def __init__(self, num_classes, feature_size, num_parts, device='cpu'):
#         self.num_classes = num_classes
#         self.feature_size = feature_size
#         self.num_parts = num_parts
#         # 初始化类别中心
#         self.centroids_g = torch.zeros(num_classes, feature_size,device=device)
#         self.centroids_p = torch.zeros(num_parts*2, num_classes, feature_size,device=device)
#         self.centroids = torch.zeros(num_classes, feature_size,device=device)
#         # 初始化累积特征和计数器
#         self.feature_sum_g = torch.zeros(num_classes, feature_size,device=device)
#         self.feature_count_g = torch.zeros(num_classes,device=device)
#         self.feature_sum_p = torch.zeros(num_parts*2, num_classes, feature_size,device=device)
#         self.feature_count_p = torch.zeros(num_parts*2, num_classes,device=device)

#     def reset(self):
#         # 重置累积特征和计数器
#         self.feature_sum_g.fill_(0)
#         self.feature_count_g.fill_(0)
#         self.feature_sum_p.fill_(0)
#         self.feature_count_p.fill_(0)

#     def update(self, f_g_x, f_g_u_w, targets_x, targets_g_u, mask_g, f_p_x, f_p_u_w, targets_p_u, mask_p):
#         # 累积全局特征
#         # 有标签数据
#         for i in range(self.num_classes):
#             idxs_g_x = (targets_x == i).nonzero(as_tuple=True)[0]
#             if idxs_g_x.numel() > 0:
#                 self.feature_sum_g[i] += f_g_x[idxs_g_x].sum(dim=0)
#                 self.feature_count_g[i] += idxs_g_x.size(0)

#         # 无标签数据，根据mask
#         mask_g_u_w = mask_g == 1#返回一个布尔值的张量mask_g_u_w
#         targets_g_u_w = targets_g_u[mask_g_u_w]
#         f_g_u_w_filtered = f_g_u_w[mask_g_u_w]
#         for i in range(self.num_classes):
#             idxs_g_u_w = (targets_g_u_w == i).nonzero(as_tuple=True)[0]
#             if idxs_g_u_w.numel() > 0:
#                 self.feature_sum_g[i] += f_g_u_w_filtered[idxs_g_u_w].sum(dim=0)
#                 self.feature_count_g[i] += idxs_g_u_w.size(0)

#         # 累积局部特征
#         for part in range(self.num_parts*2):
#             # 有标签数据
#             f_p_x_part = f_p_x[:, :, part]
#             for i in range(self.num_classes):
#                 idxs_p_x = (targets_x == i).nonzero(as_tuple=True)[0]
#                 if idxs_p_x.numel() > 0:
#                     self.feature_sum_p[part, i] += f_p_x_part[idxs_p_x].sum(dim=0)
#                     self.feature_count_p[part, i] += idxs_p_x.size(0)

#             # 无标签数据，根据mask
#             mask_p_u_w = mask_p[:, part] == 1
#             targets_p_u_w = targets_p_u[mask_p_u_w, part]
#             f_p_u_w_filtered = f_p_u_w[mask_p_u_w, :, part]
#             for i in range(self.num_classes):
#                 idxs_p_u_w = (targets_p_u_w == i).nonzero(as_tuple=True)[0]
#                 if idxs_p_u_w.numel() > 0:
#                     self.feature_sum_p[part, i] += f_p_u_w_filtered[idxs_p_u_w].sum(dim=0)
#                     self.feature_count_p[part, i] += idxs_p_u_w.size(0)

#     def compute_centroids(self):
#         # 使用累积的特征和和计数器更新类别中心
#         valid_g = self.feature_count_g > 0
#         self.centroids_g[valid_g] = self.feature_sum_g[valid_g] / self.feature_count_g[valid_g].unsqueeze(1)
#         # print(self.centroids_g.shape)
#          # 对全局特征的类别中心进行L2归一化
#         self.centroids_g = F.normalize(self.centroids_g, p=2, dim=1)
#         for part in range(self.num_parts * 2):
#             valid_p = self.feature_count_p[part] > 0
#             if valid_p.any():  # 如果该部分有有效的特征
#                 # 对每个部分的累积特征和进行归一化以得到均值
#                 self.centroids_p[part][valid_p] = self.feature_sum_p[part][valid_p] / self.feature_count_p[part][valid_p].unsqueeze(1)
#         self.centroids_p = F.normalize(self.centroids_p, p=2, dim=2)
#         # print(self.centroids_p.shape)
#         mean_centroids_p = self.centroids_p.mean(dim=0)  # 在所有局部特征部分上取平均
#         self.centroids = (self.centroids_g + mean_centroids_p) / 2  # 计算全局特征和局部特征的平均中心
#         self.centroids = F.normalize(self.centroids, p=2, dim=1)
#         # print(self.centroids.shape)