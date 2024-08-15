import torch
import torch.nn.functional as F
from torch import nn


class CentroidsLoss_p(nn.Module):
    def __init__(self, temperature=0.1):
        """
        初始化
        temperature: 温度系数，控制相似度的尺度
        """
        super(CentroidsLoss_p, self).__init__()
        self.temperature = temperature

    def forward(self, f_p, targets):
        """
        计算损失
        f_p: 局部特征 [batch_size, feature_size, num_parts]
        targets: 真实类别标签 [batch_size]
        """
        batch_size, feature_size, num_parts = f_p.shape
        assert feature_size == self.feature_size and num_parts == self.num_parts, "Feature size or num_parts mismatch"

        # 计算正样本距离
        pos_distances = torch.sqrt(((f_p - self.centroids_g[targets].unsqueeze(2)) ** 2).sum(dim=1))

        # 计算所有负样本距离，并找到最小的那个
        all_distances = torch.sqrt(((f_p.unsqueeze(1) - self.centroids_g.unsqueeze(0).unsqueeze(3)) ** 2).sum(dim=2))
        all_distances.scatter_(1, targets.unsqueeze(1).unsqueeze(2), float('inf'))  # 排除正样本
        min_neg_distances, _ = all_distances.min(dim=1)

        # 计算损失
        loss = F.relu(pos_distances - min_neg_distances + self.margin).mean()

        return loss
#局部特征和全局特征与总的类别中心计算loss
class SoftContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SoftContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_g, f_p, centroids, targets_g, targets_p, mask_g, mask_p):

        """
        f_g: 全局特征，形状 [batch_size, feature_size]
        f_p: 局部特征，形状 [batch_size, feature_size, num_parts]
        centroids: 类别中心，形状 [num_classes, feature_size]
        targets_g: 全局特征标签，形状 [batch_size]
        targets_p: 局部特征标签，形状 [batch_size, num_parts]
        """
        # 对全局特征和局部特征进行L2归一化
        f_g = F.normalize(f_g, p=2, dim=1)
        f_p = F.normalize(f_p, p=2, dim=1)

        # 筛选全局特征和对应的伪标签
        valid_g = mask_g.bool()
        f_g_filtered = f_g[valid_g]
        targets_g_filtered = targets_g[valid_g]

        # 计算全局特征的损失
        loss_g = self.compute_loss(f_g_filtered, centroids, targets_g_filtered) if len(f_g_filtered) > 0 else 0

        # 筛选局部特征和对应的伪标签
        num_parts = f_p.size(2)
        loss_p_list = []
        for part in range(num_parts):
            valid_p = mask_p[:, part].bool()
            part_feature = f_p[:, :, part][valid_p]
            part_targets = targets_p[:, part][valid_p]

            # 计算每个部分的局部特征的损失
            if len(part_feature) > 0:
                part_loss = self.compute_loss(part_feature, centroids, part_targets)
            else:
                # 同样，使用f_p的设备和dtype
                part_loss = torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype)
            loss_p_list.append(part_loss)

        # 计算局部损失的平均值，如果列表为空，返回零值张量
        if loss_p_list:
            loss_p = torch.stack(loss_p_list).mean()
        else:
            loss_p = torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype)

        # 组合全局和局部损失
        total_loss = loss_g + loss_p
        return total_loss

    def compute_loss(self, features, centroids, targets):
        sim_matrix = torch.matmul(features, centroids.T) / self.temperature
        targets_one_hot = F.one_hot(targets, num_classes=centroids.size(0)).to(features.dtype)
        pos_sim = torch.sum(sim_matrix * targets_one_hot, dim=-1)
        all_sim = torch.logsumexp(sim_matrix, dim=-1)
        loss = -pos_sim + all_sim
        return loss.mean()

#局部与局部类别中心 全局与全局类别中心
class SpecificContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SpecificContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_g, f_p, centroids_g, centroids_p, targets_g, targets_p, mask_g, mask_p):
        """
        f_g: 全局特征，形状 [batch_size, feature_size]
        f_p: 局部特征，形状 [batch_size, feature_size, num_parts]
        centroids_g: 全局特征类别中心，形状 [num_classes, feature_size]
        centroids_p: 局部特征类别中心，形状 [num_classes, feature_size, num_parts]
        targets_g: 全局特征标签，形状 [batch_size]
        targets_p: 局部特征标签，形状 [batch_size, num_parts]
        """
        # 对全局特征和局部特征进行L2归一化
        f_g = F.normalize(f_g, p=2, dim=1)
        f_p = F.normalize(f_p, p=2, dim=1)

        # 筛选全局特征和对应的伪标签
        valid_g = mask_g.bool()
        f_g_filtered = f_g[valid_g]
        targets_g_filtered = targets_g[valid_g]

        # 计算全局特征与全局类别中心之间的损失
        loss_g = self.compute_loss(f_g_filtered, centroids_g, targets_g_filtered) if f_g_filtered.size(0) > 0 else 0

        # 筛选局部特征和对应的伪标签，并计算局部特征与局部类别中心之间的损失
        num_parts = f_p.size(2)
        loss_p_list = []
        for part in range(num_parts):
            valid_p = mask_p[:, part].bool()
            part_feature = f_p[:, :, part][valid_p]
            part_targets = targets_p[:, part][valid_p]
            # part_centroids = centroids_p[:, :, part]
            part_centroids = centroids_p[part]

            if part_feature.size(0) > 0:
                part_loss = self.compute_loss(part_feature, part_centroids, part_targets)
            else:
                part_loss = torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype)
            loss_p_list.append(part_loss)

        loss_p = torch.stack(loss_p_list).mean() if loss_p_list else torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype)

        # 组合全局和局部损失
        total_loss = loss_g + loss_p
        return total_loss

    def compute_loss(self, features, centroids, targets):
        sim_matrix = torch.matmul(features, centroids.T) / self.temperature #计算相似度矩阵
        targets_one_hot = F.one_hot(targets, num_classes=centroids.size(0)).to(features.dtype)#转换为one-hot
        pos_sim = torch.sum(sim_matrix * targets_one_hot, dim=-1)#通过将 sim_matrix 与 targets_one_hot 相乘，可以选取出每个特征向量与其对应真实类别中心之间的相似度
        all_sim = torch.logsumexp(sim_matrix, dim=-1)
        loss = -pos_sim + all_sim  #-lgA+lgB
        return loss.mean()

#弱增强的局部特征(超过阈值)靠近全局类别中心
class LocalToGlobalLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(LocalToGlobalLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_p, centroids_g, targets_p, mask_p):
        """
        Calculate contrastive loss between local features and global centroids.
        
        Parameters:
        f_p (Tensor): Local features with shape [batch_size, feature_size, num_parts].
        centroids_g (Tensor): Global centroids with shape [num_classes, feature_size].
        targets_p (Tensor): Local targets with shape [batch_size, num_parts].
        mask_p (Tensor): Mask for local features with shape [batch_size, num_parts].
        
        Returns:
        Tensor: The computed loss.
        """
        # Normalize local features
        f_p_norm = F.normalize(f_p, p=2, dim=1)

        # Compute loss for local features against global centroids for each part
        num_parts = f_p.size(2)
        loss_per_part = []
        for part in range(num_parts):
            f_p_part = f_p_norm[:, :, part]
            targets_part = targets_p[:, part]
            mask_part = mask_p[:, part].bool()
            
            # Apply the mask to select only the valid samples
            valid_f_p_part = f_p_part[mask_part]
            valid_targets_part = targets_part[mask_part]

            if valid_f_p_part.size(0) > 0:
                loss_part = self.compute_loss(valid_f_p_part, centroids_g, valid_targets_part)
                loss_per_part.append(loss_part)
            else:
                loss_per_part.append(torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype))

        # Mean loss across all parts
        loss = torch.mean(torch.stack(loss_per_part)) if loss_per_part else torch.tensor(0.0, device=f_p.device, dtype=f_p.dtype)
        return loss

    def compute_loss(self, features, centroids, targets):
        """
        Helper function to compute the contrastive loss.
        """
        sim_matrix = torch.matmul(features, centroids.T) / self.temperature
        targets_one_hot = F.one_hot(targets, num_classes=centroids.size(0)).float().to(features.device)
        pos_sim = (sim_matrix * targets_one_hot).sum(dim=-1)
        all_sim = torch.logsumexp(sim_matrix, dim=-1)
        loss = -pos_sim + all_sim
        return loss.mean()

#同上，但不经过阈值筛选
class LocalallToGlobalLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(LocalallToGlobalLoss, self).__init__()
        self.temperature = temperature

    def forward(self, f_p, centroids_g, targets_p):
        """
        Calculate contrastive loss between local features and global centroids without masking.
        
        Parameters:
        f_p (Tensor): Local features with shape [batch_size, feature_size, num_parts].
        centroids_g (Tensor): Global centroids with shape [num_classes, feature_size].
        targets_p (Tensor): Local targets with shape [batch_size, num_parts].
        
        Returns:
        Tensor: The computed loss.
        """
        # Normalize local features
        f_p_norm = F.normalize(f_p, p=2, dim=1)

        # Compute loss for local features against global centroids for each part
        num_parts = f_p.size(2)
        loss_per_part = []
        for part in range(num_parts):
            # Select the features and targets for the current part
            f_p_part = f_p_norm[:, :, part]
            targets_part = targets_p[:, part]

            # Calculate contrastive loss for the current part
            loss_part = self.compute_loss(f_p_part, centroids_g, targets_part)
            loss_per_part.append(loss_part)

        # Mean loss across all parts
        loss = torch.mean(torch.stack(loss_per_part))
        return loss

    def compute_loss(self, features, centroids, targets):
        """
        Helper function to compute the contrastive loss.
        """
        # Calculate similarity matrix between features and centroids
        sim_matrix = torch.matmul(features, centroids.T) / self.temperature
        # Create one-hot encoding for targets
        targets_one_hot = F.one_hot(targets, num_classes=centroids.size(0)).float().to(features.device)
        # Calculate positive similarities
        pos_sim = (sim_matrix * targets_one_hot).sum(dim=-1)
        # Calculate log-sum-exponential for normalization
        all_sim = torch.logsumexp(sim_matrix, dim=-1)
        # Compute the final contrastive loss
        loss = -pos_sim + all_sim
        return loss.mean()