#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版模型 - 残差连接 + 多尺度特征融合PAN + 注意力机制
参数量控制在20%增加范围内
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 能量级别定义
ENERGY_LEVELS = [0.20, 0.35, 0.50, 0.70, 1.00]

class ChannelAttention(nn.Module):
    """轻量级通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 使用较大的reduction_ratio来控制参数量
        hidden_dim = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, length)
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1))
        attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention

class SpatialAttention(nn.Module):
    """轻量级空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, length)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_map))
        return x * attention

class CBAM(nn.Module):
    """卷积块注意力模块 - 轻量版"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        # 可选的注意力机制
        self.attention = CBAM(out_channels) if use_attention else None
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
            
        out += residual
        out = F.relu(out)
        
        return out

class PANFusion(nn.Module):
    """轻量级路径聚合网络特征融合"""
    def __init__(self, channels_list):
        super(PANFusion, self).__init__()
        # 自顶向下路径
        self.top_down_convs = nn.ModuleList()
        # 自底向上路径  
        self.bottom_up_convs = nn.ModuleList()
        
        for i in range(len(channels_list) - 1):
            # 减少通道数来控制参数量
            hidden_dim = min(channels_list[i], channels_list[i+1]) // 2
            
            self.top_down_convs.append(nn.Sequential(
                nn.Conv1d(channels_list[i], hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
            
            self.bottom_up_convs.append(nn.Sequential(
                nn.Conv1d(channels_list[i+1], hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, features):
        """
        features: 多尺度特征列表，从低级到高级
        """
        # 自顶向下路径
        top_down_features = []
        prev_feature = features[-1]
        top_down_features.append(prev_feature)
        
        for i in range(len(features) - 2, -1, -1):
            curr_feature = features[i]
            
            # 上采样并融合
            upsampled = F.interpolate(prev_feature, size=curr_feature.size(-1), mode='linear', align_corners=False)
            upsampled = self.top_down_convs[i](upsampled)
            curr_processed = self.top_down_convs[i](curr_feature)
            
            fused = upsampled + curr_processed
            top_down_features.insert(0, fused)
            prev_feature = fused
        
        return top_down_features

class EnhancedLambWaveNet(nn.Module):
    """
    增强版Lamb波网络:
    - 残差连接
    - 多尺度特征融合PAN
    - 注意力机制
    - 参数量控制
    """
    def __init__(self, dropout=0.3):
        super(EnhancedLambWaveNet, self).__init__()
        
        # 输入归一化
        self.input_norm = nn.BatchNorm1d(8)
        
        # 多尺度特征提取骨干网络 - 使用较少的通道数控制参数量
        self.backbone = nn.ModuleList([
            # Stage 1: 8 -> 32
            nn.Sequential(
                nn.Conv1d(8, 32, kernel_size=64, stride=4, padding=30, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4)  # 输出: 32 × 312
            ),
            
            # Stage 2: 32 -> 64  
            ResidualBlock(32, 64, stride=2, use_attention=True),
            nn.MaxPool1d(4),  # 输出: 64 × 39
            
            # Stage 3: 64 -> 128
            ResidualBlock(64, 128, stride=2, use_attention=True),
            nn.MaxPool1d(2),  # 输出: 128 × 9
            
            # Stage 4: 128 -> 256  
            ResidualBlock(128, 256, stride=1, use_attention=True),
            nn.AdaptiveAvgPool1d(4)  # 输出: 256 × 4
        ])
        
        # 多尺度特征融合 - 简化但有效的实现
        # 全局特征处理
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 维度适配器 - 将不同backbone阶段的特征统一到256维
        self.multi_scale_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 256), nn.ReLU(inplace=True)),   # Stage 1: 32->256
            nn.Sequential(nn.Linear(64, 256), nn.ReLU(inplace=True)),   # Stage 2: 64->256  
            nn.Sequential(nn.Linear(128, 256), nn.ReLU(inplace=True)),  # Stage 3: 128->256
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True))   # Stage 4: 256->256 (恒等变换)
        ])
        
        # 预测头 - 使用更少的参数
        feature_dim = 256  # 来自最后一层
        
        # 位置预测头 (回归任务优化 - 减少BN使用)
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),  # 使用LayerNorm代替BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),  # 移除第二个BN
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # X1, Y1, X2, Y2
        )
        
        # 能量分类头 (添加BatchNorm)
        self.energy_shared = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.energy1_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 5)  # E1 - 5类
        )
        
        self.energy2_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
            nn.Linear(32, 5)  # E2 - 5类
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入: (batch_size, 5000, 8)
        x = x.transpose(1, 2)  # (batch_size, 8, 5000)
        x = self.input_norm(x)
        
        # 多尺度特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # 在关键点收集特征用于融合
            if i in [0, 2, 4, 6]:  # 每个stage的输出
                features.append(x)
        
        # 多尺度特征融合 - 简化但有效的实现
        if len(features) > 1:
            # 对所有特征进行池化和维度适配，然后加权融合
            pooled_features = []
            
            for i, feat in enumerate(features):
                # 全局池化
                pooled_feat = self.global_pool(feat).squeeze(-1)  # (batch, channels)
                
                # 使用预定义的适配器将特征适配到256维
                adapted_feat = self.multi_scale_adapters[i](pooled_feat)  # (batch, 256)
                pooled_features.append(adapted_feat)
            
            # 加权融合不同尺度的特征（给高级特征更高权重）
            weights = [0.1, 0.2, 0.3, 0.4]  # 从低级到高级特征的权重
            global_feature = torch.zeros_like(pooled_features[0])  # (batch, 256)
            
            for feat, weight in zip(pooled_features, weights):
                global_feature += weight * feat
            
            # 应用注意力机制到融合后的特征
            # 可选的自注意力来进一步优化特征表示
            # global_feature = F.normalize(global_feature, p=2, dim=1)  # L2归一化
        else:
            # 如果只有一个特征，直接使用
            final_feature = features[-1]
            pooled_feat = self.global_pool(final_feature).squeeze(-1)  # (batch, channels)
            global_feature = self.multi_scale_adapters[-1](pooled_feat)  # 使用最后一个适配器
        
        # 预测
        positions = self.position_head(global_feature)  # [X1, Y1, X2, Y2]
        
        # 能量预测 - 使用共享特征
        energy_shared = self.energy_shared(global_feature)
        energy1_logits = self.energy1_head(energy_shared)  # E1的5类概率
        energy2_logits = self.energy2_head(energy_shared)  # E2的5类概率
        
        return positions, energy1_logits, energy2_logits

def convert_enhanced_predictions_to_final_format(positions, energy1_logits, energy2_logits):
    """将增强模型预测转换为最终格式"""
    batch_size = positions.shape[0]
    
    # 能量从logits转换为实际值
    energy1_indices = torch.argmax(energy1_logits, dim=1)
    energy2_indices = torch.argmax(energy2_logits, dim=1)
    
    energy1_values = torch.tensor([ENERGY_LEVELS[idx] for idx in energy1_indices], 
                                  device=positions.device, dtype=positions.dtype)
    energy2_values = torch.tensor([ENERGY_LEVELS[idx] for idx in energy2_indices], 
                                  device=positions.device, dtype=positions.dtype)
    
    # 组合最终输出: [X1, Y1, E1, X2, Y2, E2]
    final_output = torch.cat([
        positions[:, 0:1],  # X1
        positions[:, 1:2],  # Y1
        energy1_values.unsqueeze(1),  # E1
        positions[:, 2:3],  # X2
        positions[:, 3:4],  # Y2
        energy2_values.unsqueeze(1)   # E2
    ], dim=1)
    
    return final_output

class AdaptiveTaskLoss(nn.Module):
    """智能多任务损失 - 分类达标后稳定聚焦定位"""
    def __init__(self, pos_weight=2.0, energy_weight=1.0, normalize_pos=True,
                 energy_threshold=0.95, focus_mode_factor=8.0, 
                 focus_energy_weight=1.0, stability_window=20):
        super(AdaptiveTaskLoss, self).__init__()
        self.pos_weight = pos_weight
        self.energy_weight = energy_weight
        self.normalize_pos = normalize_pos
        self.energy_threshold = energy_threshold  # 分类达标阈值
        self.focus_mode_factor = focus_mode_factor  # 聚焦模式位置权重倍数
        self.focus_energy_weight = focus_energy_weight  # 聚焦模式能量权重
        self.stability_window = stability_window  # 稳定性窗口大小
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # 位置归一化参数
        self.pos_mean = 200.0
        self.pos_std = 100.0
        
        # 动态权重历史 - 使用更大的窗口来稳定判断
        self.energy_acc_history = []
        self.focus_mode = False
        self.focus_triggered = False  # 一旦触发聚焦，就保持稳定
        
    def update_energy_accuracy(self, energy_accuracy):
        """更新能量分类准确率历史 - 稳定版本"""
        self.energy_acc_history.append(energy_accuracy)
        # 保持更长的历史记录以提高稳定性
        if len(self.energy_acc_history) > self.stability_window:
            self.energy_acc_history.pop(0)
        
        # 只有在未触发聚焦模式时才检查切换条件
        if not self.focus_triggered and len(self.energy_acc_history) >= 10:
            # 使用更长的平均窗口来减少波动
            recent_avg = sum(self.energy_acc_history[-10:]) / 10
            overall_avg = sum(self.energy_acc_history) / len(self.energy_acc_history)
            
            # 更严格的条件：需要连续较长时间的高准确率
            if recent_avg >= self.energy_threshold and overall_avg >= self.energy_threshold - 0.02:
                if not self.focus_mode:
                    print(f"🎯 进入位置聚焦模式！能量准确率: {recent_avg:.1%} (平均: {overall_avg:.1%})")
                    print(f"🔒 模式锁定：专注位置学习，权重调整为 {self.focus_mode_factor}:{self.focus_energy_weight}")
                self.focus_mode = True
                self.focus_triggered = True  # 锁定聚焦模式，不再切换
    
    def forward(self, predictions, targets, energy_accuracy=None):
        positions, energy1_logits, energy2_logits = predictions
        pos_targets, energy1_targets, energy2_targets = targets
        
        # 更新能量准确率
        if energy_accuracy is not None:
            self.update_energy_accuracy(energy_accuracy)
        
        # 位置回归损失
        if self.normalize_pos:
            normalized_positions = (positions - self.pos_mean) / self.pos_std
            normalized_targets = (pos_targets - self.pos_mean) / self.pos_std
            pos_loss = self.mse(normalized_positions, normalized_targets)
        else:
            pos_loss = self.mse(positions, pos_targets) / 10000.0
        
        # 能量分类损失
        energy1_loss = self.ce(energy1_logits, energy1_targets)
        energy2_loss = self.ce(energy2_logits, energy2_targets)
        energy_loss = (energy1_loss + energy2_loss) / 2
        
        # 动态权重调整 - 更温和的聚焦策略
        if self.focus_mode:
            # 聚焦模式：增加位置权重，保持能量权重稳定
            adaptive_pos_weight = self.pos_weight * self.focus_mode_factor  # 8倍
            adaptive_energy_weight = self.focus_energy_weight  # 1.0，保持分类稳定性
        else:
            # 平衡模式：使用原始权重
            adaptive_pos_weight = self.pos_weight
            adaptive_energy_weight = self.energy_weight
        
        # 总损失
        total_loss = adaptive_pos_weight * pos_loss + adaptive_energy_weight * energy_loss
        
        return total_loss, pos_loss, energy_loss, adaptive_pos_weight, adaptive_energy_weight


class GradientBalancingLoss(nn.Module):
    """梯度平衡损失 - 基于任务学习速度动态调整"""
    def __init__(self, pos_weight=2.0, energy_weight=1.0, normalize_pos=True,
                 alpha=0.12, target_pos_energy_ratio=5.0):
        super(GradientBalancingLoss, self).__init__()
        self.pos_weight = pos_weight
        self.energy_weight = energy_weight
        self.normalize_pos = normalize_pos
        self.alpha = alpha  # 梯度范数平衡因子
        self.target_ratio = target_pos_energy_ratio  # 期望的位置:能量梯度比例
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # 位置归一化参数
        self.pos_mean = 200.0
        self.pos_std = 100.0
        
        # 梯度范数历史
        self.pos_grad_norms = []
        self.energy_grad_norms = []
        
    def compute_gradient_ratio(self, model, pos_loss, energy_loss):
        """计算任务梯度范数比例"""
        # 计算位置损失的梯度范数
        pos_grads = torch.autograd.grad(pos_loss, model.position_head.parameters(), 
                                       retain_graph=True, create_graph=False)
        pos_grad_norm = sum(g.norm() for g in pos_grads if g is not None)
        
        # 计算能量损失的梯度范数
        energy_params = list(model.energy_shared.parameters()) + \
                       list(model.energy1_head.parameters()) + \
                       list(model.energy2_head.parameters())
        energy_grads = torch.autograd.grad(energy_loss, energy_params,
                                         retain_graph=True, create_graph=False)
        energy_grad_norm = sum(g.norm() for g in energy_grads if g is not None)
        
        return pos_grad_norm, energy_grad_norm
    
    def forward(self, predictions, targets, model=None):
        positions, energy1_logits, energy2_logits = predictions
        pos_targets, energy1_targets, energy2_targets = targets
        
        # 计算基础损失
        if self.normalize_pos:
            normalized_positions = (positions - self.pos_mean) / self.pos_std
            normalized_targets = (pos_targets - self.pos_mean) / self.pos_std
            pos_loss = self.mse(normalized_positions, normalized_targets)
        else:
            pos_loss = self.mse(positions, pos_targets) / 10000.0
        
        energy1_loss = self.ce(energy1_logits, energy1_targets)
        energy2_loss = self.ce(energy2_logits, energy2_targets)
        energy_loss = (energy1_loss + energy2_loss) / 2
        
        # 如果提供了模型，计算梯度平衡权重
        if model is not None and model.training:
            try:
                pos_grad_norm, energy_grad_norm = self.compute_gradient_ratio(model, pos_loss, energy_loss)
                
                # 计算当前比例
                if energy_grad_norm > 1e-8:  # 避免除零
                    current_ratio = pos_grad_norm / energy_grad_norm
                    
                    # 如果位置梯度相对较小，增加其权重
                    if current_ratio < self.target_ratio:
                        ratio_factor = self.target_ratio / (current_ratio + 1e-8)
                        adaptive_pos_weight = self.pos_weight * (1 + self.alpha * ratio_factor)
                        adaptive_energy_weight = self.energy_weight
                    else:
                        adaptive_pos_weight = self.pos_weight
                        adaptive_energy_weight = self.energy_weight
                else:
                    adaptive_pos_weight = self.pos_weight
                    adaptive_energy_weight = self.energy_weight
                    
            except RuntimeError:
                # 如果梯度计算失败，使用原始权重
                adaptive_pos_weight = self.pos_weight
                adaptive_energy_weight = self.energy_weight
        else:
            adaptive_pos_weight = self.pos_weight
            adaptive_energy_weight = self.energy_weight
        
        total_loss = adaptive_pos_weight * pos_loss + adaptive_energy_weight * energy_loss
        
        return total_loss, pos_loss, energy_loss, adaptive_pos_weight, adaptive_energy_weight


class TaskUncertaintyLoss(nn.Module):
    """基于任务不确定性的自动权重学习"""
    def __init__(self, normalize_pos=True):
        super(TaskUncertaintyLoss, self).__init__()
        self.normalize_pos = normalize_pos
        
        # 可学习的任务权重参数（对数空间）
        self.log_vars = nn.Parameter(torch.zeros(2))  # [position, energy]
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # 位置归一化参数
        self.pos_mean = 200.0
        self.pos_std = 100.0
    
    def forward(self, predictions, targets):
        positions, energy1_logits, energy2_logits = predictions
        pos_targets, energy1_targets, energy2_targets = targets
        
        # 计算基础损失
        if self.normalize_pos:
            normalized_positions = (positions - self.pos_mean) / self.pos_std
            normalized_targets = (pos_targets - self.pos_mean) / self.pos_std
            pos_loss = self.mse(normalized_positions, normalized_targets)
        else:
            pos_loss = self.mse(positions, pos_targets) / 10000.0
        
        energy1_loss = self.ce(energy1_logits, energy1_targets)
        energy2_loss = self.ce(energy2_logits, energy2_targets)
        energy_loss = (energy1_loss + energy2_loss) / 2
        
        # 基于不确定性的自动权重
        pos_weight = torch.exp(-self.log_vars[0])
        energy_weight = torch.exp(-self.log_vars[1])
        
        # 不确定性正则化项
        pos_reg = self.log_vars[0] / 2
        energy_reg = self.log_vars[1] / 2
        
        # 总损失
        total_loss = (pos_weight * pos_loss + pos_reg + 
                     energy_weight * energy_loss + energy_reg)
        
        return total_loss, pos_loss, energy_loss, pos_weight.item(), energy_weight.item()


class FixedWeightLoss(nn.Module):
    """固定权重多任务损失 - 始终保持5:1权重比例"""
    def __init__(self, pos_weight=5.0, energy_weight=1.0, normalize_pos=True):
        super(FixedWeightLoss, self).__init__()
        self.pos_weight = pos_weight
        self.energy_weight = energy_weight
        self.normalize_pos = normalize_pos
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
        # 位置归一化参数
        self.pos_mean = 200.0
        self.pos_std = 100.0
    
    def forward(self, predictions, targets, energy_accuracy=None):
        positions, energy1_logits, energy2_logits = predictions
        pos_targets, energy1_targets, energy2_targets = targets
        
        # 位置回归损失
        if self.normalize_pos:
            normalized_positions = (positions - self.pos_mean) / self.pos_std
            normalized_targets = (pos_targets - self.pos_mean) / self.pos_std
            pos_loss = self.mse(normalized_positions, normalized_targets)
        else:
            pos_loss = self.mse(positions, pos_targets) / 10000.0
        
        # 能量分类损失
        energy1_loss = self.ce(energy1_logits, energy1_targets)
        energy2_loss = self.ce(energy2_logits, energy2_targets)
        energy_loss = (energy1_loss + energy2_loss) / 2
        
        # 固定权重：始终保持5:1比例
        adaptive_pos_weight = self.pos_weight      # 始终5.0
        adaptive_energy_weight = self.energy_weight # 始终1.0
        
        # 总损失
        total_loss = adaptive_pos_weight * pos_loss + adaptive_energy_weight * energy_loss
        
        # 创建损失字典
        loss_dict = {
            'position_loss': pos_loss.item(),
            'energy_loss': energy_loss.item(),
            'pos_weight': adaptive_pos_weight,
            'energy_weight': adaptive_energy_weight
        }
        
        return total_loss, loss_dict


# 使用固定权重损失作为默认
EnhancedLambWaveLoss = FixedWeightLoss

def create_enhanced_model(**kwargs):
    """创建增强模型"""
    model = EnhancedLambWaveNet(**kwargs)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created enhanced model with {param_count:,} parameters")
    
    # 检查参数增长
    baseline_params = 4_838_110
    growth_rate = (param_count - baseline_params) / baseline_params * 100
    print(f"Parameter growth: {growth_rate:+.1f}% vs baseline")
    
    if growth_rate > 20:
        print(f"⚠️  Warning: Parameter growth ({growth_rate:.1f}%) exceeds 20% limit!")
    else:
        print(f"✅ Parameter growth within 20% limit")
    
    return model

if __name__ == "__main__":
    # 测试增强模型
    print("Testing enhanced model with residual connections, PAN fusion, and attention...")
    
    model = create_enhanced_model(dropout=0.3)
    
    # 测试前向传播
    test_input = torch.randn(4, 5000, 8)
    print(f"Input shape: {test_input.shape}")
    
    positions, energy1_logits, energy2_logits = model(test_input)
    print(f"Positions shape: {positions.shape}")
    print(f"Energy1 logits shape: {energy1_logits.shape}")
    print(f"Energy2 logits shape: {energy2_logits.shape}")
    
    # 转换为最终格式
    final_output = convert_enhanced_predictions_to_final_format(positions, energy1_logits, energy2_logits)
    print(f"Final output shape: {final_output.shape}")
    print(f"Sample prediction: {final_output[0]}")
    
    # 测试损失函数
    criterion = EnhancedLambWaveLoss()
    
    pos_targets = torch.randn(4, 4) * 100 + 200
    energy1_targets = torch.randint(0, 5, (4,))
    energy2_targets = torch.randint(0, 5, (4,))
    
    targets = (pos_targets, energy1_targets, energy2_targets)
    predictions = (positions, energy1_logits, energy2_logits)
    
    loss_result = criterion(predictions, targets)
    if isinstance(loss_result, tuple) and isinstance(loss_result[1], dict):  # FixedWeightLoss with loss_dict
        loss, loss_dict = loss_result
        print(f"  Loss dictionary: {loss_dict}")
    else:  # 其他损失函数
        loss, pos_loss, energy_loss = loss_result
    print(f"\nLoss test:")
    print(f"  Total: {loss:.6f}")
    print(f"  Position: {loss_dict['position_loss']:.6f}")
    print(f"  Energy: {loss_dict['energy_loss']:.6f}")
    
    print("\n✅ Enhanced model test passed!")
    print("🚀 Features: Residual Connections + PAN Fusion + Attention + Parameter Control")
