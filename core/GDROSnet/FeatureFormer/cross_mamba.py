import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from mamba_ssm import Mamba


class PositionalEncoding2D(nn.Module):
    """
    Step 1: 2D Positional Encoding
    在将2D特征图展平为1D序列之前，必须注入二维位置编码，
    否则Mamba在扫描时会完全丢失像素在原图中的空间相邻关系。
    """
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # 创建可学习的2D位置编码
        self.row_embed = nn.Parameter(torch.randn(1, height, 1, channels // 2) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, width, channels // 2) * 0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x_with_pos: [B, C, H, W] 带有位置编码的特征
        """
        B, C, H, W = x.shape
        # 将位置编码插值到目标尺寸
        row_emb = F.interpolate(self.row_embed.permute(0, 3, 1, 2), 
                                size=(H, 1), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        col_emb = F.interpolate(self.col_embed.permute(0, 3, 1, 2), 
                                size=(1, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        # 扩展并拼接
        row_emb = row_emb.expand(B, H, W, C // 2)
        col_emb = col_emb.expand(B, H, W, C // 2)
        pos_enc = torch.cat([row_emb, col_emb], dim=-1)  # [B, H, W, C]
        pos_enc = pos_enc.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x + pos_enc


class FourWayScan(nn.Module):
    """
    Step 2: 四向交叉扫描策略 (VMamba策略)
    Mamba具有因果自回归特性，默认从左到右单向处理。
    但图像没有明确的先后顺序，因此使用四向扫描以最大程度保留2D局部性：
    1. 从左到右、从上到下
    2. 从右到左、从上到下
    3. 从左到右、从下到上
    4. 从右到左、从下到上
    """
    def __init__(self):
        super(FourWayScan, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            scans: list of 4 tensors, each [B, H*W, C]
            分别对应四种扫描顺序展平后的序列
        """
        B, C, H, W = x.shape
        
        # Scan 1: 从左到右，从上到下 (默认)
        scan1 = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Scan 2: 从右到左，从上到下 (水平翻转)
        scan2 = torch.flip(x, dims=[3]).view(B, C, H * W).permute(0, 2, 1)
        
        # Scan 3: 从左到右，从下到上 (垂直翻转)
        scan3 = torch.flip(x, dims=[2]).view(B, C, H * W).permute(0, 2, 1)
        
        # Scan 4: 从右到左，从下到上 (水平和垂直翻转)
        scan4 = torch.flip(x, dims=[2, 3]).view(B, C, H * W).permute(0, 2, 1)
        
        return [scan1, scan2, scan3, scan4]
    
    def reverse(self, scans, H, W, fusion_conv=None):
        """
        将四种扫描序列恢复为原始2D特征图并融合
        Args:
            scans: list of 4 tensors, each [B, H*W, C]
            H, W: 原始空间尺寸
            fusion_conv: 可学习的1x1卷积用于自适应融合
        Returns:
            x: [B, C, H, W]
        """
        B = scans[0].shape[0]
        C = scans[0].shape[2]
        
        # Reverse scan 1
        x1 = scans[0].permute(0, 2, 1).view(B, C, H, W)
        
        # Reverse scan 2 (水平翻转回来)
        x2 = torch.flip(scans[1].permute(0, 2, 1).view(B, C, H, W), dims=[3])
        
        # Reverse scan 3 (垂直翻转回来)
        x3 = torch.flip(scans[2].permute(0, 2, 1).view(B, C, H, W), dims=[2])
        
        # Reverse scan 4 (水平和垂直翻转回来)
        x4 = torch.flip(scans[3].permute(0, 2, 1).view(B, C, H, W), dims=[2, 3])
        
        # 使用可学习的1x1卷积进行自适应融合，替代硬平均
        # 将4个方向的结果在通道维度拼接 [B, 4*C, H, W]
        x_concat = torch.cat([x1, x2, x3, x4], dim=1)
        
        if fusion_conv is not None:
            # 通过1x1卷积学习各方向的权重，输出 [B, C, H, W]
            x = fusion_conv(x_concat)
        else:
            # 回退到硬平均
            x = (x1 + x2 + x3 + x4) / 4.0
        
        return x


class MambaBlock(nn.Module):
    """
    使用原始mamba_ssm库的Mamba块实现
    具有优化的CUDA核函数，支持高效的选择性扫描机制
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        
        # 原始Mamba实现，使用优化的CUDA核函数
        self.mamba = Mamba(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 输入投影：将输入投影到扩展维度
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        
        # 输出投影：将扩展维度投影回原始维度
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, C] - 批次、序列长度、通道
        Returns:
            output: [B, L, C]
        """
        # 输入投影到扩展维度
        x_inner = self.in_proj(x)  # [B, L, d_inner]
        
        # 通过原始Mamba块处理
        x_mamba = self.mamba(x_inner)  # [B, L, d_inner]
        
        # 输出投影回原始维度
        output = self.out_proj(x_mamba)  # [B, L, C]
        
        return output


class CrossGatedMambaBlock(nn.Module):
    """
    Step 3: 跨模态交叉门控Mamba块
    保持两个并行的Mamba分支，用模态A的特征去控制模态B的Mamba状态转移，反之亦然。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(CrossGatedMambaBlock, self).__init__()
        self.dim = dim
        
        # 光学分支的Mamba
        self.opt_mamba = MambaBlock(dim, d_state, d_conv, expand)
        # SAR分支的Mamba
        self.sar_mamba = MambaBlock(dim, d_state, d_conv, expand)
        
        # 交叉门控网络：用SAR特征调制光学特征
        self.opt_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 交叉门控网络：用光学特征调制SAR特征
        self.sar_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # LayerNorm
        self.norm_opt = nn.LayerNorm(dim)
        self.norm_sar = nn.LayerNorm(dim)
        
    def forward(self, opt_seq, sar_seq):
        """
        Args:
            opt_seq: [B, L, C] 光学特征序列
            sar_seq: [B, L, C] SAR特征序列
        Returns:
            opt_out: [B, L, C] 调制后的光学特征
            sar_out: [B, L, C] 调制后的SAR特征
        """
        # 保存残差连接
        opt_residual = opt_seq
        sar_residual = sar_seq
        
        # LayerNorm
        opt_norm = self.norm_opt(opt_seq)
        sar_norm = self.norm_sar(sar_seq)
        
        # 交叉门控：用SAR调制光学
        opt_gate_input = torch.cat([opt_norm, sar_norm], dim=-1)
        opt_gate = self.opt_gate(opt_gate_input)  # [B, L, C]
        opt_gated = opt_norm * opt_gate  # 逐元素相乘
        
        # 交叉门控：用光学调制SAR
        sar_gate_input = torch.cat([sar_norm, opt_norm], dim=-1)
        sar_gate = self.sar_gate(sar_gate_input)  # [B, L, C]
        sar_gated = sar_norm * sar_gate  # 逐元素相乘
        
        # 分别通过Mamba块
        opt_out = self.opt_mamba(opt_gated)
        sar_out = self.sar_mamba(sar_gated)
        
        # 残差连接
        opt_out = opt_out + opt_residual
        sar_out = sar_out + sar_residual
        
        return opt_out, sar_out


class CrossMambaFeatureTransformer(nn.Module):
    """
    完整的Cross-Mamba特征变换器，替换原有的FeatureTransformer
    
    处理流程：
    1. 2D位置编码注入
    2. 四向交叉扫描展平
    3. 跨模态交叉门控Mamba处理
    4. 特征还原为2D
    """
    def __init__(self, num_layers=6, d_model=256, d_state=16, d_conv=4, expand=2, pos_enc_size=(256, 256)):
        super(CrossMambaFeatureTransformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 2D位置编码 - 在__init__中初始化，确保优化器能追踪其参数
        # 使用F.interpolate可以自适应不同尺寸的输入
        self.pos_enc = PositionalEncoding2D(d_model, pos_enc_size[0], pos_enc_size[1])
        
        # 四向扫描
        self.four_way_scan = FourWayScan()
        
        # 多层Cross-Mamba块
        self.layers = nn.ModuleList([
            CrossGatedMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(d_model)
        
        # 可学习的1x1卷积用于四向扫描结果的自适应融合
        # 输入4*C通道（4个方向），输出C通道
        self.fusion_conv = nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False)
        
    def forward(self, feature0, feature1):
        """
        完整的Cross-Mamba前向传播流程
        
        Step 1: 2D位置编码注入
        - 在将2D特征图展平为1D序列之前，注入二维位置编码
        - 保留像素在原图中的空间相邻关系，避免Mamba扫描时丢失空间信息
        
        Step 2: 四向交叉扫描展平
        - 使用VMamba的四向扫描策略，最大程度保留2D局部性
        - 四种扫描方向：左上->右下、右上->左下、左下->右上、右下->左上
        
        Step 3: 跨模态交叉门控Mamba处理
        - 保持两个并行的Mamba分支（光学和SAR）
        - 使用交叉门控机制，让光学和SAR特征相互调制
        - 通过选择性状态空间模型（Selective SSM）实现高效的序列建模
        
        Step 4: 特征还原为2D特征图
        - 将处理后的1D序列按照扫描的逆过程还原为2D特征图
        - 融合四种扫描方向的结果，得到最终的2D特征表示
        
        Args:
            feature0: [B, C, H, W] 光学特征
            feature1: [B, C, H, W] SAR特征
        Returns:
            feature0: [B, C, H, W] 处理后的光学特征
            feature1: [B, C, H, W] 处理后的SAR特征
        """
        B, C, H, W = feature0.shape
        
        # Step 1: 注入2D位置编码（已在__init__中初始化，通过F.interpolate自适应尺寸）
        feature0 = self.pos_enc(feature0)
        feature1 = self.pos_enc(feature1)
        
        # Step 2: 四向交叉扫描展平为1D序列
        opt_scans = self.four_way_scan(feature0)  # list of 4 x [B, H*W, C]
        sar_scans = self.four_way_scan(feature1)  # list of 4 x [B, H*W, C]
        
        # 将4个方向的特征在Batch维度拼接，实现并行计算
        # opt_scans_stacked: [4*B, H*W, C]
        opt_scans_stacked = torch.cat(opt_scans, dim=0)
        sar_scans_stacked = torch.cat(sar_scans, dim=0)
        
        # Step 3: 跨模态交叉门控Mamba处理（并行处理4个方向）
        opt_processed = opt_scans_stacked
        sar_processed = sar_scans_stacked
        
        for layer in self.layers:
            opt_processed, sar_processed = layer(opt_processed, sar_processed)
        
        # FFN（并行处理）
        opt_processed = opt_processed + self.ffn(self.norm(opt_processed))
        sar_processed = sar_processed + self.ffn(self.norm(sar_processed))
        
        # 拆分回4个方向
        opt_processed_scans = torch.chunk(opt_processed, 4, dim=0)
        sar_processed_scans = torch.chunk(sar_processed, 4, dim=0)
        
        # Step 4: 特征还原为2D特征图（使用可学习的1x1卷积融合）
        feature0 = self.four_way_scan.reverse(opt_processed_scans, H, W, self.fusion_conv)
        feature1 = self.four_way_scan.reverse(sar_processed_scans, H, W, self.fusion_conv)
        
        return feature0, feature1