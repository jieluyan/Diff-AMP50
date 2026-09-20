# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from config import *


# ==================== 扩散过程工具类 ====================
class GaussianDiffusion:
    """高斯扩散过程实现"""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, 
                 schedule='linear', device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # 生成 beta 调度
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        
        # 计算扩散过程所需的系数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于前向扩散 q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # 用于反向去噪 p(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        
        # 后验方差 q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))
        
        # 后验均值系数
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦调度，生成更平滑的 beta 值"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _extract(self, a, t, x_shape):
        """提取指定时间步的系数并调整形状以便广播"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始样本 x_0"""
        sqrt_recip_alphas_cumprod = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
    
    def q_posterior(self, x_start, x_t, t):
        """计算后验分布 q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, model, x_t, t, y=None, clip_denoised=True):
        """预测 p(x_{t-1} | x_t)"""
        # 预测噪声
        pred_noise = model(x_t, t, y)
        
        # 从噪声预测 x_0
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        # 计算后验均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, y=None, clip_denoised=True):
        """从 p(x_{t-1} | x_t) 采样"""
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, y, clip_denoised=clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        # t == 0 时不添加噪声
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, y=None, guidance_scale=1.0):
        """
        完整的反向采样过程
        guidance_scale > 1.0 时使用分类器引导
        """
        device = self.device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='采样中', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            if guidance_scale != 1.0 and y is not None:
                # 分类器引导
                # 同时预测条件和无条件
                x_combined = torch.cat([x, x], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, torch.zeros_like(y)], dim=0)
                
                noise_pred = model(x_combined, t_combined, y_combined)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                
                # 引导公式：noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # 使用引导后的噪声预测进行采样
                x_recon = self.predict_start_from_noise(x, t, noise_pred)
                x_recon = torch.clamp(x_recon, -1.0, 1.0)
                model_mean, _, model_log_variance = self.q_posterior(x_recon, x, t)
                
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                x = model_mean + torch.exp(0.5 * model_log_variance) * noise
            else:
                x = self.p_sample(model, x, t, y)
        
        return x


# ==================== U-Net 模块 ====================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块，支持时间和类别嵌入"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes=None, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # 类别嵌入
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, out_channels)
        else:
            self.class_emb = None
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb, y=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # 添加类别嵌入
        if self.class_emb is not None and y is not None:
            class_emb = self.class_emb(y)
            h = h + class_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)
        
        # Attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        out = self.proj(out)
        
        return x + out


class UNet(nn.Module):
    """
    U-Net 架构用于扩散模型
    支持条件生成（时间步 + 类别）
    """
    
    def __init__(self, in_channels=1, out_channels=1, model_channels=64,
                 channel_mult=(1, 2, 4), num_res_blocks=2, 
                 attention_resolutions=(8, 16), dropout=0.1, num_classes=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_classes = num_classes
        
        # 时间步嵌入
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 下采样路径
        self.downs = nn.ModuleList([])
        channels = [model_channels]
        now_channels = model_channels
        
        for i, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(now_channels, out_channels, time_emb_dim, num_classes, dropout)
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            # 下采样
            if i != len(channel_mult) - 1:
                self.downs.append(nn.Conv2d(now_channels, now_channels, kernel_size=3, stride=2, padding=1))
                channels.append(now_channels)
        
        # 中间层
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim, num_classes, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim, num_classes, dropout)
        
        # 上采样路径
        self.ups = nn.ModuleList([])
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_channels = model_channels * mult
            
            # 每层的 ResBlock 数量：第一层多1个（匹配中间层），其他层 num_res_blocks+1 个
            num_blocks = num_res_blocks + 1
            
            for j in range(num_blocks):
                in_ch = now_channels + channels.pop()
                self.ups.append(
                    ResidualBlock(
                        in_ch,
                        out_channels,
                        time_emb_dim,
                        num_classes,
                        dropout
                    )
                )
                now_channels = out_channels
            
            # 上采样
            if i != len(channel_mult) - 1:
                self.ups.append(nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1))
        
        # 输出层
        self.out_norm = nn.GroupNorm(8, now_channels)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(now_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x, timesteps, y=None):
        """
        x: (B, C, H, W) 输入图像/token
        timesteps: (B,) 时间步
        y: (B,) 类别标签（可选）
        """
        # 保存输入尺寸用于输出裁剪
        input_size = x.shape[-2:]
        
        # 时间嵌入
        t_emb = self.time_mlp(timesteps)
        
        # 初始卷积
        h = self.conv_in(x)
        hs = [h]
        
        # 下采样
        for module in self.downs:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb, y)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间层
        h = self.mid_block1(h, t_emb, y)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, y)
        
        # 上采样
        for module in self.ups:
            if isinstance(module, ResidualBlock):
                if len(hs) > 0:  # 确保有 skip connection 可用
                    skip = hs.pop()
                    # 如果尺寸不匹配，使用插值调整 skip connection 的尺寸
                    if h.shape[-2:] != skip.shape[-2:]:
                        skip = torch.nn.functional.interpolate(
                            skip, size=h.shape[-2:], mode='nearest'
                        )
                    h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb, y)
            else:
                h = module(h)
        
        # 输出
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.conv_out(h)
        
        # 裁剪到输入尺寸（修复上采样导致的尺寸变化）
        if h.shape[-2:] != input_size:
            h = torch.nn.functional.interpolate(h, size=input_size, mode='nearest')
        
        return h


# ==================== EMA 模型 ====================
class EMA:
    """指数移动平均，用于稳定训练"""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # 测试代码
    set_seed()
    
    # 获取 config 中的潜在空间维度（Latent Diffusion 要求）
    latent_dim = ENDECODER_CONFIG.get('latent_dim', 64)
    
    # 创建模型
    model = UNet(
        in_channels=latent_dim,  # 【修复】：原本硬编码为 1，现改为 latent_dim (64)
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        num_classes=2
    ).to(DEVICE)
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        timesteps=1000,
        device=DEVICE
    )
    
    # 测试前向传播
    batch_size = 4
    # 【修复】：原本为 torch.randn(batch_size, 1, 40, 1)，现对齐隐空间形状
    x = torch.randn(batch_size, latent_dim, 1, 1).to(DEVICE)
    t = torch.randint(0, 1000, (batch_size,)).to(DEVICE)
    y = torch.randint(0, 2, (batch_size,)).to(DEVICE)
    
    # 前向扩散
    noise = torch.randn_like(x)
    x_noisy = diffusion.q_sample(x, t, noise)
    
    # 噪声预测
    noise_pred = model(x_noisy, t, y)
    
    print(f"输入形状: {x.shape}")
    print(f"噪声形状: {noise.shape}")
    print(f"预测噪声形状: {noise_pred.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(" 模型测试通过！")