# -*- coding: utf-8 -*-
"""
编码器-解码器实现
用于学习肽序列的潜在表示
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# 自动对齐词表大小：0(PAD), 1-20(AA), 21(END) 共 22 个类别
VOCAB_SIZE = END_TOKEN + 1
EMBED_DIM = 16

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


class Encoder(nn.Module):
    """编码器：动态获取 seq_len (从 config.py 读取)"""
    def __init__(self, seq_len=TOKEN_LEN, latent_dim=ENDECODER_CONFIG['latent_dim'], 
                 hidden_dims=ENDECODER_CONFIG['hidden_dims'], embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_dim = seq_len * embed_dim 
        
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder_net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, seq_len)
        x = self.embedding(x.long()) # (B, seq_len, 16)
        x = x.view(x.size(0), -1)    
        return self.encoder_net(x)

class Decoder(nn.Module):
    """解码器：对应重建到 config.TOKEN_LEN 长度"""
    def __init__(self, latent_dim=ENDECODER_CONFIG['latent_dim'], seq_len=TOKEN_LEN, 
                 hidden_dims=list(reversed(ENDECODER_CONFIG['hidden_dims']))):
        super().__init__()
        self.seq_len = seq_len
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
            
        self.decoder_base = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, seq_len * VOCAB_SIZE) 
    
    def forward(self, z):
        h = self.decoder_base(z)
        logits = self.fc_out(h) 
        return logits.view(-1, self.seq_len, VOCAB_SIZE)

class AutoEncoder(nn.Module):
    def __init__(self, seq_len=None, latent_dim=None, 
                 encoder_hidden=None, decoder_hidden=None, embed_dim=EMBED_DIM):
        super().__init__()
        
        # 使用config默认值
        if seq_len is None:
            seq_len = TOKEN_LEN
        if latent_dim is None:
            latent_dim = ENDECODER_CONFIG['latent_dim']
        if encoder_hidden is None:
            encoder_hidden = ENDECODER_CONFIG['hidden_dims']
        if decoder_hidden is None:
            decoder_hidden = list(reversed(ENDECODER_CONFIG['hidden_dims']))
        
        self.encoder = Encoder(
            seq_len=seq_len, 
            latent_dim=latent_dim, 
            hidden_dims=encoder_hidden, 
            embed_dim=embed_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim, 
            seq_len=seq_len, 
            hidden_dims=decoder_hidden
        )
    
    def forward(self, x):
        # 确保输入是long类型
        if x.dtype != torch.long:
            x = x.long()
        
        z = self.encoder(x)  # 编码
        logits = self.decoder(z)  # 解码 [batch, seq_len, vocab_size]
        return logits, z
    
    def encode(self, x):
        if x.dtype != torch.long:
            x = x.long()
        return self.encoder(x)
    
    def decode(self, z, temperature=1.0):
        """从潜在向量解码，可控制采样温度"""
        logits = self.decoder(z)  # [batch, seq_len, vocab_size]
        
        if temperature == 0:
            # 贪婪解码
            tokens = torch.argmax(logits, dim=-1)
        else:
            # 采样解码
            probs = F.softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1)
            tokens = tokens.view(z.size(0), -1)
        
        return tokens
    
class VAE(nn.Module):
    """VAE：完全通过 config 驱动"""
    def __init__(self, seq_len=TOKEN_LEN, latent_dim=ENDECODER_CONFIG['latent_dim'], hidden_dims=ENDECODER_CONFIG['hidden_dims'], embed_dim=EMBED_DIM):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        
        # 编码器
        encoder_layers = []
        in_dim = seq_len * embed_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        self.encoder_net = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(in_dim, seq_len * VOCAB_SIZE)
        self.seq_len = seq_len
    
    def encode(self, x):
        x_emb = self.embedding(x.long()).view(x.size(0), -1)
        h = self.encoder_net(x_emb)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_net(z)
        logits = self.fc_out(h)
        return logits.view(-1, self.seq_len, VOCAB_SIZE)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE 损失函数
    beta: KL 散度的权重（β-VAE）
    """
    target = x.long()
    # 重建损失（MSE）
    recon_loss = F.cross_entropy(x_recon.transpose(1, 2), target, reduction='sum')
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def train_autoencoder(model, train_loader, num_epochs=100, lr=1e-3, device='cuda'):
    """训练自编码器"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            
            # 前向传播
            x_recon, _ = model(x)
            
            # 计算损失
            target = x.long()
            loss = F.cross_entropy(x_recon.transpose(1, 2), target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    return model


def train_vae(model, train_loader, num_epochs=100, lr=1e-3, beta=1.0, device='cuda'):
    """训练 VAE"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            
            # 前向传播
            x_recon, mu, logvar = model(x)
            
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
    return model


if __name__ == "__main__":
    set_seed(42)
    
    # 参数
    batch_size = 32
    seq_len = TOKEN_LEN
    latent_dim = ENDECODER_CONFIG['latent_dim']
    
    # 构造模拟数据
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(DEVICE)
    
    print("="*60)
    print("测试自编码器")
    print("="*60)
    
    # 测试AE
    ae_model = AutoEncoder().to(DEVICE)
    logits_ae, z_ae = ae_model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输入示例: {x[0][:10].cpu().numpy()}...")
    print(f"潜在向量形状: {z_ae.shape}")
    print(f"重建logits形状: {logits_ae.shape}")
    
    # 测试重建准确率
    pred = torch.argmax(logits_ae, dim=-1)
    accuracy = (pred == x).float().mean()
    print(f"随机数据重建准确率: {accuracy.item():.4f}")
    
    print(f"\n模型参数量:")
    total_params = sum(p.numel() for p in ae_model.parameters())
    print(f"  总计: {total_params / 1e3:.1f}K")
    
    # 测试VAE
    print("\n" + "="*60)
    print("测试VAE")
    print("="*60)
    
    vae_model = VAE().to(DEVICE)
    logits_vae, mu, logvar = vae_model(x)
    
    print(f"VAE输出形状: {logits_vae.shape}")
    print(f"mu形状: {mu.shape}, logvar形状: {logvar.shape}")
    
    # 计算损失
    loss, recon_loss, kl_loss = vae_loss(logits_vae, x, mu, logvar, beta=1.0)
    print(f"VAE损失: 总={loss.item():.2f}, 重建={recon_loss.item():.2f}, KL={kl_loss.item():.2f}")