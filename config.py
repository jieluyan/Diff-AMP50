# -*- coding: utf-8 -*-
"""
项目配置文件 - AMP Challenge 2027 参赛专用版
架构: AutoEncoder (Token -> Latent 64) + U-Net (Latent Diffusion)
MAX_SEQ_LEN = 50, 支持大规模抗菌肽生成
"""
import os
from pathlib import Path
import torch

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.absolute()

DATA_DIR = PROJECT_ROOT / "AMP_data"
NEW_DATASET_DIR = DATA_DIR / "new_dataset"
TOKEN_DIR = DATA_DIR / "token"
FEATURE_DIR = DATA_DIR / "feature"

MODEL_DIR = PROJECT_ROOT / "model"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULT_DIR = PROJECT_ROOT / "result"
GENE_SEQS_DIR = RESULT_DIR / "gene_seqs"
PICS_DIR = PROJECT_ROOT / "pics"

# Kaggle 评测要求输出目录
KAGGLE_GENERATE_DIR = PROJECT_ROOT / "generate"


# ==================== 数据文件路径 ====================
POS_SAMPLES_CSV = NEW_DATASET_DIR / "positive_samples.csv"
NEG_SAMPLES_CSV = NEW_DATASET_DIR / "negative_samples.csv"
POS_TOKENS_CSV = NEW_DATASET_DIR / "positive_tokens.csv"
NEG_TOKENS_CSV = NEW_DATASET_DIR / "negative_tokens.csv"

# ==================== 核心序列与 Token 参数 ====================
AA_CODES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# 修复：AA_DICT 从 1 开始映射 (0 留给 PAD)
AA_DICT = {aa: idx + 1 for idx, aa in enumerate(AA_CODES)}
NUM_AA = len(AA_CODES)

MAX_SEQ_LEN = 50                    # 适配比赛要求的最大长度
TOKEN_LEN = MAX_SEQ_LEN + 1         # 51维 (预留给 END_TOKEN)
PAD_TOKEN = 0                       # 填充位
END_TOKEN = len(AA_CODES) + 1       # 结束标记 = 21
VOCAB_SIZE = END_TOKEN + 1          # 词表总大小 = 22 (0到21)

# ==================== 模型超参数 (Latent Diffusion) ====================
LATENT_DIM = 64                     # AutoEncoder 映射的潜在空间维度

DIFFUSION_CONFIG = {
    "timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "beta_schedule": "linear",
}

# 修正：U-Net 在 Latent 空间中运行
UNET_CONFIG = {
    "in_channels": LATENT_DIM,     # 64
    "out_channels": LATENT_DIM,    # 64
    "model_channels": 128,         # 基础通道数，扩大以提升表达能力
    "channel_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "dropout": 0.1,
    "use_class_cond": False,
    "use_time_embed": True,
    "use_position_embed": True,
}

# 修正：编码器-解码器适配 51 维 Token 序列
ENDECODER_CONFIG = {
    "input_dim": TOKEN_LEN,        # 51
    "latent_dim": LATENT_DIM,      # 64
    "hidden_dims": [256, 128],     # 扩大容量
}

# ==================== 训练参数 ====================
TRAIN_CONFIG = {
    "batch_size": 128,          # 针对 2.6万数据提升批次大小
    "num_epochs": 50,
    "learning_rate": 2e-4,
    "weight_decay": 1e-6,
    "ema_decay": 0.9999,
    "grad_clip": 1.0,
    "save_interval": 10,
    "sample_interval": 50,
    "num_workers": 0,
    "accumulate_grad_batches": 2,
}

# ==================== 评测生成参数 ====================
GENERATE_CONFIG = {
    "num_samples": 50000,       # Kaggle 要求生成数量
    "top_k": 100,               # Kaggle 要求排名截取
    "min_seq_len": 8,           # Kaggle 规范最低长度
    "max_seq_len": MAX_SEQ_LEN, # Kaggle 规范最高长度 (50)
    "temperature": 0.9,
    "sampling_steps": 250,      # 可切换为 ddim 加速
    "sampling_method": "ddpm",
}

# ==================== 设备智能分配 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True
else:
    GPU_NAME = "CPU"
