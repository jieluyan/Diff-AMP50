# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from config import *


class DiffusionTokenDataset(Dataset):
    """扩散模型专用的Token数据集"""
    
    def __init__(self, tokens, labels=None, use_one_hot=False):
        """
        tokens: numpy array, shape (N, seq_len)
        labels: 可选，分类标签
        use_one_hot: 是否转换为one-hot编码
        """
        if isinstance(tokens, pd.DataFrame):
            tokens = tokens.values
        
        # 保持为整数
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None
        
        self.use_one_hot = use_one_hot
        self.vocab_size = VOCAB_SIZE  # 22
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        token = self.tokens[idx]
        
        if self.use_one_hot:
            # 转换为one-hot编码 [seq_len, vocab_size]
            token_one_hot = torch.nn.functional.one_hot(
                token, num_classes=self.vocab_size
            ).float()
            token = token_one_hot
        
        if self.labels is not None:
            return token, self.labels[idx]
        else:
            return token


def getToken(seq, max_len=MAX_SEQ_LEN, aa_codes=AA_CODES):
    """将序列转为token（与步骤3一致）"""
    seq_len = len(seq)
    target_len = max_len + 1
    token = [PAD_TOKEN] * target_len
    
    for i in range(min(seq_len, max_len)):
        if seq[i] in aa_codes:
            token[i] = aa_codes.index(seq[i]) + 1
    
    # 添加END_TOKEN
    used_len = min(seq_len, max_len)
    token[used_len] = END_TOKEN
    
    return np.array(token, dtype=np.int32)


def token2seq(token, aa_codes=AA_CODES):
    """token转序列（简化正确版）"""
    if torch.is_tensor(token):
        token = token.cpu().numpy()
    
    token = token.flatten().astype(np.int32)
    seq_chars = []
    
    for t in token:
        t_int = int(t)
        
        if t_int == END_TOKEN:
            break
        
        if t_int == PAD_TOKEN:
            continue
        
        if 1 <= t_int <= len(aa_codes):
            seq_chars.append(aa_codes[t_int - 1])
    
    return ''.join(seq_chars)


def create_diffusion_dataloaders(
    pos_csv, neg_csv=None, batch_size=256, val_split=0.1, 
    num_workers=0, use_one_hot=False, shuffle=True
):
    """
    创建扩散模型数据加载器
    
    Args:
        pos_csv: 正样本token文件（必须）
        neg_csv: 负样本token文件（可选，用于分类条件）
        use_one_hot: 是否使用one-hot编码
    """
    # 读取正样本
    pos_df = pd.read_csv(pos_csv)
    pos_tokens = extract_token_columns(pos_df)
    
    all_tokens = pos_tokens
    all_labels = None
    
    # 如果有负样本，合并并添加标签
    if neg_csv is not None:
        neg_df = pd.read_csv(neg_csv)
        neg_tokens = extract_token_columns(neg_df)
        
        all_tokens = np.vstack([pos_tokens, neg_tokens])
        all_labels = np.array([1] * len(pos_tokens) + [0] * len(neg_tokens))
    
    # 划分数据集
    n_total = len(all_tokens)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    indices = np.random.permutation(n_total)
    
    train_tokens = all_tokens[indices[:n_train]]
    val_tokens = all_tokens[indices[n_train:]]
    
    if all_labels is not None:
        train_labels = all_labels[indices[:n_train]]
        val_labels = all_labels[indices[n_train:]]
    else:
        train_labels = val_labels = None
    
    # 创建数据集
    train_dataset = DiffusionTokenDataset(
        train_tokens, train_labels, use_one_hot=use_one_hot
    )
    val_dataset = DiffusionTokenDataset(
        val_tokens, val_labels, use_one_hot=use_one_hot
    )
    
    # 创建加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  使用one-hot: {use_one_hot}")
    print(f"  使用标签: {all_labels is not None}")
    
    return train_loader, val_loader


def extract_token_columns(df):
    """从DataFrame中提取token列"""
    token_cols = [col for col in df.columns if col.startswith('token_')]
    if not token_cols:
        # 如果没有token_前缀，假设所有数值列都是token
        token_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    return df[token_cols].values


# 测试函数
def test_token_functions():
    """测试token编解码"""
    test_seq = "ACDEFG"
    print(f"测试序列: {test_seq}")
    
    # 编码
    token = getToken(test_seq)
    print(f"编码token: {token[:10]}...")
    
    # 解码
    decoded = token2seq(token)
    print(f"解码序列: {decoded}")
    print(f"匹配: {'✓' if test_seq == decoded else '✗'}")
    
    # 测试数据集
    tokens = np.array([token, token])
    dataset = DiffusionTokenDataset(tokens, use_one_hot=False)
    sample = dataset[0]
    print(f"数据集样本形状: {sample.shape}")
    
    return test_seq == decoded


def tokens2seqs(tokens, aa_codes=AA_CODES):
    """批量转换tokens为序列"""
    seqs = []
    for token in tokens:
        seq = token2seq(token, aa_codes)
        seqs.append(seq)
    return seqs

# 【修复】：适配 Kaggle 赛制长度限制 (min=8, max=50)
def save_generated_seqs(seqs, output_path, prefix="gene_AMP"):
    """保存生成的序列到 CSV 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 【修复】：原先为 5 <= len <= 40，现改为动态读取 MAX_SEQ_LEN，最低限制为 8
    valid_seqs = [seq for seq in seqs if 8 <= len(seq) <= MAX_SEQ_LEN]
    
    # 创建 DataFrame
    ids = [f"{prefix}_{i+1}" for i in range(len(valid_seqs))]
    df = pd.DataFrame({
        "ID": ids,
        "SEQUENCE": valid_seqs,
        "Length": [len(seq) for seq in valid_seqs]
    })
    
    # 保存
    df.to_csv(output_path, index=False)
    
    print(f"生成序列已保存:")
    print(f"  文件路径: {output_path}")
    print(f"  有效序列数: {len(valid_seqs)} / {len(seqs)}")
    if len(valid_seqs) > 0:
        print(f"  长度范围: {df['Length'].min()} - {df['Length'].max()}")

# 添加别名
create_dataloaders = create_diffusion_dataloaders

if __name__ == "__main__":
    success = test_token_functions()
    if success:
        print("\n 所有测试通过！")
    else:
        print("\n 测试失败，请检查代码")