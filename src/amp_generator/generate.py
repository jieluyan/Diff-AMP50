# -*- coding: utf-8 -*-
"""
Kaggle AMP Challenge 2027 - 生成入口 (generate.py)
支持断点续传，引入级联过滤策略：
1. 双重随机森林：(AAC+DDE) 抗菌活性初筛 - 溶血毒性过滤 (Optimal Selectivity)
2. 80% Levenshtein 严格查重
3. ESM-2 蛋白质流形纠偏 (FBD Proxy)
"""

import argparse
import os
# 强制使用 Hugging Face 国内加速镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

# 导入本地配置和模型
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import *
from diffusion_pytorch import UNet, GaussianDiffusion, EMA 
from geneTokens_pytorch import tokens2seqs
from encoder_decoder_pytorch import Decoder

# ==================== 特征提取与打分模块 ====================
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')

def extract_aac_features(seq):
    aac = np.zeros(20)
    if len(seq) == 0: return aac
    for aa in seq:
        if aa in AA_LIST:
            aac[AA_LIST.index(aa)] += 1
    return aac / len(seq)

DIPEPTIDES_LIST = [aa1 + aa2 for aa1 in AA_LIST for aa2 in AA_LIST]
def extract_dde_features(seq):
    dde = np.zeros(400)
    if len(seq) < 2: return dde
    for i in range(len(seq) - 1):
        dipep = seq[i:i+2]
        if dipep in DIPEPTIDES_LIST:
            dde[DIPEPTIDES_LIST.index(dipep)] += 1
    total = len(seq) - 1
    return dde / total if total > 0 else dde

def extract_features_batch(sequences):
    features = []
    for seq in sequences:
        aac = extract_aac_features(seq)
        dde = extract_dde_features(seq)
        feat = np.concatenate([aac, dde])
        features.append(feat)
    return np.array(features)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==================== 主程序 ====================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="AMP Generator with Cascading Selectivity and Manifold Filter")
    # 官方要求的基本参数
    parser.add_argument('--n-sequences', type=int, default=50000, help='生成序列总数')
    parser.add_argument('--top-k', type=int, default=100, help='输出排名靠前的序列数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--length', type=int, default=50, help='要求的序列长度')
    parser.add_argument('--antibacterial-fasta', type=str, default='data/antibacterial.fasta', help='官方参照验证库')
    
    # 隐藏的本地模型参数
    parser.add_argument('--checkpoint', type=str, default='checkpoints/diffusion_amp_diffusion_50len_150ep/best.pth', help='U-Net权重路径')
    parser.add_argument('--classifier', type=str, default='acp_feature_comparison_results/models/acp_classifier_aac_dde.joblib', help='抗菌活性打分器路径')
    parser.add_argument('--hemo-classifier', type=str, default='acp_feature_comparison_results/models/hemolysis_classifier.joblib', help='溶血毒性打分器路径')
    parser.add_argument('--batch_size', type=int, default=256, help='生成批次大小')
    args = parser.parse_args()

    set_seed(args.seed)
    
    print("=" * 80)
    print("AMP Challenge 2027".center(80))
    print("=" * 80)
    print(f" -> 目标生成数: {args.n_sequences}")
    print(f" -> 运行设备:   {DEVICE}")
    
    LATENT_DIM = ENDECODER_CONFIG['latent_dim'] 
    MODEL_TOKEN_LEN = 51  # 强制固定为你训练时的长度，防止官方传 --length 40 测试时崩溃                 
    
    entry_point = Path(sys.argv[0]).stem # 动态获取入口名，必定为 'generate'
    out_dir = Path(entry_point)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_fasta = out_dir / "library.fasta"
    top_fasta = out_dir / "top.fasta"

    # ==================== 0. 提前加载官方查重库 ====================
    print("\n[加载天然抗菌肽参照库 (用于大库绝对去重)]")
    known_seq_set = set()
    ref_fasta = Path(args.antibacterial_fasta)
    if ref_fasta.exists():
        with open(ref_fasta, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith(">"):
                    known_seq_set.add(line.strip().upper())
    print(f" -> 已加载 {len(known_seq_set)} 条官方参照序列。")

    # ==================== 1. 生成断点检查与扩散采样 ====================
    # 使用字典替代 set 以保持读取和生成的确定性顺序
    existing_seqs = {}
    if lib_fasta.exists():
        print(f"\n[检测到已存在的生成文件: {lib_fasta}]")
        with open(lib_fasta, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith(">"):
                    existing_seqs[line.strip()] = True
        print(f" -> 成功加载历史进度: 已有 {len(existing_seqs)} 条序列。")

    if len(existing_seqs) >= args.n_sequences:
        print(f"目标数量 {args.n_sequences} 已达成！跳过生成，进入打分查重阶段。")
        # 字典的 keys() 会严格按照 Fasta 文件中的原始顺序输出
        valid_seqs = list(existing_seqs.keys())[:args.n_sequences]
    else:
        print("\n[加载生成模型...]")
        model = UNet(
            in_channels=LATENT_DIM, out_channels=LATENT_DIM,
            model_channels=UNET_CONFIG['model_channels'],
            channel_mult=tuple(UNET_CONFIG['channel_mult']),
            num_res_blocks=UNET_CONFIG['num_res_blocks'],
            num_classes=None, dropout=0.0
        ).to(DEVICE)
        
        if not Path(args.checkpoint).exists():
            print(f"❌ 找不到扩散模型权重: {args.checkpoint}")
            return
            
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
            ema = EMA(model)
            ema.shadow = checkpoint['ema_shadow']
            ema.apply_shadow()
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        decoder = Decoder(latent_dim=LATENT_DIM, seq_len=MODEL_TOKEN_LEN, hidden_dims=list(reversed(ENDECODER_CONFIG['hidden_dims']))).to(DEVICE)
        decoder_path = CHECKPOINT_DIR / "decoder_best.pth"
        if decoder_path.exists():
            decoder_checkpoint = torch.load(decoder_path, map_location=DEVICE)
            decoder.load_state_dict(decoder_checkpoint.get('state_dict', decoder_checkpoint))
        decoder.eval()
        
        diffusion = GaussianDiffusion(timesteps=DIFFUSION_CONFIG['timesteps'], device=DEVICE)
        
        print(f"\n[开始潜在空间扩散生成与实时落盘]")
        target_count = args.n_sequences
        f_lib = open(lib_fasta, "a", encoding="utf-8")
        pbar = tqdm(initial=len(existing_seqs), total=target_count, desc="增量生成中")
        
        try:
            while len(existing_seqs) < target_count:
                curr_batch = min(args.batch_size, target_count - len(existing_seqs) + 50)
                shape = (curr_batch, LATENT_DIM, 1, 1)
                
                z_samples = diffusion.sample(model, shape, y=None, guidance_scale=1.0)
                if z_samples.dim() == 4:
                    z_samples = z_samples.squeeze(-1).squeeze(-1)
                    
                logits = decoder(z_samples)
                tokens = torch.argmax(logits, dim=-1)
                seqs = tokens2seqs(tokens.cpu().numpy())
                
                for s in seqs:
                    if 8 <= len(s) <= args.length and s not in existing_seqs and s not in known_seq_set:
                        existing_seqs[s] = True
                        # 修改为官方最简格式 >seqX，去掉下划线
                        f_lib.write(f">seq{len(existing_seqs)}\n{s}\n")
                        pbar.update(1)
                        if len(existing_seqs) >= target_count:
                            break
                f_lib.flush()
        except KeyboardInterrupt:
            print("\n收到中止信号！当前序列已安全落盘，下次可继续。")
            f_lib.close()
            return
            
        f_lib.close()
        pbar.close()
        valid_seqs = list(existing_seqs)[:args.n_sequences]

    # ==================== 2. 双重随机森林初筛 (高活性 + 低毒性) ====================
    print("\n[加载双重 Ranker 进行全量最优选择性初筛]")
    
    # 加载抗菌模型
    if not Path(args.classifier).exists():
        print(f" 找不到抗菌打分模型: {args.classifier}")
        return
    amp_model = joblib.load(args.classifier)
    if isinstance(amp_model, dict) and 'model' in amp_model: amp_model = amp_model['model']

    # 加载溶血/毒性模型
    use_hemo = False
    if Path(args.hemo_classifier).exists():
        hemo_model = joblib.load(args.hemo_classifier)
        if isinstance(hemo_model, dict) and 'model' in hemo_model: hemo_model = hemo_model['model']
        use_hemo = True
        print(" -> 成功加载溶血性/毒性预测模型！将启用最优选择性过滤。")
    else:
        print(f" 找不到溶血模型 ({args.hemo_classifier})，仅使用抗菌活性。")

    print(" -> 提取统计学特征中 (AAC + DDE)...")
    X_features = extract_features_batch(valid_seqs)
    
    print(" -> 计算抗菌潜力概率...")
    amp_probs = amp_model.predict_proba(X_features)[:, 1]
    
    if use_hemo:
        print(" -> 计算溶血/毒性概率...")
        hemo_probs = hemo_model.predict_proba(X_features)[:, 1]
        
        # 核心指标：Selectivity Score = 抗菌潜力 - 毒性概率
        final_scores = amp_probs - hemo_probs
        
        # 为了绝对安全，强制把预测出高毒性 (>0.4) 的序列得分拉低，直接淘汰
        final_scores[hemo_probs > 0.4] = -999 
    else:
        final_scores = amp_probs
        hemo_probs = np.zeros(len(valid_seqs))

    # 【截取初筛池】：先取前 2000 条高选择性序列进入下一阶段查重
    PRE_FILTER_K = min(2000, len(valid_seqs))
    sorted_indices = np.argsort(final_scores)[::-1][:PRE_FILTER_K]
    
    candidate_seqs = [valid_seqs[i] for i in sorted_indices]
    candidate_scores = [final_scores[i] for i in sorted_indices]
    candidate_amps = [amp_probs[i] for i in sorted_indices]
    candidate_hemos = [hemo_probs[i] for i in sorted_indices]

    # ==================== 3. 加载参照库 & 严格查重阶段 ====================
    
    known_seqs = sorted(list(known_seq_set))

    print(f"\n[执行严格 80% Levenshtein 相似度过滤]")
    novel_seqs = []
    novel_scores = []
    novel_amps = []
    novel_hemos = []
    
    pbar = tqdm(total=PRE_FILTER_K, desc="查重初筛池")
    for seq, score, amp, hemo in zip(candidate_seqs, candidate_scores, candidate_amps, candidate_hemos):
        is_novel = True
        for ref_seq in known_seqs:
            if Levenshtein.ratio(seq, ref_seq) > 0.80:
                is_novel = False
                break 
        if is_novel:
            novel_seqs.append(seq)
            novel_scores.append(score)
            novel_amps.append(amp)
            novel_hemos.append(hemo)
        pbar.update(1)
    pbar.close()
    print(f" -> 查重后剩余 {len(novel_seqs)} 条合规且高选择性的候选序列。")

    # ==================== 4. ESM-2 生物学流形精筛 ====================
    print("\n[加载 ESM-2 进行生物学流形距离评估]")
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 防止国内网络报错
    # esm_name = "facebook/esm2_t6_8M_UR50D"
    # tokenizer = AutoTokenizer.from_pretrained(esm_name)
    esm_name = "checkpoints/esm2"  # 指向你的本地文件夹
    tokenizer = AutoTokenizer.from_pretrained(esm_name, local_files_only=True)
    esm_model = AutoModel.from_pretrained(esm_name, local_files_only=True, add_pooling_layer=False).to(DEVICE)
    esm_model.eval()

    def get_esm_embeddings(seq_list, batch_size=64):
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(seq_list), batch_size), desc="提取深层特征"):
                batch_seqs = seq_list[i:i+batch_size]
                inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, max_length=50,truncation=True).to(DEVICE)
                outputs = esm_model(**inputs)
                
                hidden_states = outputs.last_hidden_state

                mask = inputs['attention_mask'].unsqueeze(-1).float()
                hidden_states = hidden_states * mask

                lengths = mask.sum(dim=1)
                mean_embeddings = hidden_states.sum(dim=1) / lengths.clamp(min=1e-8)
                
                embeddings.append(mean_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

    print(" -> 建立天然抗菌肽的参考质心...")
    ref_embeds = get_esm_embeddings(known_seqs[:2000])
    ref_centroid = torch.mean(ref_embeds, dim=0)

    print(" -> 评估生成序列的生物学逼真度...")
    cand_embeds = get_esm_embeddings(novel_seqs)

    distances = torch.norm(cand_embeds - ref_centroid.unsqueeze(0), dim=1).numpy()
   # 【改版：流形惩罚软排序 (Soft Ranking)】
    print(" -> 执行流形惩罚软排序 (以选择性得分为主导)...")
    
    # 将 ESM-2 距离归一化，找出分布特征
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    
    final_candidates = []
    # 设置一个惩罚系数 alpha (可根据需要微调，0.1表示惩罚适中)
    alpha = 0.1 
    
    for seq, score, amp, hemo, dist in zip(novel_seqs, novel_scores, novel_amps, novel_hemos, distances):
        # 计算 Z-score 形式的惩罚项：距离均值越远，惩罚越大
        # 如果距离小于均值，不奖励也不惩罚；大于均值开始扣分
        penalty = max(0, (dist - dist_mean) / (dist_std + 1e-8)) * alpha
        
        # 综合惩罚后的最终得分 (Penalized Selectivity Score)
        penalized_score = score - penalty
        
        final_candidates.append({
            'seq': seq,
            'orig_score': score,
            'penalized_score': penalized_score,
            'amp': amp,
            'hemo': hemo,
            'dist': dist
        })
            
    # 加入 x['seq'] 作为第二排序键，彻底杜绝平局导致的乱序
    final_candidates.sort(key=lambda x: (x['penalized_score'], x['seq']), reverse=True)
    
    # 截取最终的 Top-K
    top_seqs = [x['seq'] for x in final_candidates[:args.top_k]]
    top_scores = [x['orig_score'] for x in final_candidates[:args.top_k]]
    top_amps = [x['amp'] for x in final_candidates[:args.top_k]]
    top_hemos = [x['hemo'] for x in final_candidates[:args.top_k]]
    top_dists = [x['dist'] for x in final_candidates[:args.top_k]]

    # ==================== 5. 结果落盘 ====================
    with open(top_fasta, "w", encoding="utf-8") as f:
        for i, seq in enumerate(top_seqs, start=1):
            f.write(f">seq{i}\n{seq}\n")
            
    print(f"\n 大功告成！文件已全部生成至: {out_dir}/")
    print(f"   - {lib_fasta.name} ({len(valid_seqs)} 条)")
    print(f"   - {top_fasta.name} ({len(top_seqs)} 条，已通过 毒性过滤、流形纠偏 与 严格查重)")
    
    print("\n Top 5 最具潜力的合规新颖无毒序列:")
    for i in range(min(5, len(top_seqs))):
        print(f"  {i+1}. {top_seqs[i]}")
        print(f"     -> 选择性得分: {top_scores[i]:.4f} | 抗菌: {top_amps[i]:.4f} | 溶血毒性: {top_hemos[i]:.4f}")

if __name__ == "__main__":
    main()