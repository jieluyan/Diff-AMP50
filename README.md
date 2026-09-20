本实验已在一块5070ti和一块5060系显卡上试验过，结果均保持一致。
1.从原始数据集提取符合要求的正样本，再通过特定算法（有改动）生成符合生物学性质的负样本
2.再提取特征
3.训练扩散模型
4.训练解码器
5.训练抗菌活性分类器
6.训练毒性/溶血性分类器
7.下载ESM2筛选生物学合理性
8.生成符合要求的生物学序列

python要求 3.10
--n-sequences 50000 (生成总数阈值)
--top-k 100 (最终截取数量)
--seed 42 (全局随机数种子)
--length 50 (最大长度上限,>=50)
具体项目结构如下所示
```text
amp-generator/
├── acp_feature_comparison_results/
│   └── models/
│       ├── hemolysis_classifier.joblib    # 溶血毒性分类器模型权重 (包含在仓库中)
│       └── acp_classifier_aac_dde.joblib  # 抗菌性初筛随机森林模型 (运行时从 Releases 动态下载)
├── checkpoints/
│   ├── decoder_best.pth                   # 解码器权重文件 (包含在仓库中)
    ├── esm2 
        └──model.safetensors
│   └── diffusion_amp_diffusion_50len_150ep/
│       └── best.pth                       # U-Net 扩散模型权重 (运行时从 Releases 动态下载)
├── data/
│   └── antibacterial.fasta                # 官方提供的查重参考序列库
├── src/
│   └── amp_generator/
│       ├── __init__.py                    # Python 包声明文件
│       └── generate.py                    # 核心入口：多肽生成、大模型加载、打分及过滤逻辑
├── config.py                              # 扩散与解码模型相关的参数配置文件
├── diffusion_pytorch.py                   # 扩散模型 (U-Net) 网络架构定义
├── encoder_decoder_pytorch.py             # VAE/编解码器网络架构定义
├── geneTokens_pytorch.py                  # 序列词汇表 (Vocabulary) 及 Tokenizer 处理逻辑
├── .gitignore                             # Git 忽略文件 (拦截虚拟环境及超大权重文件)
├── pyproject.toml                         # uv 包管理器依赖配置及入口点定义
└── README.md                              # 项目说明文档
```