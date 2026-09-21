# Diff-AMP50: AMP Challenge 2027
A Diffusion-Based Framework for De Novo Antimicrobial Peptide Generation and Multi-Stage Screening

# Abstract
We propose a two-stage latent diffusion framework coupled with a multidimensional biological cascade screening pipeline for the de novo design of antimicrobial peptides (AMPs). The framework employs a U-Net-based diffusion model to generate peptide candidates in a continuous latent space, followed by a self-trained sequence decoder to reconstruct amino acid sequences of 8–50 residues. To improve biological relevance and candidate quality, we further develop a three-stage screening pipeline that integrates activity and hemolytic-toxicity prediction, sequence-level novelty assessment against natural AMP references, and biological manifold evaluation using the ESM-2 protein language model. The resulting Top-100 candidates exhibit high predicted antimicrobial activity, low hemolytic toxicity, and substantial sequence and structural novelty, demonstrating the potential of the proposed framework for efficient and biologically informed AMP discovery.

## The pipeline
This experiment has been tested on a 5070ti and a 5060 series graphics card, and the results keep consistent.
1. Extract the positive samples that meet the requirements from the original dataset, and then generate negative samples that conform to biological properties through a specific algorithm (with modifications)
2. Extract features
3. Train the diffusion model
4. Train the decoder
5. Train the antibacterial activity classifier
6. Train the toxicity/hemolysis classifier
7. Download ESM2 to screen for biological rationality
8. Generate biological sequences that meet the requirements

## The arguments
python == 3.10
--n-sequences 50000 (set maximum number of generated sequences)
--top-k 100 (extract top k AMP candidates)
--seed 42 (seed)
--length 50 (set maximum length for all the generated sequences, >=50)

## The project structure
All the files and their locations when installed should be:
```text
amp-generator/
├── acp_feature_comparison_results/
│   └── models/
│       ├── hemolysis_classifier.joblib    # Hemolytic toxicity classifier model weights (download from google drive)
│       └── acp_classifier_aac_dde.joblib  # Antibacterial screening random forest model (download from google drive)
├── checkpoints/
│   ├── decoder_best.pth                   # Decoder weight file (download from google drive)
    ├── esm2 
        └──model.safetensors
│   └── diffusion_amp_diffusion_50len_150ep/
│       └── best.pth                       # U-Net diffusion model weights (download from google drive)
├── data/
│   └── antibacterial.fasta                # Sequence database provided by AMP chanllenge 2027 for checking similarity
├── src/
│   └── amp_generator/
│       ├── __init__.py                    # Initial file
│       └── generate.py                    # Core file: Peptide generation, loading of large model, scoring and filtering method
├── generate/
│   ├── library.fasta                      # Full 50,000-sequence library
│   └── top.fasta                          # top-100 ranked seauences
├── config.py                              # Configure the parameters related to the diffusion and decoding models
├── diffusion_pytorch.py                   # Definition of the diffusion model (U-Net) network architecture
├── encoder_decoder_pytorch.py             # Definition of VAE/Decoder-Encoder Network Architecture
├── geneTokens_pytorch.py                  # Sequence vocabulary list (Vocabulary) and Tokenizer processing method
├── .gitignore                             # Git ignores files (prevents virtual environment and extremely large weight files from being included)
├── pyproject.toml                         # The uv packages that need to be deployed before running the project.
└── README.md                              # Project description document
```
google drive link: https://drive.google.com/drive/folders/1QWwJus5zXt1-xv321l-VGVGF_g5J23xG?usp=sharing
