visit goole drive by the link:
https://drive.google.com/drive/folders/1QWwJus5zXt1-xv321l-VGVGF_g5J23xG?usp=sharing
and download all the files in the link, put all the files follow the tree below:
Diff-AMP50
├── abp_feature_comparison_results/
│   └── models/
│       ├── hemolysis_classifier.joblib    # Hemolytic toxicity classifier model weights (included in the repository)
│       └── abp_classifier_aac_dde.joblib  # Antibacterial screening random forest model (downloaded from google drive)
├── checkpoints/
│   ├── decoder_best.pth                   # Decoder weight file (included in the repository)
    ├── esm2 
        └──model.safetensors
│   └── diffusion_amp_diffusion_50len_150ep/
│       └── best.pth                       # U-Net diffusion model weights (downloaded from google drive)
