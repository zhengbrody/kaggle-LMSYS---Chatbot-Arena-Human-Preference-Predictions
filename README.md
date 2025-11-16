# LMSYS Chatbot Arena - Human Preference Predictions

Kaggle Competition Solution: Predicting human preferences between two chatbot responses using ensemble of Gemma-2 9B and Llama-3 8B with QLoRA fine-tuning.

## Overview

This project implements a 3-class classification model to predict which chatbot response humans prefer:
- **Class 0**: Model A wins
- **Class 1**: Model B wins
- **Class 2**: Tie

## Project Structure

```
├── gemma_qlora_ft_v38_train.py       # Main training script (Gemma-2 with QLoRA)
├── gemma2_infer.py                   # Gemma-2 inference script
├── llama3_infer.py                   # Llama-3 inference script
├── lmsys-inference-ensemble.ipynb    # Ensemble inference notebook
├── lmsys-llama-31-tpu-train.ipynb    # Llama-3 TPU training notebook
└── README.md                         # This file
```

## Quick Start

### 1. Installation

```bash
pip install -U "transformers>=4.42.3" bitsandbytes accelerate peft
pip install pandas numpy scikit-learn torch datasets
```

### 2. Data Preparation

Set up the following directory structure:

```
../model/
└── gemma-2-9b-it-bnb-4bit/    # Pre-quantized Gemma-2 model

../data/
└── data_78k.csv               # Training data (78k labeled examples)
```

**Training data format** (`data_78k.csv`):
```csv
id,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
0,["text1","text2"],["resp1","resp2"],["resp3","resp4"],1,0,0
```

### 3. Training

#### Gemma-2 Training (Main Script)

```bash
python gemma_qlora_ft_v38_train.py
```

**Configuration** (modify in `gemma_qlora_ft_v38_train.py`):

```python
@dataclass
class Config:
    checkpoint: str = "../model/gemma-2-9b-it-bnb-4bit"  # Model path
    max_length: int = 1536          # Max token length
    n_splits: int = 100             # K-fold splits
    fold_idx: int = 0               # Current fold (0-99)
    per_device_train_batch_size: int = 16
    n_epochs: int = 1
    lr: float = 8e-5
    lora_r: int = 32                # LoRA rank
    lora_alpha: float = 64          # LoRA scaling
    freeze_layers: int = 2          # Freeze first N layers
```

**Key Features**:
- 4-bit quantization (reduces 36GB model to ~9GB)
- LoRA fine-tuning (~2-5M trainable params vs 9B total)
- 8-bit AdamW optimizer
- fp16 mixed precision training
- 100-fold cross-validation support

### 4. Inference

#### Single Model Inference

**Gemma-2**:
```bash
python gemma2_infer.py
# Outputs: gemma2.csv
```

**Llama-3**:
```bash
python llama3_infer.py
# Outputs: llama.csv
```

**Requirements**: 2 GPUs with ~40GB total VRAM

#### Ensemble Inference (Recommended)

Use the Jupyter notebook for full ensemble pipeline:

```bash
jupyter notebook lmsys-inference-ensemble.ipynb
```

The notebook:
1. Runs Gemma-2 inference → `gemma2.csv`
2. Runs Llama-3 inference → `llama.csv`
3. Combines predictions with 70/30 weighting → `submission.csv`

```python
# Ensemble formula
final_pred = gemma2_pred * 0.7 + llama3_pred * 0.3
```

## Model Architecture

### Gemma-2 9B QLoRA Configuration

```python
LoraConfig(
    r=32,                               # Low-rank dimension
    lora_alpha=64,                      # Scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # Feed-forward
    ],
    layers_to_transform=[i for i in range(42) if i >= 2],  # Skip first 2 layers
    task_type=TaskType.SEQ_CLS,
)
```

### Input Format

```
<prompt>: {concatenated_prompt_turns}

<response_a>: {concatenated_response_a}

<response_b>: {concatenated_response_b}
```

Text is processed from JSON arrays:
- `["hello", "world"]` → `"hello world"`

## Hardware Requirements

### Training
- **Gemma-2**: 1 GPU with 40GB+ VRAM (e.g., A100, V100)
- **Llama-3 TPU**: TPU pod with 8+ cores

### Inference
- 2 GPUs with 20GB+ each
- Total ~40GB VRAM for ensemble

## Performance

- **Evaluation Metric**: Log Loss (lower is better)
- **Training Time**: ~8 hours/epoch for 78k samples (batch_size=16)
- **Inference Speed**: Optimized with dual-GPU parallelization and dynamic padding

## Output Format

Submission CSV format:
```csv
id,winner_model_a,winner_model_b,winner_tie
0,0.45,0.35,0.20
1,0.80,0.15,0.05
```

## Advanced Usage

### Cross-Validation

Run multiple folds for robust validation:

```bash
# Modify fold_idx in Config
for i in range(100):
    config.fold_idx = i
    # Run training
```

### Test Time Augmentation (TTA)

Enable response swapping in inference scripts:

```python
Config.tta = True  # Swaps response_a and response_b
```

### Custom Data Path

Modify paths in the config:

```python
Config.checkpoint = "your/model/path"
# In training script:
train_df = pd.read_csv("your/data/path.csv")
```

## Files Description

| File | Purpose | Hardware | Output |
|------|---------|----------|--------|
| `gemma_qlora_ft_v38_train.py` | Fine-tune Gemma-2 9B | 1 GPU (40GB) | Checkpoints in `output/` |
| `gemma2_infer.py` | Gemma-2 predictions | 2 GPUs | `gemma2.csv` |
| `llama3_infer.py` | Llama-3 predictions | 2 GPUs | `llama.csv` |
| `lmsys-inference-ensemble.ipynb` | Full ensemble pipeline | 2 GPUs | `submission.csv` |
| `lmsys-llama-31-tpu-train.ipynb` | Train Llama-3 on TPU | TPU pod | Model weights |

## Dependencies

```
transformers>=4.42.3
peft>=0.6.0
torch>=2.1.0
accelerate>=0.24.1
bitsandbytes>=0.41.1
datasets
pandas
numpy
scikit-learn
```

## License

This project is for educational and competition purposes. Please respect the licenses of the base models:
- Gemma-2: [Google Gemma License](https://ai.google.dev/gemma/terms)
- Llama-3: [Meta Llama License](https://llama.meta.com/llama3/license/)

## Acknowledgments

- Kaggle LMSYS Chatbot Arena competition
- Hugging Face for transformers and PEFT libraries
- Google for Gemma-2 model
- Meta for Llama-3 model
