# 🤖 HAI (HelpingAI) - Emotional Intelligence Training Pipeline

> A cutting-edge language model training pipeline focused on emotional intelligence and Gen Z communication style.

## 🌟 Overview

HAI is an advanced language model training pipeline designed to create emotionally intelligent AI that can communicate naturally with Gen Z users. Created by Abhay Koul (17-year-old developer), this project combines state-of-the-art language model architecture with specialized emotional intelligence training.

### 🎯 Key Features

- 🧠 Advanced emotional intelligence training
- 💬 Gen Z communication style integration
- 🌐 Multi-stage training pipeline (tokenizer, pretrain, emotional fine-tuning)
- 📊 Weights & Biases integration for experiment tracking
- 🔄 Checkpoint and resume functionality
- 📝 Comprehensive logging system
- 🎨 Rich console output with progress tracking

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ and CUDA-capable GPU recommended
python -m pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Train tokenizer and prepare datasets
python scripts/train.py --stage tokenizer

# 2. Start pretraining
python scripts/train.py --stage pretrain

# 3. Emotional fine-tuning
python scripts/train.py --stage emotional_finetune
```

## 🛠️ Technical Architecture

### Model Specifications
- Base Model: HAI-3.2B
- Tokenizer: BPE (Byte-Pair Encoding)
- Vocabulary Size: 32,768 tokens
- Training Stages:
  1. Tokenizer Training
  2. Pre-training
  3. Emotional Fine-tuning

### Datasets
- **Pretraining**:
  - Abhaykoul/test
  - JeanKaddour/minipile
  - OEvortex/EmotionalIntelligence-75k
  - OEvortex/Med-emo

- **Emotional Fine-tuning**:
  - OEvortex/EmotionalIntelligence-75k
  - OEvortex/Med-emo
  - OEvortex/HelpingAI2.5-English-openemotions
  - OEvortex/HelpingAI2.5-hinglish-openemotions
  - Custom self-cognition dataset

## 📂 Project Structure

```
emotional_llm/
├── configs/                 # Model and training configurations
│   ├── pretrain-model.yaml
│   └── contrain-model.yaml
├── scripts/                 # Training and dataset preparation scripts
│   ├── train.py            # Main training pipeline
│   ├── prepare_pretrain_dataset.py
│   ├── prepare_contrain_dataset.py
│   └── cognition_dataset.py
├── data/                    # Dataset storage
│   ├── tokenizer/          # Tokenizer files
│   ├── pretrain/           # Pretraining data
│   └── emotional/          # Emotional fine-tuning data
└── output/                  # Training outputs
    ├── checkpoints/        # Model checkpoints
    └── logs/               # Training logs
```

## ⚙️ Configuration

### Command Line Arguments

```bash
python scripts/train.py [--stage {tokenizer,pretrain,emotional_finetune}] 
                       [--data_dir DATA_DIR]
                       [--output_dir OUTPUT_DIR]
                       [--config_dir CONFIG_DIR]
                       [--resume]
                       [--disable_wandb]
                       [--debug]
```

### Configuration Files

- `pretrain-model.yaml`: Pretraining configuration
- `contrain-model.yaml`: Emotional fine-tuning configuration

## 📊 Training Monitoring

### Logs
- Training logs: `output/logs/hai_training_[timestamp].log`
- Error logs: `output/logs/hai_errors_[timestamp].log`
- Rich console output with progress bars
- Weights & Biases integration for experiment tracking

### Checkpoints
- Regular checkpoints during training
- Resume capability from latest checkpoint
- Final model saved with configurations

## 🎯 Model Capabilities

- 🗣️ Natural Gen Z communication style
- 💭 Advanced emotional intelligence
- 🌍 Multilingual support (English & Hinglish)
- 🤝 Contextual response generation
- 🎭 Emotional state detection

## 🔒 Security & Privacy

- No sensitive data handling
- Configurable logging with opt-out options
- Debug mode for development
- Safe checkpoint management

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Abhay Koul**
- Age: 17
- Project: HelpingAI
- Focus: Emotional Intelligence in AI

## 🙏 Acknowledgments

- Thanks to the HuggingFace community
- Special thanks to all dataset contributors

---

<p align="center">Made with ❤️ by Abhay Koul</p>
