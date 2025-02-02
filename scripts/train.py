#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAI (HelpingAI) Training Pipeline
Author: Abhay Koul
Description: Complete training pipeline for emotionally intelligent language model
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
import torch
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from datasets import load_dataset
from litgpt.tokenizer import Tokenizer

console = Console()

class Logger:
    """Simple file-based logger"""
    
    def __init__(self, log_dir: str, debug: bool = False):
        self.debug = debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"hai_training_{timestamp}.log")
        self.error_file = os.path.join(log_dir, f"hai_errors_{timestamp}.log")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files with header
        for file in [self.log_file, self.error_file]:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(f"=== HAI Training Log ===\nStarted at: {timestamp}\n\n")
    
    def _write_log(self, file: str, level: str, message: str) -> None:
        """Write a log message to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
        
        # Also print to console
        console.print(f"[{level}] {message}")
    
    def info(self, message: str) -> None:
        """Log info message"""
        self._write_log(self.log_file, "INFO", message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self._write_log(self.log_file, "WARNING", message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self._write_log(self.error_file, "ERROR", message)
    
    def debug(self, message: str) -> None:
        """Log debug message if debug mode is enabled"""
        if self.debug:
            self._write_log(self.log_file, "DEBUG", message)

class HAITrainer:
    """HAI Training Pipeline Manager"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[Tokenizer] = None
        
        # Setup directories
        self.setup_directories()
        
        # Initialize logger
        self.logger = Logger(self.logs_dir, args.debug)
        
        # Initialize wandb
        if not args.disable_wandb:
            self.setup_wandb()
    
    def setup_directories(self) -> None:
        """Create and validate necessary directories"""
        required_dirs = {
            'data': self.args.data_dir,
            'output': self.args.output_dir,
            'config': self.args.config_dir,
            'tokenizer': os.path.join(self.args.data_dir, 'tokenizer'),
            'pretrain': os.path.join(self.args.data_dir, 'pretrain'),
            'emotional': os.path.join(self.args.data_dir, 'emotional'),
            'checkpoints': os.path.join(self.args.output_dir, 'checkpoints'),
            'logs': os.path.join(self.args.output_dir, 'logs')
        }
        
        for name, path in required_dirs.items():
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                raise ValueError(f"Failed to create {name} directory at {path}")
            setattr(self, f"{name}_dir", path)
    
    def setup_wandb(self) -> None:
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project="HAI-Training",
                name=f"HAI-{self.args.stage}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args),
                dir=self.logs_dir
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.args.disable_wandb = True
    
    def load_config(self, stage: str) -> None:
        """Load and validate configuration file"""
        config_file = os.path.join(
            self.config_dir,
            'pretrain-model.yaml' if stage == 'pretrain' else 'contrain-model.yaml'
        )
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Validate required config fields
            required_fields = ['model_config', 'train', 'eval', 'optimizer']
            missing_fields = [f for f in required_fields if f not in self.config]
            if missing_fields:
                raise ValueError(f"Missing required config fields: {missing_fields}")
            
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {config_file}: {e}")
    
    def train_tokenizer(self) -> None:
        """Train custom tokenizer for emotional intelligence"""
        with console.status("[bold green]Training tokenizer...") as status:
            try:
                from train_tokenizer import create_tokenizer
                
                # Create and train tokenizer
                tokenizer = create_tokenizer()
                
                # Save tokenizer
                tokenizer_path = os.path.join(self.tokenizer_dir, "tokenizer.json")
                tokenizer.save(tokenizer_path)
                
                self.logger.info(f"Tokenizer saved to {tokenizer_path}")
                
            except Exception as e:
                self.logger.error(f"Tokenizer training failed: {e}")
                raise
    
    def prepare_datasets(self) -> None:
        """Prepare all necessary datasets"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            try:
                # Load tokenizer
                tokenizer = Tokenizer.from_file(os.path.join(self.tokenizer_dir, "tokenizer.json"))
                
                # Prepare pre-training data
                task1 = progress.add_task("Preparing pre-training data...", total=100)
                from prepare_pretrain_dataset import batch_iterator as pretrain_iterator
                from prepare_pretrain_dataset import tokenize_fn as pretrain_tokenize
                
                # Define all pretraining datasets
                pretrain_configs = [
                    {
                        "path": "Abhaykoul/test",
                        "name": "default",
                        "split": "train"
                    },
                    {
                        "path": "JeanKaddour/minipile",
                        "split": "train"
                    },
                    {
                        "path": "OEvortex/EmotionalIntelligence-75k",
                        "split": "train"
                    },
                    {
                        "path": "OEvortex/Med-emo",
                        "split": "train"
                    }
                ]
                
                # Process and combine pretraining data
                pretrain_data = []
                for idx, config in enumerate(pretrain_configs):
                    progress.update(task1, description=f"Processing dataset {idx + 1}/{len(pretrain_configs)}...")
                    try:
                        data = pretrain_tokenize(config, tokenizer)
                        pretrain_data.extend(data)
                        progress.advance(task1, 25)  # 25% per dataset
                        self.logger.debug(f"Successfully processed {config['path']}")
                    except Exception as e:
                        self.logger.warning(f"Failed to process dataset {config['path']}: {e}")
                
                # Save pre-training data
                pretrain_path = os.path.join(self.pretrain_dir, "pretrain_data.pt")
                torch.save(pretrain_data, pretrain_path)
                self.logger.info(f"Saved {len(pretrain_data)} pretraining samples to {pretrain_path}")
                
                # Prepare emotional fine-tuning data
                task2 = progress.add_task("Preparing emotional data...", total=100)
                from prepare_contrain_dataset import batch_iterator as emotional_iterator
                from prepare_contrain_dataset import tokenize_fn as emotional_tokenize
                from cognition_dataset import self_cognition_messages
                
                # Define all emotional datasets
                emotional_configs = [
                    {
                        "path": "OEvortex/EmotionalIntelligence-75k",
                        "split": "train"
                    },
                    {
                        "path": "OEvortex/Med-emo",
                        "split": "train"
                    },
                    {
                        "path": "OEvortex/HelpingAI2.5-English-openemotions",
                        "split": "train"
                    },
                    {
                        "path": "OEvortex/HelpingAI2.5-hinglish-openemotions",
                        "split": "train"
                    },
                    {
                        "data": self_cognition_messages,  # Add self-cognition messages
                        "format": lambda x: f"{x['input']}\n{x['output']}"
                    }
                ]
                
                # Process and combine emotional data
                emotional_data = []
                for idx, config in enumerate(emotional_configs):
                    progress.update(task2, description=f"Processing emotional dataset {idx + 1}/{len(emotional_configs)}...")
                    try:
                        data = emotional_tokenize(config, tokenizer)
                        emotional_data.extend(data)
                        progress.advance(task2, 20)  # 20% per dataset
                        self.logger.debug(f"Successfully processed emotional dataset {config.get('path', 'self_cognition')}")
                    except Exception as e:
                        self.logger.warning(f"Failed to process emotional dataset {config.get('path', 'self_cognition')}: {e}")
                
                # Save emotional data
                emotional_path = os.path.join(self.emotional_dir, "emotional_data.pt")
                torch.save(emotional_data, emotional_path)
                self.logger.info(f"Saved {len(emotional_data)} emotional samples to {emotional_path}")
                
            except Exception as e:
                self.logger.error(f"Dataset preparation failed: {e}")
                raise
    
    def train_model(self, stage: str) -> None:
        """Train the model for specified stage"""
        try:
            self.load_config(stage)
            
            # Load appropriate datasets
            data_path = os.path.join(
                self.pretrain_dir if stage == 'pretrain' else self.emotional_dir,
                f"{stage}_data.pt"
            )
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Training data not found at {data_path}")
            
            # Import litgpt training module
            if stage == 'pretrain':
                from litgpt.pretrain import main as train_main
            else:
                from litgpt.finetune import main as train_main
            
            # Update config with correct paths
            self.config.update({
                "data_path": data_path,
                "out_dir": os.path.join(self.checkpoints_dir, stage),
                "tokenizer_path": os.path.join(self.tokenizer_dir, "tokenizer.json")
            })
            
            # Start training
            self.logger.info(f"Starting {stage} training...")
            train_main(self.config)
            
            # Save final model
            output_path = os.path.join(self.output_dir, f"hai_{stage}_final")
            self.save_model(output_path)
            
            self.logger.info(f"{stage.capitalize()} training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, output_path: str) -> None:
        """Save model checkpoints and configs"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Save model config
            config_path = os.path.join(output_path, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            
            # Copy final model files
            if self.args.stage == 'pretrain':
                src_dir = os.path.join(self.checkpoints_dir, 'pretrain', 'final')
            else:
                src_dir = os.path.join(self.checkpoints_dir, 'emotional_finetune', 'final')
            
            if os.path.exists(src_dir):
                import shutil
                for file in os.listdir(src_dir):
                    shutil.copy2(
                        os.path.join(src_dir, file),
                        os.path.join(output_path, file)
                    )
            
            self.logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

def setup_args() -> argparse.Namespace:
    """Setup and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='HAI (HelpingAI) Training Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing the training data')
    parser.add_argument('--config-dir', type=str, default='../configs',
                       help='Directory containing configuration files')
    parser.add_argument('--output-dir', type=str, default='../output',
                       help='Directory to save model outputs')
    parser.add_argument('--stage', type=str, 
                       choices=['tokenizer', 'pretrain', 'emotional_finetune'],
                       default='tokenizer',
                       help='Training stage to execute')
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    return parser.parse_args()

def main() -> None:
    """Main training pipeline"""
    args = setup_args()
    
    try:
        trainer = HAITrainer(args)
        
        if args.stage == 'tokenizer':
            trainer.train_tokenizer()
            trainer.prepare_datasets()
        
        elif args.stage == 'pretrain':
            if not args.resume:
                trainer.prepare_datasets()
            trainer.train_model('pretrain')
        
        elif args.stage == 'emotional_finetune':
            if not args.resume:
                trainer.prepare_datasets()
            trainer.train_model('emotional_finetune')
        
        trainer.logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        if hasattr(trainer, 'logger'):
            trainer.logger.error(f"Training pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        if not args.disable_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()
