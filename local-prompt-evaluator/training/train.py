"""
Training Loop for DistilBERT Prompt Evaluator

Trains the model on prepared data with metrics tracking.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = nn.BCELoss()(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, auc


def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.BCELoss()(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, accuracy, f1, auc


def train_model():
    """Main training function."""
    from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG
    from training.model import PromptEvaluatorModel
    from training.dataset import create_dataloaders
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Setup
    device = torch.device(TRAINING_CONFIG["device"])
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CONFIG["base_model"])
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        DATASET_CONFIG["train_file"],
        DATASET_CONFIG["test_file"],
        tokenizer,
        TRAINING_CONFIG["batch_size"],
        MODEL_CONFIG["max_length"],
        TRAINING_CONFIG["num_workers"],
    )
    
    # Create model
    model = PromptEvaluatorModel(
        base_model_name=MODEL_CONFIG["base_model"],
        dropout=MODEL_CONFIG["dropout"],
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    
    total_steps = len(train_loader) * TRAINING_CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )
    
    # Training loop
    history = []
    best_f1 = 0
    
    for epoch in range(TRAINING_CONFIG["epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']}")
        
        # Train
        train_loss, train_acc, train_f1, train_auc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Evaluate
        test_loss, test_acc, test_f1, test_auc = evaluate(
            model, test_loader, device
        )
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        logger.info(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
        })
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            model.save_pretrained(MODEL_CONFIG["model_path"])
            tokenizer.save_pretrained(MODEL_CONFIG["tokenizer_path"])
            logger.info(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(TRAINING_CONFIG["log_file"], index=False)
    logger.info(f"✓ Saved training history to {TRAINING_CONFIG['log_file']}")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
