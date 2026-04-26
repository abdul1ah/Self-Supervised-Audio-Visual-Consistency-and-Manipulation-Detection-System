import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from config import *
from dataset import get_dataloader
from models import AudioVisualFusion
from loss import AudioVisualLoss

def calculate_epoch_metrics(all_labels, all_probs):
    """
    Calculates advanced metrics for an entire epoch using scikit-learn.
    """
    all_preds = (all_probs > 0.5).astype(float)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0 
        
    return acc, precision, recall, f1, roc_auc

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on device: {device}\n")

    print("Loading DataLoaders...")
    train_loader = get_dataloader(TRAIN_CSV, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(VAL_CSV, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AudioVisualFusion().to(device)
    criterion = AudioVisualLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    
    # THE MAIN LOOP
    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")
        
        # TRAINING PHASE
        model.train() 
        train_loss = 0.0
        
        # Arrays to store every prediction and label for the epoch
        train_all_labels = []
        train_all_probs = []
        
        loop = tqdm(train_loader, leave=False, desc="Training")
        for visual_batch, audio_batch, labels in loop:
            visual_batch = visual_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(visual_batch, audio_batch)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Store labels and probabilities for metric calculation
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_all_probs.extend(probs)
            train_all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Calculate Advanced Training Metrics
        train_acc, train_prec, train_rec, train_f1, train_auc = calculate_epoch_metrics(
            np.array(train_all_labels), np.array(train_all_probs)
        )

        # VALIDATION PHASE
        model.eval() 
        val_loss = 0.0
        
        val_all_labels = []
        val_all_probs = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=False, desc="Validating")
            for visual_batch, audio_batch, labels in val_loop:
                visual_batch = visual_batch.to(device)
                audio_batch = audio_batch.to(device)
                labels = labels.to(device)

                logits = model(visual_batch, audio_batch)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_all_probs.extend(probs)
                val_all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Advanced Validation Metrics
        val_acc, val_prec, val_rec, val_f1, val_auc = calculate_epoch_metrics(
            np.array(val_all_labels), np.array(val_all_probs)
        )

        # EPOCH SUMMARY & CHECKPOINTING
        print(f"TRAIN | Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f} | AUC: {train_auc:.4f}")
        print(f"VAL   | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Validation Loss improved! Saved snapshot to {save_path}")
        else:
            print("Validation Loss did not improve.")

if __name__ == "__main__":
    main()