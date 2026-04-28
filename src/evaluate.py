import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from config import *
from dataset import get_dataloader
from models import AudioVisualFusion

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation on device: {device}\n")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No saved model found at {checkpoint_path}. Run train.py first!")

    print("Loading Test DataLoader...")
    test_loader = get_dataloader(TEST_CSV, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AudioVisualFusion()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    if torch.cuda.device_count() > 1:
        print(f"Evaluation utilizing {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    print("\nRunning Final Inference on Test Set...")
    with torch.no_grad():
        loop = tqdm(test_loader, leave=False, desc="Testing")
        for visual_batch, audio_batch, labels in loop:

            visual_batch = visual_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits = model(visual_batch, audio_batch)
            
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()            
            all_probs.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            
            flat_labels = labels.squeeze().cpu().numpy()
            all_labels.extend(flat_labels.tolist() if flat_labels.ndim > 0 else [flat_labels.item()])

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(float)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    
    if cm.size == 4:
        true_negatives, false_positives, false_negatives, true_positives = cm.ravel()
    else:
        true_negatives, false_positives, false_negatives, true_positives = 0, 0, 0, 0
        print("Warning: Confusion matrix is not 2x2. Test set might be missing a class.")

    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"ROC-AUC Score:     {roc_auc:.4f}")
    print(f"F1-Score:          {f1:.4f}")
    print(f"Precision:         {precision:.4f} (When it guesses 'Match', it is right {precision*100:.1f}% of the time)")
    print(f"Recall:            {recall:.4f} (It successfully finds {recall*100:.1f}% of all real Matches)")
    print("-" * 50)
    print("CONFUSION MATRIX:")
    print(f"True Positives (Correctly identified Match):       {true_positives}")
    print(f"True Negatives (Correctly identified Mismatch):    {true_negatives}")
    print(f"False Positives (Tricked! Guessed Match wrongly):  {false_positives}")
    print(f"False Negatives (Missed it! Guessed Mismatch):     {false_negatives}")
    print("="*50 + "\n")

    print("Generating visual plots...")
    
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))

        plt.plot(fpr, tpr, label=f'Model ROC (AUC = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        
        plt.xlabel('False Positive Rate (Mistaking a Mismatch for a Match)')
        plt.ylabel('True Positive Rate (Successfully finding a Match)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        roc_path = os.path.join(ARTIFACTS_DIR, 'roc_curve.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
    else:
        print("Skipped ROC Curve generation (only one class present in test set).")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Mismatch', 'Pred: Match'], 
                yticklabels=['Actual: Mismatch', 'Actual: Match'])
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print(f"Visuals saved to the Kaggle working/artifacts directory!")

if __name__ == "__main__":
    main()