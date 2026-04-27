import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from config import *
from dataset import get_dataloader
from models import AudioVisualFusion
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Evaluation on device: {device}\n")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No saved model found at {checkpoint_path}. Run train.py first!")


    print("Loading Test DataLoader...")

    test_loader = get_dataloader(TEST_CSV, batch_size=BATCH_SIZE, shuffle=False)
    

    model = AudioVisualFusion().to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
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

            logits = model(visual_batch, audio_batch)
            
            probs = torch.sigmoid(logits).cpu().numpy()            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

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
    true_negatives, false_positives, false_negatives, true_positives = cm.ravel()


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
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, label=f'Model ROC (AUC = {roc_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    plt.xlabel('False Positive Rate (Mistaking a Mismatch for a Match)')
    plt.ylabel('True Positive Rate (Successfully finding a Match)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = os.path.join(CHECKPOINT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Mismatch', 'Pred: Match'], 
                yticklabels=['Actual: Mismatch', 'Actual: Match'])
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print(f"Visuals saved to Drive inside the checkpoints folder!")

if __name__ == "__main__":
    main()