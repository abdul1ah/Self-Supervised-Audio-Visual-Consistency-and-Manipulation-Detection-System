import os

INPUT_DIR = "/kaggle/input/datasets/mabashf/vggsound-sanity-check/content/drive/MyDrive/DNN_Project/dataset"
OUTPUT_DIR = "/kaggle/working"

FRAMES_DIR = os.path.join(INPUT_DIR, "frames")
SPECTROGRAMS_DIR = os.path.join(INPUT_DIR, "spectrograms")
METADATA_PATH = os.path.join(INPUT_DIR, "metadata.csv")
AUDIO_DIR = os.path.join(INPUT_DIR, "audio")


CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

# Ensure the output folders actually exist before the script runs
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(INPUT_DIR, 'train_metadata.csv')
VAL_CSV = os.path.join(INPUT_DIR, 'val_metadata.csv')
TEST_CSV = os.path.join(INPUT_DIR, 'test_metadata.csv')


# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================
MIN_FRAMES = 45
TARGET_FRAME_SIZE = (224, 224)   
NUM_MEL_BINS = 128               
AUDIO_SAMPLE_RATE = 16000
TARGET_AUDIO_STEPS = 313


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_WORKERS = 2 

# Seed for reproducibility
RANDOM_SEED = 42