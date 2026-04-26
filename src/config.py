import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# root of the mounted Google Drive in Colab
DRIVE_ROOT = '/content/drive/MyDrive/DNN_Project'


DATA_DIR = os.path.join(DRIVE_ROOT, 'dataset')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')
SPECTROGRAMS_DIR = os.path.join(DATA_DIR, 'spectrograms')
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')

# Checkpoint saving directory
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================
MIN_FRAMES = 5
TARGET_FRAME_SIZE = (224, 224)   
NUM_MEL_BINS = 128               
AUDIO_SAMPLE_RATE = 16000


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_WORKERS = 2 

# Seed for reproducibility
RANDOM_SEED = 42