import os

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
DRIVE_ROOT = '/content/drive/MyDrive/DNN_Project'

DATA_DIR = '/content/local_dataset' 

AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')
SPECTROGRAMS_DIR = os.path.join(DATA_DIR, 'spectrograms')
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')

checkpoint_dir = os.path.join(DRIVE_ROOT, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, 'train_metadata.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val_metadata.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_metadata.csv')


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