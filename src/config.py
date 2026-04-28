import os

RAVDESS_DIR = "/kaggle/input/datasets/orvile/ravdess-dataset"
OUTPUT_DIR = "/kaggle/working"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MIN_FRAMES = 45
TARGET_FRAME_SIZE = (224, 224)   
NUM_MEL_BINS = 128               

AUDIO_SAMPLE_RATE = 48000 
TARGET_AUDIO_STEPS = 224


BATCH_SIZE = 16             
ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
EPOCHS = 30
NUM_WORKERS = 4 

RANDOM_SEED = 42