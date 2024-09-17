import dotenv
import os

dotenv.load_dotenv()

IMG_SIZE = 512
MIN_LABELS_PER_IMAGE = 2
MIN_COVERAGE = 0.2

SHOW_IMAGES = False
CLEAR_OUTPUT_DIR = False

ORG_ID = os.getenv("ORG_ID")
WK_TOKEN = os.getenv("WK_TOKEN")

TRAINING_DATASET_FILE = "datasets/dataset.yaml"
TRAINING_EPOCHS = 10

MODEL_SAVE_FILE_NAME = "latest_model.pt"

TEST_DATA_DIR = "dl/test_data/"
TEST_IMAGE_RESULT_FILE_NAME = "result.png"
        
