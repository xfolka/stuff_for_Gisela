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
        
