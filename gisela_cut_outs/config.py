import dotenv
import os

dotenv.load_dotenv()

IMG_SIZE = 3500
NR_OF_CROPS_PER_IMAGE = 5
MAX_ITERATIONS = 100
#MIN_LABELS_PER_IMAGE = 2
#MIN_COVERAGE = 0.2

MAG_LIST_INDEX = 6

SHOW_IMAGES = False
CLEAR_OUTPUT_DIR = True

ORG_ID = os.getenv("ORG_ID")
WK_TOKEN = os.getenv("WK_TOKEN")
        
