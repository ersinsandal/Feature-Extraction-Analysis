from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"

RANDOM_STATE = 42

# Eğer None yaparsan tüm dataset kullanılır.
# Hızlı deneme için 10000 / 2000 iyi.
TRAIN_LIMIT = 10000
TEST_LIMIT = 2000

# Computer Vision projesinde kullanılacak 3 dataset
DATASETS = ["fashion_mnist", "cifar10", "kth_tips"]

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# KTH-TIPS texture dataset ayarları
KTH_TIPS_FOLDER_NAME = "KTH_TIPS"
KTH_TIPS_IMAGE_SIZE = (128, 128)
KTH_TIPS_TEST_SIZE = 0.2