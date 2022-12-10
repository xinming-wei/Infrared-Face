import torch

# 缩放图片的大小
IMAGE_SIZE = (256, 192)
# 分析目标的图片所在的文件夹
IMAGE_DIR = "./dataset/figure"
# 定义各个图片中人脸区域与分类的 CSV 文件
ANNOTATION_DIR = "./dataset/label"

OUT_DIR = "./out/run1"

TRAIN_VAL_TEST = [0.8, 0.1, 0.1]

# 分类列表
CLASSES = ["other", "1", "2", "3", "4", "5", "6" ]

CLASSES_MAPPING = { c: index for index, c in enumerate(CLASSES) }
# 判断是否存在对象使用的区域重叠率的阈值
IOU_POSITIVE_THRESHOLD = 0.35
IOU_NEGATIVE_THRESHOLD = 0.10
# 判断是否应该合并重叠区域的重叠率阈值
IOU_MERGE_THRESHOLD = 0.35

# 用于启用 GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 20
MAX_EPOCH = 10000

# 定义这个块的实际出通道是 channels_out 的几倍，depends on available GPU memory
# RESNET_EXPAND = 4
