import torch
import os

from config import *
from util import *
from model import MyModel

test_record = os.path.join(OUT_DIR, "test_images.txt")

def prepare_save_batch(batch, image_tensors, image_boxes_labels, image_paths):
    """准备训练 - 保存单个批次的数据"""
    # 按索引值列表生成输入和输出 tensor 对象的函数
    def split_dataset(indices, mode="train"):
        image_in = []
        boxes_labels_out = {}
        for new_image_index, original_image_index in enumerate(indices.tolist()):
            image_in.append(image_tensors[original_image_index])
            boxes_labels_out[new_image_index] = image_boxes_labels[original_image_index]
            if mode == "test":
                with open(test_record, 'a+') as f:
                    print(image_paths[original_image_index], file=f)
        tensor_image_in = torch.stack(image_in) # 维度: B,C,W,H
        return tensor_image_in, boxes_labels_out

    # 切分训练集 (80%)，验证集 (10%) 和测试集 (10%)
    random_indices = torch.randperm(len(image_tensors))
    training_indices = random_indices[:int(len(random_indices)*TRAIN_VAL_TEST[0])]
    validating_indices = random_indices[int(len(random_indices)*TRAIN_VAL_TEST[0]):int(len(random_indices)*(TRAIN_VAL_TEST[0]+TRAIN_VAL_TEST[1])):]
    testing_indices = random_indices[int(len(random_indices)*(TRAIN_VAL_TEST[0]+TRAIN_VAL_TEST[1])):]

    training_set = split_dataset(training_indices)
    validating_set = split_dataset(validating_indices, mode="val")
    testing_set = split_dataset(testing_indices, mode="test")

    # 保存到硬盘
    save_tensor(training_set, os.path.join(OUT_DIR, f"data/training_set.{batch}.pt"))
    save_tensor(validating_set, os.path.join(OUT_DIR, f"data/validating_set.{batch}.pt"))
    save_tensor(testing_set, os.path.join(OUT_DIR, f"data/testing_set.{batch}.pt"))
    print(f"batch {batch} saved")

def prepare():
    """准备训练"""
    # 数据集转换到 tensor 以后会保存在 ${OUT_DIR}/data/ 文件夹下
    if not os.path.isdir(os.path.join(OUT_DIR, "data")):
        os.makedirs(os.path.join(OUT_DIR, "data"))

    # 加载图片和图片对应的区域与分类列表
    # { 图片名: [ 区域与分类, 区域与分类, .. ] }
    box_map = defaultdict(lambda: [])
    for filename in os.listdir(IMAGE_DIR):
        # xml_path = os.path.join(ANNOTATION_DIR, filename.split(".")[0] + ".xml")
        xml_path = os.path.join(ANNOTATION_DIR, filename.split(".")[0] + "." + filename.split(".")[1] + ".xml")

        if not os.path.isfile(xml_path):
            continue
        tree = ET.ElementTree(file=xml_path)
        objects = tree.findall("object")
        for obj in objects:
            # class_name = obj.find("name").text
            class_name = filename.split(".")[0]
            x1 = int(obj.find("bndbox/xmin").text)
            x2 = int(obj.find("bndbox/xmax").text)
            y1 = int(obj.find("bndbox/ymin").text)
            y2 = int(obj.find("bndbox/ymax").text)
            box_map[filename].append((x1, y1, x2-x1, y2-y1, CLASSES_MAPPING[class_name]))
    # 保存图片和图片对应的分类与区域列表
    batch = 0
    image_tensors = [] # 图片列表
    image_boxes_labels = {} # 图片对应的真实区域与分类列表，和候选区域与区域偏移
    image_paths = []
    for filename, original_boxes_labels in box_map.items():
        image_path = os.path.join(IMAGE_DIR, filename)
        image_paths.append(image_path)

        with Image.open(image_path) as img_original: # 加载原始图片
            sw, sh = img_original.size # 原始图片大小
            img = resize_image(img_original) # 缩放图片
            image_index = len(image_tensors) # 图片在批次中的索引值
            image_tensors.append(image_to_tensor(img)) # 添加图片到列表
            true_boxes_labels = [] # 图片对应的真实区域与分类列表
        # 添加真实区域与分类列表
        for box_label in original_boxes_labels:
            x, y, w, h, label = box_label
            x, y, w, h = map_box_to_resized_image((x, y, w, h), sw, sh) # 缩放实际区域
            if w < 10 or h < 10:
                continue # 缩放后区域过小
            # 检查计算是否有问题
            # child_img = img.copy().crop((x, y, x+w, y+h))
            # child_img.save(f"{filename}_{x}_{y}_{w}_{h}_{label}.png")
            true_boxes_labels.append((label, (x, y, w, h)))
        # 如果图片中的所有区域都过小则跳过
        if not true_boxes_labels:
            image_tensors.pop()
            image_index = len(image_tensors)
            continue
        # 根据锚点列表寻找候选区域，并计算区域偏移
        actual_rpn_labels = []
        actual_rpn_labels_mask = []
        actual_rpn_offsets = []
        actual_rpn_offsets_mask = []
        positive_index_set = set()
        for index, anchor_box in enumerate(MyModel.Anchors):
            # 如果候选区域和任意一个实际区域重叠率大于阈值，则认为是正样本
            # 如果候选区域和所有实际区域重叠率都小于阈值，则认为是负样本
            # 重叠率介于两个阈值之间的区域不参与学习
            iou_list = [ calc_iou(anchor_box, true_box) for (_, true_box) in true_boxes_labels ]
            positive_index = next((index for index, iou in enumerate(iou_list) if iou > IOU_POSITIVE_THRESHOLD), None)
            is_negative = all(iou < IOU_NEGATIVE_THRESHOLD for iou in iou_list)
            if positive_index is not None:
                positive_index_set.add(positive_index)
                actual_rpn_labels.append(1)
                actual_rpn_labels_mask.append(index)
                # 只有包含对象的区域参需要调整偏移
                true_box = true_boxes_labels[positive_index][1]
                actual_rpn_offsets.append(calc_box_offset(anchor_box, true_box))
                actual_rpn_offsets_mask.append(index)
            elif is_negative:
                actual_rpn_labels.append(0)
                actual_rpn_labels_mask.append(index)
        # 输出找不到候选区域的真实区域，调整锚点生成参数时使用
        # for index in range(len(true_boxes_labels)):
        #    if index not in positive_index_set:
        #        print(true_boxes_labels[index][1])
        # print("-----")
        # 如果一个候选区域都找不到则跳过
        if not positive_index_set:
            image_tensors.pop()
            image_index = len(image_tensors)
            continue
        image_boxes_labels[image_index] = (
            true_boxes_labels,
            torch.tensor(actual_rpn_labels, dtype=torch.long),
            torch.tensor(actual_rpn_labels_mask, dtype=torch.long),
            torch.tensor(actual_rpn_offsets, dtype=torch.float),
            torch.tensor(actual_rpn_offsets_mask, dtype=torch.long))
        # 保存批次
        if len(image_tensors) >= BATCH_SIZE:
            prepare_save_batch(batch, image_tensors, image_boxes_labels, image_paths)
            image_tensors.clear()
            image_boxes_labels.clear()
            image_paths.clear()
            batch += 1
    # 保存剩余的批次
    if len(image_tensors) > 10:
        prepare_save_batch(batch, image_tensors, image_boxes_labels, image_paths)
