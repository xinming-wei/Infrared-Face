import torch
import itertools
import os, sys
import numpy as np
from matplotlib import pyplot
from PIL import ImageDraw
from collections import Counter

from config import *
from model import MyModel
from util import *
from data_prepare import *


def train():
    """开始训练"""
    # 创建模型实例
    model = MyModel().to(device)

    # 创建多任务损失计算器
    loss_function = MyModel.loss_function

    # 创建参数调整器
    optimizer = torch.optim.Adam(model.parameters())

    # 记录训练集和验证集的正确率变化
    training_rpn_accuracy_history = []
    training_cls_accuracy_history = []
    validating_rpn_accuracy_history = []
    validating_cls_accuracy_history = []

    # 记录最高的验证集正确率
    validating_rpn_accuracy_highest = -1
    validating_rpn_accuracy_highest_epoch = 0
    validating_cls_accuracy_highest = -1
    validating_cls_accuracy_highest_epoch = 0

    # 读取批次的工具函数
    def read_batches(base_path):
        for batch in itertools.count():
            path = f"{base_path}.{batch}.pt"
            if not os.path.isfile(path):
                break
            x, y = load_tensor(path)
            yield x.to(device), y

    # 计算正确率的工具函数
    calc_accuracy = MyModel.calc_accuracy
    
    if not os.path.isdir(OUT_DIR):
        print("Error: You should run 'python main.py prepare' before training")
        exit(1)

    def print_log(str):
        with open(os.path.join(OUT_DIR, "train.log"), 'a+') as f:
            print(str, file=f)

    if os.path.isfile(os.path.join(OUT_DIR, "train.log")):
        os.remove(os.path.join(OUT_DIR, "train.log"))

    # 开始训练过程
    for epoch in range(1, MAX_EPOCH):
        print(f"epoch: {epoch}")
        print_log(f"epoch: {epoch}")

        # 根据训练集训练并修改参数
        # 切换模型到训练模式，将会启用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.train()
        training_rpn_accuracy_list = []
        training_cls_accuracy_list = []
        for batch_index, batch in enumerate(read_batches(os.path.join(OUT_DIR, "data/training_set"))):
            # 划分输入和输出
            batch_x, batch_y = batch
            # 计算预测值
            predicted = model(batch_x)
            # 计算损失
            loss = loss_function(predicted, batch_y)
            # 从损失自动微分求导函数值
            loss.backward()
            # 使用参数调整器调整参数
            optimizer.step()
            # 清空导函数值
            optimizer.zero_grad()
            # 记录这一个批次的正确率，torch.no_grad 代表临时禁用自动微分功能
            with torch.no_grad():
                training_batch_rpn_accuracy, training_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            # 输出批次正确率
            training_rpn_accuracy_list.append(training_batch_rpn_accuracy)
            training_cls_accuracy_list.append(training_batch_cls_accuracy)
            print_log(f"epoch: {epoch}, batch: {batch_index}: " +
                f"batch rpn accuracy: {training_batch_rpn_accuracy}, cls accuracy: {training_batch_cls_accuracy}")
            print(f"epoch: {epoch}, batch: {batch_index}: " +
                f"batch rpn accuracy: {training_batch_rpn_accuracy}, cls accuracy: {training_batch_cls_accuracy}")
        training_rpn_accuracy = sum(training_rpn_accuracy_list) / len(training_rpn_accuracy_list)
        training_cls_accuracy = sum(training_cls_accuracy_list) / len(training_cls_accuracy_list)
        training_rpn_accuracy_history.append(training_rpn_accuracy)
        training_cls_accuracy_history.append(training_cls_accuracy)
        print(f"training rpn accuracy: {training_rpn_accuracy}, cls accuracy: {training_cls_accuracy}")
        print_log(f"training rpn accuracy: {training_rpn_accuracy}, cls accuracy: {training_cls_accuracy}")
        # 检查验证集
        # 切换模型到验证模式，将会禁用自动微分，批次正规化 (BatchNorm) 与 Dropout
        model.eval()
        validating_rpn_accuracy_list = []
        validating_cls_accuracy_list = []
        for batch in read_batches(os.path.join(OUT_DIR, "data/validating_set")):
            batch_x, batch_y = batch
            predicted = model(batch_x)
            validating_batch_rpn_accuracy, validating_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
            validating_rpn_accuracy_list.append(validating_batch_rpn_accuracy)
            validating_cls_accuracy_list.append(validating_batch_cls_accuracy)
        validating_rpn_accuracy = sum(validating_rpn_accuracy_list) / len(validating_rpn_accuracy_list)
        validating_cls_accuracy = sum(validating_cls_accuracy_list) / len(validating_cls_accuracy_list)
        validating_rpn_accuracy_history.append(validating_rpn_accuracy)
        validating_cls_accuracy_history.append(validating_cls_accuracy)
        print(f"validating rpn accuracy: {validating_rpn_accuracy}, cls accuracy: {validating_cls_accuracy}")
        print_log(f"validating rpn accuracy: {validating_rpn_accuracy}, cls accuracy: {validating_cls_accuracy}")

        # 记录最高的验证集正确率与当时的模型状态，判断是否在 20 次训练后仍然没有刷新记录
        if validating_rpn_accuracy > validating_rpn_accuracy_highest:
            validating_rpn_accuracy_highest = validating_rpn_accuracy
            validating_rpn_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
            print("highest rpn validating accuracy updated")
            print_log("highest rpn validating accuracy updated")

        elif validating_cls_accuracy > validating_cls_accuracy_highest:
            validating_cls_accuracy_highest = validating_cls_accuracy
            validating_cls_accuracy_highest_epoch = epoch
            save_tensor(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
            print("highest cls validating accuracy updated")
            print_log("highest cls validating accuracy updated")

        elif (epoch - validating_rpn_accuracy_highest_epoch > 20 and
            epoch - validating_cls_accuracy_highest_epoch > 20):
            # 在 20 次训练后仍然没有刷新记录，结束训练
            print("stop training because highest validating accuracy not updated in 20 epoches")
            print_log("stop training because highest validating accuracy not updated in 20 epoches")
            break

    # 使用达到最高正确率时的模型状态
    print(f"highest rpn validating accuracy: {validating_rpn_accuracy_highest}",
        f"from epoch {validating_rpn_accuracy_highest_epoch}")
    print(f"highest cls validating accuracy: {validating_cls_accuracy_highest}",
        f"from epoch {validating_cls_accuracy_highest_epoch}")
    print_log(f"highest rpn validating accuracy: {validating_rpn_accuracy_highest} from epoch {validating_rpn_accuracy_highest_epoch}")
    print_log(f"highest cls validating accuracy: {validating_cls_accuracy_highest} from epoch {validating_cls_accuracy_highest_epoch}")
    model.load_state_dict(load_tensor(os.path.join(OUT_DIR, "model.pt")))

    # 检查测试集
    testing_rpn_accuracy_list = []
    testing_cls_accuracy_list = []
    for batch in read_batches(os.path.join(OUT_DIR, "data/testing_set")):
        batch_x, batch_y = batch
        predicted = model(batch_x)
        testing_batch_rpn_accuracy, testing_batch_cls_accuracy = calc_accuracy(batch_y, predicted)
        testing_rpn_accuracy_list.append(testing_batch_rpn_accuracy)
        testing_cls_accuracy_list.append(testing_batch_cls_accuracy)
    testing_rpn_accuracy = sum(testing_rpn_accuracy_list) / len(testing_rpn_accuracy_list)
    testing_cls_accuracy = sum(testing_cls_accuracy_list) / len(testing_cls_accuracy_list)
    print(f"testing rpn accuracy: {testing_rpn_accuracy}, cls accuracy: {testing_cls_accuracy}")
    print_log(f"testing rpn accuracy: {testing_rpn_accuracy}, cls accuracy: {testing_cls_accuracy}")

    # 显示训练集和验证集的正确率变化
    pyplot.plot(training_rpn_accuracy_history, label="training_rpn_accuracy")
    pyplot.plot(training_cls_accuracy_history, label="training_cls_accuracy")
    pyplot.plot(validating_rpn_accuracy_history, label="validating_rpn_accuracy")
    pyplot.plot(validating_cls_accuracy_history, label="validating_cls_accuracy")
    pyplot.ylim(0, 1)
    pyplot.legend()
    pyplot.savefig(os.path.join(OUT_DIR, "train_val_acc.png"))
    pyplot.show()

def test_model():
    """test the classification accuracy"""
    model = MyModel().to(device)
    model.load_state_dict(load_tensor(os.path.join(OUT_DIR, "model.pt")))
    model.eval()

    n_classes = len(CLASSES) - 1   # others are omitted
    confmat = np.zeros((n_classes, n_classes), dtype=int)

    test_images_file = open(os.path.join(OUT_DIR, "test_images.txt"))
    for image_path in test_images_file.readlines():
        image_path = image_path.strip()
        image_name = os.path.basename(image_path)
        real_cls = image_name.split('.')[0]
        predicted_cls = -1

        if not image_path:
            continue
        with Image.open(image_path) as img_original: # 加载原始图片
            img = resize_image(img_original) # 缩放图片
            tensor_in = image_to_tensor(img)
            # 预测输出
        cls_result = model(tensor_in.unsqueeze(0).to(device))[-2][0]
        if cls_result == None:
            print(f"Cannot detect face in image {image_name}, skip.")
            continue
        # 合并重叠的结果区域, 结果是 [ [标签列表, 合并后的区域], ... ]
        final_result = []
        for label, box in cls_result:
            for index in range(len(final_result)):
                exists_labels, exists_box = final_result[index]
                if calc_iou(box, exists_box) > IOU_MERGE_THRESHOLD:
                    exists_labels.append(label)
                    final_result[index] = (exists_labels, merge_box(box, exists_box))
                    break
            else:
                final_result.append(([label], box))
        # 合并标签 (重叠区域的标签中数量最多的分类为最终分类)
        for index in range(len(final_result)):
            labels, box = final_result[index]
            final_label = Counter(labels).most_common(1)[0][0]
            final_result[index] = (final_label, box)

        for label, box in final_result:
            predicted_cls = CLASSES[label]
            if predicted_cls == real_cls:
                break   
        print(real_cls, predicted_cls)
        confmat[int(real_cls)-1, int(predicted_cls)-1] += 1
    test_images_file.close()

    test_metrics_file = os.path.join(OUT_DIR, "test_metrics.rpt")
    with open(test_metrics_file, 'a+') as f:
        print("Confusion Matrix:", file=f)
        print(confmat, file=f)
        print("Confusion Matrix:")
        print(confmat)

        test_num = np.sum(confmat)
        correct_num = np.trace(confmat)
        print(f"#Total Tests: {test_num}, #Correct Tests: {correct_num}", file=f)
        print(f"ACC on test set: {correct_num/test_num}", file=f)
        print(f"#Total Tests: {test_num}, #Correct Tests: {correct_num}")
        print(f"ACC on test set: {correct_num/test_num}")


def eval_model():
    """使用训练好的模型"""
    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = MyModel().to(device)
    model.load_state_dict(load_tensor(os.path.join(OUT_DIR, "model.pt")))
    model.eval()

    # 询问图片路径，并显示所有可能是人脸的区域
    while True:
        try:
            image_path = input("Image path: ")
            if not image_path:
                continue
            # 构建输入
            with Image.open(image_path) as img_original: # 加载原始图片
                sw, sh = img_original.size # 原始图片大小
                img = resize_image(img_original) # 缩放图片
                img_output = img_original.copy() # 复制图片，用于后面添加标记
                tensor_in = image_to_tensor(img)
            # 预测输出
            cls_result = model(tensor_in.unsqueeze(0).to(device))[-2][0]
            
            # Feature vector shape: (32, 128) -> features of top 32 possible regions
            face_feature = model(tensor_in.unsqueeze(0).to(device))[-1][0]
            face_feature = torch.mean(face_feature, dim=0)
            face_feature /= torch.sum(face_feature)
            # print(face_feature.shape, face_feature, torch.sum(face_feature))
            
            # 合并重叠的结果区域, 结果是 [ [标签列表, 合并后的区域], ... ]
            final_result = []
            for label, box in cls_result:
                for index in range(len(final_result)):
                    exists_labels, exists_box = final_result[index]
                    if calc_iou(box, exists_box) > IOU_MERGE_THRESHOLD:
                        exists_labels.append(label)
                        final_result[index] = (exists_labels, merge_box(box, exists_box))
                        break
                else:
                    final_result.append(([label], box))
            # 合并标签 (重叠区域的标签中数量最多的分类为最终分类)
            for index in range(len(final_result)):
                labels, box = final_result[index]
                final_label = Counter(labels).most_common(1)[0][0]
                final_result[index] = (final_label, box)
            # 标记在图片上
            draw = ImageDraw.Draw(img_output)
            for label, box in final_result:
                x, y, w, h = map_box_to_original_image(box, sw, sh)
                draw.rectangle((x, y, x+w, y+h), outline="#FF0000")
                draw.text((x, y-10), CLASSES[label], fill="#FF0000")
                print((x, y, w, h), CLASSES[label])

            outimg_dir = os.path.join(OUT_DIR, "eval")
            if not os.path.isdir(outimg_dir):
                os.makedirs(outimg_dir)
            img_name = os.path.basename(image_path)
            img_name = img_name.split('.')[0] + "." + img_name.split('.')[1] + "(cls).png"
            img_output.save(os.path.join(outimg_dir, img_name))
            print(f"saved to {os.path.join(outimg_dir, img_name)}")
            print()
        except Exception as e:
            print("error:", e)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(f"Please run: {sys.argv[0]} prepare|train|eval|test")
        exit()

    # 给随机数生成器分配一个初始值，使得每次运行都可以生成相同的随机数
    # 这是为了让过程可重现，你也可以选择不这样做
    # random.seed(0)
    # torch.random.manual_seed(0)

    MyModel.Anchors = MyModel._generate_anchors(8)

    # 根据命令行参数选择操作
    operation = sys.argv[1]
    if operation == "prepare":
        prepare()
    elif operation == "train":
        train()
    elif operation == "eval":
        eval_model()
    elif operation == "test":
        test_model()
    else:
        raise ValueError(f"Unsupported operation: {operation}")

if __name__ == "__main__":
    main()
