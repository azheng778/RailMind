# coding:utf-8
import os

import cv2
import numpy as np
import onnxruntime as ort

# 定义类别名称映射
class_names = {
    0: "board",
    1: "handle"
}

def xywh2xyxy(x):

    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制 (Non-Maximum Suppression, NMS)

    参数:
        boxes (numpy.ndarray[N, 4]): 每个框的坐标，格式为 [x1, y1, x2, y2]
        scores (numpy.ndarray[N]): 每个框的置信度分数
        iou_threshold (float): 用于决定是否抑制的 IoU 阈值

    返回:
        keep (numpy.ndarray): 保留的框的索引
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    # 根据分数排序，获取索引
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        # 选择当前分数最高的框
        idx = order[0]
        keep.append(idx)

        # 计算当前框与剩余框的 IoU
        xx1 = np.maximum(boxes[idx, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[idx, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[idx, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[idx, 3], boxes[order[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area0 = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
        area1 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area0 + area1 - inter)

        # 找到 IoU 大于阈值的框，并将它们从排序列表中移除
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return np.array(keep, dtype=np.int32)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        rotated=False,
):
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    prediction = prediction.transpose(0, 2, 1)

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = np.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

        # 获取每行的最大值和最大值的索引
        conf = np.max(cls, axis=1)[:, np.newaxis]  # 最大值
        j = np.argmax(cls, axis=1)[:, np.newaxis]  # 最大值的索引

        # 将 j 转换为浮点类型
        j = j.astype(np.float32)

        # 拼接 box, conf, j, mask
        x = np.concatenate((box, conf, j, mask), axis=1)

        # 筛选出置信度大于阈值的行
        x = x[x[:, 4] > conf_thres]  # 假设 conf 在第 5 列（索引为 4）

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores

        boxes = x[:, :4] + c  # boxes (offset by class)
        i = nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections

        output[xi] = x[i]

    return output


def find_intersection_and_horizontal_slope(preds, class_id=0):
    """
    根据预测结果，找到指定类别（class_id）的第一与第二关键点的直线与第三与第四关键点的直线的交点，
    并计算第1与第4关键点拟合直线的斜率。

    :param preds: 预测结果，格式为 [N, 6 + num_keypoints * 3]，其中 N 是检测框数量。
                  每个检测框的格式为 [x1, y1, x2, y2, conf, cls, kx1, ky1, kv1, kx2, ky2, kv2, ...]
    :param class_id: 目标类别 ID，默认为 0。
    :return: 交点坐标 (x, y) 和水平斜率 m_horizontal，如果无法计算交点或斜率则返回 None。
    """
    # 遍历预测结果，找到类别为 class_id 的检测结果
    for det in preds:
        if int(det[5]) == class_id:  # 检查类别是否匹配
            keypoints = det[6:].reshape(-1, 3)  # 提取关键点坐标
            if len(keypoints) < 4:  # 确保至少有4个关键点
                print("检测结果中关键点数量不足4个，无法计算交点。")
                return None

            # 提取第一、第二、第三和第四关键点的坐标
            x1, y1 = keypoints[0, :2]
            x2, y2 = keypoints[1, :2]
            x3, y3 = keypoints[2, :2]
            x4, y4 = keypoints[3, :2]

            center_2_3 = [(x2+x3)/2, (y2+y3)/2]

            # 计算第1与第4关键点的直线斜率
            if x4 == x1:  # 防止除以零
                print("第1与第4关键点的直线斜率无穷大，无法计算水平斜率。")
                return None
            m_horizontal = (y4 - y1) / (x4 - x1)

            # 计算第1与第2关键点的直线斜率
            if x2 == x1:  # 防止除以零
                print("第1与第2关键点的直线斜率无穷大，无法计算交点。")
                return None
            m1 = (y2 - y1) / (x2 - x1)

            # 5 度的正切近似值作为透视修正系数
            tan_5 = -0.06
            # 运用两角和的正切公式来计算旋转后的斜率
            if m1 * tan_5 == 1:
                # 分母为零的特殊情况，旋转后斜率为无穷大
                print("修正后斜率无穷大，取消修正")
            else:
                m1 = (m1 + tan_5) / (1 - m1 * tan_5)

            # 计算第3与第4关键点的直线斜率
            if x4 == x3:  # 防止除以零
                print("第3与第4关键点的直线斜率无穷大，无法计算交点。")
                return None
            m2 = (y4 - y3) / (x4 - x3)

            # 计算两条直线的截距
            b1 = y1 - m1 * x1
            b2 = y3 - m2 * x3

            # 检查是否平行（斜率相同）
            if m1 == m2:
                print("两条直线平行，没有交点。")
                return None

            # 解方程组找到交点
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1

            return (x, y), m_horizontal, center_2_3

    print(f"未找到类别为 {class_id} 的检测结果。")
    return None

def find_vertical_slope_and_angle(preds, intersection, horizontal_slope, center_2_3, output_frame=None, class_id=1,
                                  image_size=(1920, 1080)):
    results = []
    # image = output_frame

    for det in preds:
        if int(det[5]) == class_id:
            keypoints = det[6:].reshape(-1, 3)
            if len(keypoints) < 6:
                continue

            # 提取关键点坐标（第五和第六关键点）
            x5, y5 = keypoints[4, :2]
            x6, y6 = keypoints[5, :2]

            try:
                # 计算新的垂直轴向量（交点到第5关键点）
                dx_vertical = x5 - intersection[0]
                dy_vertical = y5 - intersection[1]

                # 计算关键点5到关键点6的向量
                dx = x6 - x5
                dy = y6 - y5

                # 计算与垂直轴（新坐标系Y轴）的夹角
                angle_rad = np.arctan2(dx * dy_vertical - dy * dx_vertical, dx * dx_vertical + dy * dy_vertical)
                angle_deg = np.degrees(angle_rad)

                # 计算理论垂直斜率
                if abs(dx_vertical) < 1e-6:  # 垂直轴为90度
                    m_vertical = np.inf
                else:
                    m_vertical = dy_vertical / dx_vertical

                # 计算horizontal_slope与垂直轴的角度
                if m_vertical == np.inf:
                    angle_with_horizontal = np.degrees(np.pi / 2 - np.arctan(horizontal_slope)) if horizontal_slope != np.inf else 0
                else:
                    angle_with_horizontal = np.degrees(np.arctan(
                        (m_vertical - horizontal_slope) / (1 + m_vertical * horizontal_slope)))

                if angle_with_horizontal > 0:
                    angle_with_horizontal = angle_with_horizontal - 180

                degree = angle_deg * 90 / np.abs(angle_with_horizontal)
                if degree > 0:
                    degree = 0
                results.append((m_vertical, np.abs(degree)))

                # # 绘制线条
                # cv2.line(image, (int(x5), int(y5)), (int(x6), int(y6)), (0, 255, 0), 2)
                #
                # # 绘制理论垂直线
                # if m_vertical == np.inf:
                #     cv2.line(image, (int(x5), 0), (int(x5), image_size[1]), (0, 0, 255), 2)
                # else:
                #     y1 = int(m_vertical * (0 - x5) + y5)
                #     y2 = int(m_vertical * (image_size[0] - x5) + y5)
                #     cv2.line(image, (0, y1), (image_size[0], y2), (0, 0, 255), 2)
                #
                # # 绘制斜率为horizontal_slope的水平轴
                # if abs(horizontal_slope) > 1e6:  # 接近垂直
                #     cv2.line(image, (int(x5), 0), (int(x5), image_size[1]), (255, 0, 0), 2)
                # else:
                #     # 计算直线在图像左右边界的y坐标
                #     y_left = int(horizontal_slope * (0 - x5) + y5)
                #     y_right = int(horizontal_slope * (image_size[0] - x5) + y5)
                #     cv2.line(image, (0, y_left), (image_size[0], y_right), (255, 0, 0), 2)

            except Exception as e:
                print(f"计算错误: {str(e)}")
                continue

    return results if results else None


def warp_and_crop_roi(image, preds, class_id=0, output_size=500, margin=100):
    """
    改进版ROI提取函数，适配YOLO格式关键点排列
    参数说明：
        image: 原始图像 (H,W,C)
        preds: 预测结果 [N,6+num_kpts*3]
        class_id: 目标类别ID
        output_size: 输出正方形尺寸
    返回：
        List: 透视变换后的ROI图像列表
    """
    warped_images = []

    for det in preds:
        if int(det[5]) != class_id:
            continue

        # 从第7位开始取前4个关键点（跳过：x1,y1,x2,y2,conf,class_id）
        kpts = det[6:6 + 4 * 3].reshape(4, 3)  # 转换为(4,3)数组
        valid_kpts = kpts

        if len(valid_kpts) < 4:
            print(f"警告：有效关键点不足4个（{len(valid_kpts)}个有效）")
            continue

        try:
            # 提取坐标并转换顺序（模型输出->目标顺序）
            src_points = reorder_kpoints(valid_kpts[:, :2].astype(np.float32))

            # 定义目标坐标
            dst = np.array([
                [0, 0],  # 左上
                [output_size, 0],  # 右上
                [output_size, output_size],  # 右下
                [0, output_size]  # 左下
            ], dtype=np.float32)

            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(src_points, dst)

            # 执行双三次插值变换
            warped = cv2.warpPerspective(image, M, (output_size, output_size),
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REFLECT_101)
            cropped_image = warped[margin:-margin, margin:-margin]
            warped_images.append(cropped_image)

        except Exception as e:
            print(f"几何变换失败：{str(e)}")
            continue

    return warped_images


def reorder_kpoints(pts):
    """
    将输入点从[左上, 左下, 右下, 右上]转换为标准透视变换顺序
    参数：
        pts: (4,2) 输入点数组，格式为左上、左下、右下、右上
    返回：
        (4,2) 按标准透视顺序排列的数组
    """
    # 验证输入形状
    assert pts.shape == (4, 2), "输入点格式必须为(4,2)"

    # 直接按指定顺序排列（假设输入顺序正确）
    return np.array([
        pts[0],  # 左上
        pts[3],  # 右上
        pts[2],  # 右下
        pts[1],  # 左下
    ], dtype=np.float32)

def merge_lines(lines: np.ndarray, min_rho=100, min_theta=np.pi / 9):
    '''
        函数根据HoughLine函数返回参数进行直线合并，且自动剔除相近直线
        函数将把theta差值小于一定范围，并且在此基础上，rho差值小于一定范围的直线剔除
    '''
    merged_lines = []
    if len(lines) != 0:
        rho, theta = lines[0, 0]
        merged_lines.append([[rho, theta]])
    for rho, theta in lines[1:, 0]:
        is_merged = False
        for past_line in merged_lines:
            past_rho, past_theta = past_line[0]
            theta_error = abs(past_theta - theta)
            rho_error = abs(past_rho - rho)
            if theta_error <= min_theta and rho_error <= min_rho:
                is_merged = True
                break
        if not is_merged:
            merged_lines.append([[rho, theta]])

    merged_lines = np.array(merged_lines)
    return merged_lines


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(1024, 1024), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

class YOLOv8Keypoint:
    """YOLOv8关键点检测模型类，用于处理推理和可视化操作。"""
    def __init__(self,
                 det_ckpt_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.onnx"),
                 confidence_thres=0.8,
                 nms_thres=0.5):
        self.onnx_model = det_ckpt_path
        self.confidence_thres = confidence_thres
        self.nms_thres = nms_thres

        # 初始化ONNX会话
        # print("====== load model =====")
        self.initialize_session(self.onnx_model)
        # print("====== finsh load =====")

    def pre_transform(self, im):
        letterbox = LetterBox()
        return [letterbox(image=x) for x in im]

    def preprocess(self, im):
        if im.ndim == 3:  # 单张图像 (h, w, c)
            im = im[None, ...]  # 添加一个 batch 维度，变为 (1, h, w, c)
        elif im.ndim != 4:
            raise ValueError(f"Unexpected input shape: {im.shape}. Expected 3D or 4D array.")

        im = np.stack(self.pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous

        im = im.astype(np.float32)  # uint8 to fp16/32

        global_brightness_mean = np.mean(im)
        bright_threshold = global_brightness_mean * 1.2
        dark_threshold = global_brightness_mean * 0.8

        im = np.where(im > bright_threshold, im*0.8, np.where(im<dark_threshold, im*1.2, im))

        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def initialize_session(self, onnx_model):
        """
        使用 onnxruntime 初始化 ONNX 模型。
        """
        # 指定使用 GPU 进行推理
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # 首先尝试使用 GPU，如果不可用则回退到 CPU
        self.session = ort.InferenceSession(onnx_model, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取输入尺寸
        self.input_width = self.session.get_inputs()[0].shape[2]
        self.input_height = self.session.get_inputs()[0].shape[3]

    def process_image(self, image):
        """
        修改后的process_image函数，直接接收图像变量作为输入
        """
        original_shape = image.shape[:2]  # 保存原始图像的尺寸 (height, width)

        # 预处理图像
        img_data = self.preprocess(image)

        # 运行推理
        outputs = self.session.run([self.output_name], {self.input_name: img_data})[0]  # 获取推理结果
        preds = non_max_suppression(
            outputs,
            self.confidence_thres,
            self.nms_thres,
            agnostic=False,
            max_det=300,
            nc=2,
        )
        preds = preds[0]

        # 提取类别为0的预测结果
        class_0_preds = preds[preds[:, 5] == 0]
        # 找到置信度最高的0类别预测结果
        if len(class_0_preds) > 0:
            highest_conf_0 = class_0_preds[np.argmax(class_0_preds[:, 4])]
        else:
            highest_conf_0 = None

        # 提取类别为1的预测结果
        class_1_preds = preds[preds[:, 5] == 1]
        # 找到前两个置信度最高的1类别预测结果
        if len(class_1_preds) > 0:
            top_conf_1 = class_1_preds[np.argsort(class_1_preds[:, 4])[-2:]]
        else:
            top_conf_1 = None

        # 将结果组合成一个二维数组
        if highest_conf_0 is not None:
            final_result = [highest_conf_0]
        else:
            final_result = []

        if top_conf_1 is not None:
            final_result.extend(top_conf_1)

        # 转换为二维数组
        final_result = np.array(final_result)

        preds = [final_result]

        # 如果有检测结果
        if len(preds[0]) > 0:
            # 缩放检测框和关键点的坐标到原始图像尺寸
            preds[0] = self.rescale_preds(preds[0], (self.input_height, self.input_width), original_shape)

            for det_index, det in enumerate(preds[0]):
                # 绘制检测框
                x1, y1, x2, y2, conf, cls = det[:6]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 根据 cls 获取对应的类别名称
                # class_name = class_names.get(cls, "Unknown")

                # 在图像上添加文本
                # cv2.putText(image, f"{class_name} Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7,
                #             (0, 255, 0), 2)

                # 绘制关键点
                keypoints = det[6:].reshape(-1, 3)  # 假设每个关键点有 (x, y, visibility)
                for i, kp in enumerate(keypoints):
                    x, y, v = int(kp[0]), int(kp[1]), kp[2]
                    # 确保关键点在检测框内
                    x = max(x1, min(x2, x))
                    y = max(y1, min(y2, y))
                    # 更新关键点坐标
                    # keypoints[i][0] = x
                    # keypoints[i][1] = y
                    # 更新preds[0]
                    preds[0][det_index, 6 + i * 3] = x
                    preds[0][det_index, 6 + i * 3 + 1] = y
                    # if v > 0.2:  # 只绘制置信度大于 0.2 的关键点
                    #     cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 绘制关键点

        return preds[0]

    def rescale_preds(self, preds, current_shape, original_shape):
        """
        将预测结果中的检测框和关键点坐标从当前尺寸缩放回原始尺寸。
        :param preds: 预测结果 (N, 6 + num_keypoints * 3)，格式为 [x1, y1, x2, y2, conf, cls, kx1, ky1, kv1, ...]
        :param current_shape: 当前图像尺寸 (height, width)
        :param original_shape: 原始图像尺寸 (height, width)
        :return: 缩放后的预测结果
        """
        # 计算缩放比例
        gain = min(current_shape[0] / original_shape[0], current_shape[1] / original_shape[1])

        # 计算填充区域
        pad = (current_shape[1] - original_shape[1] * gain) / 2, (current_shape[0] - original_shape[0] * gain) / 2

        # 将 pad 转换为 NumPy 数组
        pad_array = np.array([pad[0], pad[1], pad[0], pad[1]])

        # 减去填充区域
        preds[:, :4] -= pad_array
        preds[:, :4] /= gain  # 缩放回原始尺寸

        # 限制检测框坐标在原始图像范围内
        preds[:, [0, 2]] = np.clip(preds[:, [0, 2]], 0, original_shape[1])  # x1, x2
        preds[:, [1, 3]] = np.clip(preds[:, [1, 3]], 0, original_shape[0])  # y1, y2

        # 缩放关键点坐标
        num_keypoints = (preds.shape[1] - 6) // 3  # 每个关键点有 3 个值 (x, y, v)
        for i in range(num_keypoints):
            kx, ky = 6 + i * 3, 7 + i * 3  # 关键点坐标索引
            preds[:, kx] -= pad[0]  # x 坐标减去填充
            preds[:, ky] -= pad[1]  # y 坐标减去填充
            preds[:, kx] /= gain  # 缩放 x 坐标
            preds[:, ky] /= gain  # 缩放 y 坐标
            preds[:, kx] = np.clip(preds[:, kx], 0, original_shape[1])  # x 坐标限制在原始宽度范围内
            preds[:, ky] = np.clip(preds[:, ky], 0, original_shape[0])  # y 坐标限制在原始高度范围内

        return preds

    # def Cross_Val(self, image, horizontal_slope):
    #     """
    #     对传入的图像进行检测，识别出其中的两条明显边缘线。
    #
    #     参数:
    #         image: 输入图像，应为 NumPy 数组格式。
    #
    #     返回:
    #         lines: 检测到的两条边缘线的参数列表，每条线由 (rho, theta) 表示。
    #         output_image: 绘制了检测到的边缘线的图像。
    #     """
    #     # 转换为灰度图像
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
    #
    #     # 使用 Canny 边缘检测
    #     edges = cv2.Canny(gray, 50, 150)
    #
    #     # 使用霍夫变换检测直线
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=130)
    #
    #     if lines is not None:
    #         lines = merge_lines(lines)
    #
    #     # 过滤与 horizontal_slope 相近的直线
    #     if lines is not None:
    #         filtered_lines = []
    #         for line in lines:
    #             rho, theta = line[0]
    #             # 计算斜率
    #             if theta != 0:
    #                 slope = -np.cos(theta) / np.sin(theta)
    #                 # 判断斜率是否与 horizontal_slope 相近
    #                 if abs(slope - horizontal_slope) > 0.2:  # 可以调整这个阈值
    #                     filtered_lines.append(line)
    #         lines = np.array(filtered_lines)
    #         print(lines)
    #
    #     if lines is not None and len(lines) == 2:
    #         return lines
    #     else:
    #         return None

    # threshold:对板子边缘检测像素的阈值
    def predict(self, image, camera_id=0, threshold = 3900, visualize = False):
        #定义水平斜率
        cv_horizontal_slope = None
        # 初始化返回信息列表
        return_info = []
        # return_info.append(camera_id)

        # 推理图片
        preds = self.process_image(np.copy(image))

        if len(preds) == 0:
            return_info.append(['T', '未检测到目标', 0, 0, 0, 0])
            return return_info

        # 提取类别为1的预测结果
        class_1_preds = preds[preds[:, 5] == 1]

        # 按照框的中心横坐标（第 0 列）进行排序,使得靠左的框位于前面
        class_1_preds = class_1_preds[class_1_preds[:, 0].argsort()]

        # 提取类别为0的预测结果
        class_0_preds = preds[preds[:, 5] == 0]

        #当具有两个把手一个大板区域
        if len(class_1_preds) == 2 and len(class_0_preds) == 1:
            # 提取前两个检测结果
            detection_1 = class_1_preds[0]
            detection_2 = class_1_preds[1]

            # 假设前四个值是边界框的坐标 (x1, y1, x2, y2)
            # 计算中心点
            center_1_x = (detection_1[0] + detection_1[2]) / 2
            center_1_y = (detection_1[1] + detection_1[3]) / 2
            center_2_x = (detection_2[0] + detection_2[2]) / 2
            center_2_y = (detection_2[1] + detection_2[3]) / 2

            # 计算水平斜率
            if center_2_x - center_1_x != 0:
                cv_horizontal_slope = (center_2_y - center_1_y) / (center_2_x - center_1_x)

            # # 提取框选区域的坐标
            # x_min, y_min, x_max, y_max = class_0_preds[0][:4].astype(int)
            # # 裁剪框选区域
            # cropped_image = image[y_min:y_max, x_min:x_max]
            # cv_vertical_lines = self.Cross_Val(cropped_image, cv_horizontal_slope)
        else:
            return_info.append(['T', '信息不全进行屏蔽', 0, 0, 0, 0])
            return return_info



        # 调用函数计算交点和水平斜率
        result = find_intersection_and_horizontal_slope(preds, class_id=0)


        if result:
            intersection, horizontal_slope, center_2_3 = result

            #当cv2识别到水平斜率时使用cv2的水平斜率
            if cv_horizontal_slope is not None:
                horizontal_slope = (cv_horizontal_slope + horizontal_slope) / 2

            # info.append(f"交点坐标: {intersection}")
            # info.append(f"水平斜率: {horizontal_slope}")

            # 调用函数计算垂直斜率和偏差角
            vertical_results = find_vertical_slope_and_angle(class_1_preds, intersection, horizontal_slope, center_2_3, output_frame=None, class_id=1)
            if vertical_results:
                for i, (vertical_slope, angle_deg) in enumerate(vertical_results):
                    bbox = class_1_preds[i]

                    # 增加返回信息
                    # 计算警告等级
                    if angle_deg <= 50:
                        level = 0
                        # text_color = (0, 255, 0)
                        # return_info.append(['T', f'把手{i}正常', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])
                    elif angle_deg <= 75:
                        # level = 1
                        # text_color = (255, 255, 0)
                        return_info.append(['F', f'把手{i}警告', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])
                    else:
                        # level = 2
                        # text_color = (0, 0, 255)
                        return_info.append(['F', f'把手{i}异常', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])

                    # 构造新的垂直信息
                    # vertical_info = [0, angle_deg, level]
                    # 在图像左上角显示偏差角
                    # cv2.putText(output_frame, f"handle: {angle_deg:.2f} deg", (10, 100 + i * 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    #             text_color, 3)

            # 在绘制完成后添加：
            # 执行ROI变换（使用500x500输出，裁剪100像素边缘）
            roi_images = warp_and_crop_roi(image, preds, output_size=500, margin=110)

            # 将彩色图像转换为灰度图像
            gray_image = cv2.cvtColor(roi_images[0], cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # 使用Canny边缘检测
            edges = cv2.Canny(blurred_image, 80, 180)

            # 使用形态学操作去除孤立的边缘点
            kernel = np.ones((3, 3), np.uint8)
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 统计边缘像素的数量
            edge_count = np.sum(edges > 0)

            bbox = class_0_preds[0]
            board_error = False
            if edge_count > threshold:
                # 盖子异常
                board_error = True
                # cv2.putText(output_frame, "Board: Abnormal", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                # 盖子正常
                board_error = False
                # cv2.putText(output_frame, "Board: Normal", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


            if len(class_1_preds) == 2 and board_error == False:
                return_info.append(
                    ["T", '检查门正常', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])
            else:
                return_info.append(
                    ["F", '检查门异常', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])

            # 显示处理结果
            # for idx, roi in enumerate(roi_images):
            #     cv2.imshow(f'Transformed ROI {idx + 1}', output_frame)
            #     cv2.waitKey(1000)
        else:
            bbox = class_0_preds[0]
            return_info.append(
                ["T", '无法计算角度', int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])])

        # 直接返回 return_info 列表
        return return_info



# ---- RailMind 补丁：移除模块级实例化，模型按需加载 ----
