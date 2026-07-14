"""
集装箱表面缺陷检测模型训练脚本
==============================================
选题D：集装箱航运网络优化 - 集装箱表面缺陷检测子任务

数据集：3713张集装箱表面图像（3300训练 / 413测试）
类别：Dent(凹陷/0)、Hole(破洞/1)、Rusty(锈蚀/2)
标注格式：YOLO (class x_center y_center width height)
"""

import sys
from pathlib import Path
import yaml

# ============================================================
# 0. 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent        # D-Answer/
ROOT_DIR = BASE_DIR.parent                        # module/ (仓库根目录)
DATASET_DIR = ROOT_DIR / "选题D" / "数据集3713"
TRAIN_IMG = DATASET_DIR / "images" / "train"
TRAIN_LBL = DATASET_DIR / "labels" / "train"
TEST_IMG  = DATASET_DIR / "images" / "test"
TEST_LBL  = DATASET_DIR / "labels" / "test"
OUTPUT_DIR = BASE_DIR                              # D-Answer/ (输出也放这里)
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Dent", "Hole", "Rusty"]
CLASS_NAMES_ZH = {"Dent": "凹陷", "Hole": "破洞", "Rusty": "锈蚀"}

# ============================================================
# 1. 数据集统计
# ============================================================
def dataset_statistics():
    """统计数据集基本信息"""
    print("=" * 60)
    print(">> 数据集统计")
    print("=" * 60)

    for split, img_dir, lbl_dir in [("train", TRAIN_IMG, TRAIN_LBL),
                                     ("test",  TEST_IMG,  TEST_LBL)]:
        imgs = sorted(img_dir.glob("*.jpg"))
        lbls = sorted(lbl_dir.glob("*.txt"))
        assert len(imgs) == len(lbls), f"{split}: 图片与标签数量不匹配!"

        # 统计各类别实例数
        class_counts = {0: 0, 1: 0, 2: 0}
        total_bboxes = 0
        for lbl_file in lbls:
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        total_bboxes += 1

        print(f"\n{split} 集:")
        print(f"  图片数: {len(imgs)}")
        print(f"  总标注框: {total_bboxes}")
        for cls_id, count in class_counts.items():
            zh = CLASS_NAMES_ZH.get(CLASSES[cls_id], CLASSES[cls_id])
            print(f"  {CLASSES[cls_id]}({zh}): {count} ({count/total_bboxes*100:.1f}%)")

    return True


# ============================================================
# 2. 生成 data.yaml
# ============================================================
def generate_data_yaml():
    """生成Ultralytics所需的data.yaml配置文件"""
    data_yaml_path = OUTPUT_DIR / "data.yaml"

    # 使用绝对路径避免路径问题
    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "nc": 3,
        "names": CLASSES,
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[OK] data.yaml has been generated: {data_yaml_path}")
    return str(data_yaml_path)


# ============================================================
# 3. YOLO 训练
# ============================================================
def train_yolo(data_yaml: str, model_size: str = "m", epochs: int = 200):
    """
    使用Ultralytics YOLO训练模型

    Parameters
    ----------
    data_yaml : str
        数据集配置文件路径
    model_size : str
        模型规模: n/s/m/l/x (nano/small/medium/large/xlarge)
    epochs : int
        训练轮数
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Please install ultralytics: pip install ultralytics")
        sys.exit(1)

    model_name = f"yolov8{model_size}.pt"
    print(f"\n>> Start training YOLOv8{model_size}, epochs={epochs}")
    print(f"   数据集配置: {data_yaml}")

    # 加载预训练模型
    model = YOLO(model_name)

    # 训练
    results = model.train(
        data=data_yaml,
        # ---- 核心参数（建议根据需求调整） ----
        imgsz=1280,        # 【关键】输入分辨率，1280能捕获细小缺陷（如锈蚀纹理）
                           #  640 → 速度快、省显存，适合快速验证
                           #  1280 → 细节更丰富，对Rusty/Hole等小目标提升明显
        batch=8,           # 【关键】批次大小，1280分辨率下8GB显存建议4~8
                           #  显存不足报OOM时减小，如4
        model_size=model_size,  # 【关键】模型规模，m比s大3倍参数，拟合能力更强
                                #  n(3M) < s(11M) < m(26M) < l(46M) < x(87M)
        device="cuda",
        workers=4,
        # ---- 训练控制 ----
        epochs=epochs,         # 【关键】训练轮数，200轮给模型足够时间收敛
        patience=30,           # 早停：30轮mAP不涨则停止（轮数多了相应放宽）
        save=True,
        save_period=10,
        project=str(OUTPUT_DIR),
        name=f"yolov8{model_size}_defect",
        exist_ok=True,
        pretrained=True,
        # ---- 优化器 ----
        optimizer="auto",      # auto自动选AdamW（适合大数据集）或SGD
        lr0=0.01,              # 初始学习率
        lrf=0.01,              # 最终学习率 = lr0 * lrf = 0.0001
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,           # 余弦退火：平滑降低学习率，比阶梯下降更稳定
        # ---- 数据增强 ----
        augment=True,
        close_mosaic=15,       # 最后15轮关闭mosaic，让模型适应真实分辨率分布
        amp=False,             # 混合精度，关闭避免兼容问题（有GPU可尝试改为True加速）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.1,
        fliplr=0.5,
        # ---- 小目标优化 ----
        mosaic=1.0,            # mosaic拼接增强，小目标检测必备
        mixup=0.1,             # mixup混合增强，缓解类别不平衡
        copy_paste=0.1,        # copy-paste增强，增加稀少类别样本
    )

    print("\n>> Training completed!")
    return results


# ============================================================
# 4. 模型评估
# ============================================================
def evaluate_model(model_path: str, data_yaml: str, imgsz: int = 1280):
    """评估训练好的模型"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Please install ultralytics: pip install ultralytics")
        sys.exit(1)

    print(f"\n>> Model evaluation: {model_path}")

    model = YOLO(model_path)

    # 验证集评估（imgsz需与训练时一致）
    metrics = model.val(data=data_yaml, split="test", imgsz=imgsz)

    # 输出关键指标
    print("\n" + "=" * 60)
    print(">> Evaluation Results")
    print("=" * 60)
    print(f"  mAP@50:       {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:    {metrics.box.map:.4f}")
    if hasattr(metrics.box, "mp"):
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")

    # 各类别AP
    if hasattr(metrics.box, "ap_class_index"):
        print("\n  各类别 AP@50-95:")
        for i, ap in zip(range(len(metrics.box.ap)), metrics.box.ap):
            cls_name = CLASSES[int(i)]
            zh = CLASS_NAMES_ZH.get(cls_name, cls_name)
            print(f"    {cls_name}({zh}): {ap:.4f}")

    return metrics


# ============================================================
# 5. 推理预测（单图/文件夹）
# ============================================================
def predict(model_path: str, source: str, conf: float = 0.25, imgsz: int = 1280):
    """
    使用训练好的模型进行预测

    Parameters
    ----------
    model_path : str
        模型权重路径
    source : str
        预测源（图片路径或文件夹）
    conf : float
        置信度阈值
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Please install ultralytics: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        save=True,
        save_txt=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True,
    )
    print(f"\n>> Prediction done! Results saved to {OUTPUT_DIR / 'predictions'}")
    return results


# ============================================================
# 6. 主流程
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="集装箱缺陷检测训练")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["stats", "train", "eval", "predict", "all"],
                        help="运行模式")
    parser.add_argument("--model", type=str, default="m",
                        choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 模型大小 (默认: m)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数 (默认: 200)")
    parser.add_argument("--weights", type=str, default=None,
                        help="用于eval/predict的模型权重路径")
    parser.add_argument("--source", type=str, default=None,
                        help="predict模式下的图片/文件夹路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="预测置信度阈值 (默认: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="输入分辨率 (默认: 1280)")
    parser.add_argument("--batch", type=int, default=8,
                        help="批次大小 (默认: 8, OOM时改小)")
    args = parser.parse_args()

    # ---- 统计 ----
    if args.mode in ("stats", "all"):
        dataset_statistics()

    # ---- 生成配置 ----
    data_yaml = generate_data_yaml()

    # ---- 训练 ----
    model_path = args.weights
    if args.mode in ("train", "all"):
        results = train_yolo(data_yaml, model_size=args.model, epochs=args.epochs)
        # 训练结束后自动得到最佳权重路径
        model_path = str(OUTPUT_DIR / f"yolov8{args.model}_defect" / "weights" / "best.pt")
        print(f"\n>> Best weights: {model_path}")

    # ---- 评估 ----
    if args.mode in ("eval", "all"):
        if model_path is None:
            # 尝试默认路径
            model_path = str(OUTPUT_DIR / f"yolov8{args.model}_defect" / "weights" / "best.pt")
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
        else:
            evaluate_model(model_path, data_yaml, imgsz=args.imgsz)

    # ---- 预测 ----
    if args.mode == "predict":
        if model_path is None:
            model_path = str(OUTPUT_DIR / f"yolov8{args.model}_defect" / "weights" / "best.pt")
        if args.source is None:
            print("[ERROR] predict mode requires --source (image path or folder)")
        elif not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
        else:
            predict(model_path, args.source, conf=args.conf, imgsz=args.imgsz)

    print("\n>> All done!")
