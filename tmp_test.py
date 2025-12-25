import matplotlib.pyplot as plt

# 示例数据
models = [
    "YOLOv9 (Ours)", "GELAN (Ours)", "PPYOLOE", "YOLOv5 r7.0",
    "YOLOv6 v3.0", "YOLOv7", "YOLOv8", "DAMO YOLO",
    "Gold YOLO", "RTMDet", "RT DETR", "YOLO MS"
]
params = [2, 7.1, 20, 25.3, 30, 35, 45, 50, 55, 60, 70, 80]  # 参数数量（M）
aps = [38.3, 46.8, 51.4, 53.0, 55.6, 52.5, 48.6, 54.1, 53.2, 52.9, 50.5, 46.3]  # 平均精度AP (%)

# 绘制图表
plt.figure(figsize=(10, 8))
plt.scatter(params, aps, c='r', label="YOLOv9 (Ours)")
plt.plot(params, aps, c='r')

# 添加其他模型的数据
other_models = {
    "GELAN (Ours)": (35, 54.5),
    "PPYOLOE": (30, 51.4),
    "YOLOv5 r7.0": (25, 52.1),
    # 添加其他模型数据
}

for model, (param, ap) in other_models.items():
    plt.scatter(param, ap, label=model)
    plt.plot(param, ap)

# 设置标题和标签
plt.title("Performance on MS COCO Object Detection Dataset")
plt.xlabel("Number of Parameters (M)")
plt.ylabel("MS COCO Object Detection AP (%)")
plt.legend(loc="lower right")

# 显示网格线
plt.grid(True)

# 显示图表
plt.show()
