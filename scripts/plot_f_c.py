import pandas as pd
import matplotlib.pyplot as plt

# 1. 把表格数据一次性写进 DataFrame
data = {
    "Model": ["YOLOv5-L", "YOLOv8-S", "YOLOv8-M", "YOLOv9-S", "YOLOv9-M",
              "YOLOv10-S", "YOLOv10-M", "YOLOv11-S", "YOLOv11-M",
              "L-FFCA-YOLO", "TPH-YOLOv5-L", "RTDETR", "MFEL-YOLO", "YOLO-DKR", "Ours"],
    "FPS":   [61, 90, 56, 89, 71, 147, 91, 145, 83,
              89, 45, 54, 85, 75, 119],
    "mAP_50":[40.1, 38.6, 42.1, 39.5, 43.4,
              38.6, 42.3, 37.9, 44.1,
              41.0, 40.0, 27.2, 44.7, 49.5, 54.6]
}
df = pd.DataFrame(data)

# 2. 画图
plt.figure(figsize=(6,4))
#  baseline 散点
plt.scatter(df.FPS, df.mAP_50, s=60, color='tab:gray', label='others')
# 高亮 Ours
plt.scatter(119, 54.6, s=250, color='tab:red', marker='*',
            label='Ours', zorder=5)

# 3. 坐标轴与标注
plt.xlim(40, 160)
plt.ylim(25, 60)
plt.xlabel('FPS')
plt.ylabel('mAP@0.5 (%)')
plt.grid(alpha=.3)
plt.legend()

# 4. 保存
plt.tight_layout()
plt.savefig('speed_vs_map.pdf')
plt.show()