import matplotlib.pyplot as plt
import numpy as np

# 参数设置
iou_values = np.linspace(0, 1, 100)
iou_thresh = 0.5
mix_threshold = 0.7
poly_degree = 2

# 计算衰减
linear_decay = 1 - iou_values
polynomial_decay = (1 - iou_values) ** poly_degree

# 选择线性和多项式衰减
linear_mask = (iou_values >= iou_thresh) & (iou_values < mix_threshold)
polynomial_mask = (iou_values >= mix_threshold)

# 混合衰减
mixed_decay = np.where(linear_mask, linear_decay,
                       np.where(polynomial_mask, polynomial_decay, 1.0))

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(iou_values, linear_decay, label='Linear Decay', linestyle='--')
plt.plot(iou_values, polynomial_decay, label='Polynomial Decay', linestyle=':')
plt.plot(iou_values, mixed_decay, label='Mixed Decay', color='black')
plt.axvline(x=iou_thresh, color='r', linestyle='--', label='IoU Threshold')
plt.axvline(x=mix_threshold, color='g', linestyle='--', label='Mix Threshold')
plt.xlabel('IoU')
plt.ylabel('Decay')
plt.title('Decay Functions for Soft-NMS')
plt.legend()
plt.grid(True)
plt.show()
