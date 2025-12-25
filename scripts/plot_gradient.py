import json
import matplotlib.pyplot as plt

g = json.load(open("/home/sdb/pk/workspace/yolov9_src_pc/grad_magnitude.json"))
d = json.load(open("/home/sdb/pk/workspace/yolov9_src_pc/grad_direction.json"))

# 1 行 2 列 → 水平排列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 左图：梯度大小
ax1.plot(g)
ax1.set_xlabel("iterations")
ax1.set_ylabel("avg gradient magnitude")
ax1.set_title("Gradient Magnitude Curve")

# 右图：梯度方向变化
layer = list(d.keys())[0]
ax2.plot(d[layer])
ax2.set_xlabel("iterations")
ax2.set_ylabel("cosine similarity")
ax2.set_title(f"Gradient Direction Change ({layer})")

plt.tight_layout()
plt.show()