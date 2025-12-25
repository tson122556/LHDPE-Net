import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 修改为你自己的 CSV 文件名
# ==================================================
file_no_soeiou = "/home/sdb/pk/workspace/yolov9_src_pc/out_data_exper/yolov9t_gradient_analysis.csv"       # 未使用SOE-IoU
file_soeiou    = "/home/sdb/pk/workspace/yolov9_src_pc/out_data_exper/lhdpe_gradient_analysis.csv"         # 使用SOE-IoU

# ============================
# 读取两份 gradient CSV
# ============================
df_no = pd.read_csv(file_no_soeiou)
df_so = pd.read_csv(file_soeiou)

mag_no = df_no.iloc[:, 0]        # 第一列：gradient magnitude
dir_no = df_no.iloc[:, 1]        # 第二列：gradient direction

mag_so = df_so.iloc[:, 0]
dir_so = df_so.iloc[:, 1]

# ============================
#  绘制同一图中两条曲线（Magnitude）
# ============================
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(mag_no, label="No SA-IoU", alpha=0.8)
plt.plot(mag_so, label="SA-IoU", alpha=0.8)
plt.title("Gradient Magnitude Comparison")
plt.xlabel("Iterations")
plt.ylabel("Avg Gradient Magnitude")
plt.legend()

# ============================
# 绘制同一图中两条曲线（Direction）
# ============================
plt.subplot(1, 2, 2)
plt.plot(dir_no, label="No SA-IoU", alpha=0.8)
plt.plot(dir_so, label="SA-IoU", alpha=0.8)
plt.title("Gradient Direction Comparison")
plt.xlabel("Iterations")
plt.ylabel("Cosine Similarity")
plt.legend()

plt.tight_layout()
plt.savefig("/home/sdb/pk/workspace/yolov9_src_pc/out_data_exper/lhdpe_gradient_compare.png", dpi=300)

print("绘制完成并已保存：gradient_compare.png")
