import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec  # 用于自定义子图布局

# ===================== 1. 基础参数与数据加载 =====================
# 通用参数
frequency = 10  # 目标频率
velocity_model = 'simple'
use_PML = False
PML = "PML" if use_PML else "NOPML"

# 加载两个模型的训练历史
# SIREN (无课程学习)
history_siren = np.load(f'Results/training_history.npy', allow_pickle=True).item()
# SIREN-FCL (带课程学习)
history_fcl = np.load(f'T1Results/training_history.npy', allow_pickle=True).item()

# 提取SIREN数据
train_loss_siren = history_siren['training_loss']
val_loss_siren = history_siren['validation_loss']
val_error_siren = history_siren['validation_error']
epochs_siren = range(1, len(train_loss_siren) + 1)

# 提取SIREN-FCL数据
train_loss_fcl = history_fcl['training_loss']
val_loss_fcl = history_fcl['validation_loss']
val_error_fcl = history_fcl['validation_error']
epochs_fcl = range(1, len(train_loss_fcl) + 1)
num_epochs_fcl = len(train_loss_fcl)

# ===================== 2. SIREN-FCL课程学习阶段计算 =====================
curriculum_stages = 4
start_frequency = 2.5
end_frequency = 10
stage_epochs_ratios = [0.15, 0.15, 0.15, 0.55]

# 计算每个阶段的epoch数
stage_epochs_list = [int(ratio * num_epochs_fcl) for ratio in stage_epochs_ratios]
total_allocated = sum(stage_epochs_list)
if total_allocated != num_epochs_fcl:
    stage_epochs_list[-1] += (num_epochs_fcl - total_allocated)

# 计算阶段起始/结束epoch
stage_boundaries = []
cumulative = 0
for i in range(curriculum_stages):
    start_epoch = cumulative
    cumulative += stage_epochs_list[i]
    end_epoch = cumulative
    stage_boundaries.append((start_epoch, end_epoch))

# 生成阶段对应的频率
frequencies = np.linspace(start_frequency, end_frequency, curriculum_stages)

# ===================== 3. 设置全局绘图风格 =====================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14
})

# ===================== 4. 创建自定义子图布局 =====================
# 第一行：2个子图（SIREN损失、SIREN-FCL损失）
# 第二行：1个跨两列的居中子图（合并的验证误差）
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig)
# 第一行两个子图
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
# 第二行合并为一个跨两列的子图（居中）
ax2 = fig.add_subplot(gs[1, :])

'''fig.suptitle(f'PINN Training Loss & Validation Error Comparison\n'
             f'{velocity_model} model, Target: {frequency}Hz, PML: {PML}',
             fontsize=14, fontweight='bold')'''

# ===================== 5. 绘制第一行：损失曲线 =====================
# 5.1 第一行第一列：SIREN损失曲线
ax11.plot(epochs_siren, train_loss_siren, label='Training Loss', color='blue', alpha=0.7, linewidth=1)
ax11.plot(epochs_siren[::100], val_loss_siren, label='Validation Loss', color='red',
          marker='o', markersize=3, linewidth=1, alpha=0.7)
ax11.set_title('SIREN - Loss Curves')
ax11.set_xlabel('Epoch')
ax11.set_ylabel('Loss')
ax11.set_yscale('log')
ax11.legend()
ax11.grid(alpha=0.3, linestyle='--')

# 5.2 第一行第二列：SIREN-FCL损失曲线（保留课程学习标注）
ax12.plot(epochs_fcl, train_loss_fcl, label='Training Loss', color='blue', alpha=0.7, linewidth=1)
if len(val_loss_fcl) > 0:
    val_epochs_fcl = np.linspace(0, len(epochs_fcl) - 1, len(val_loss_fcl), dtype=int)
    ax12.plot(val_epochs_fcl, val_loss_fcl, label='Validation Loss', color='red',
              marker='o', markersize=3, linewidth=1, alpha=0.7)

# 添加课程学习阶段背景色和标签
for i, (start_epoch, end_epoch) in enumerate(stage_boundaries):
    freq = frequencies[i]
    color = plt.cm.viridis(i / curriculum_stages)
    # 阶段背景色
    ax12.axvspan(start_epoch, end_epoch, alpha=0.1, color=color)
    # 阶段频率标签
    mid_epoch = (start_epoch + end_epoch) / 2
    ax12.text(mid_epoch, ax12.get_ylim()[1] * 0.95, f'{freq:.1f}Hz',
             ha='center', va='top', fontsize=8, alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))

# 阶段分隔线
for start_epoch, end_epoch in stage_boundaries:
    ax12.axvline(x=end_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

ax12.set_title('SIREN-FCL - Loss Curves')
ax12.set_xlabel('Epoch')
ax12.set_ylabel('Loss')
ax12.set_yscale('log')
ax12.legend()
ax12.grid(alpha=0.3, linestyle='--')

# ===================== 6. 绘制第二行：合并的验证误差曲线（居中） =====================
# 6.1 绘制SIREN验证误差
ax2.plot(epochs_siren[::100], val_error_siren, label='SIREN - Validation Error',
         color='darkgreen', marker='s', markersize=3, linewidth=1.5, alpha=0.8)

# 6.2 绘制SIREN-FCL验证误差
if len(val_error_fcl) > 0:
    error_epochs_fcl = np.linspace(0, len(epochs_fcl) - 1, len(val_error_fcl), dtype=int)
    ax2.plot(error_epochs_fcl, val_error_fcl, label='SIREN-FCL - Validation Error',
             color='orange', marker='^', markersize=3, linewidth=1.5, alpha=0.8)

# 6.3 保留SIREN-FCL课程学习阶段标注
# 添加课程学习阶段背景色
for i, (start_epoch, end_epoch) in enumerate(stage_boundaries):
    freq = frequencies[i]
    color = plt.cm.viridis(i / curriculum_stages)
    ax2.axvspan(start_epoch, end_epoch, alpha=0.1, color=color)
    # 阶段频率标签（放置在子图上方）
    mid_epoch = (start_epoch + end_epoch) / 2
    ax2.text(mid_epoch, ax2.get_ylim()[1] * 0.95, f'Stage {i+1}: {freq:.1f}Hz',
             ha='center', va='top', fontsize=8, alpha=0.8,
             bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))

# 添加阶段分隔线
for start_epoch, end_epoch in stage_boundaries:
    ax2.axvline(x=end_epoch, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

# 6.4 设置子图属性
ax2.set_title('Validation Error Comparison (SIREN vs SIREN-FCL)', fontweight='medium')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Error')
ax2.set_yscale('log')
ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95))  # loc指定右上角，bbox_to_anchor微调偏移
ax2.grid(alpha=0.3, linestyle='--')

# ===================== 7. 调整布局并保存/显示 =====================
plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 预留底部空间
plt.savefig('Results/mloss_error_comparison.eps',format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Results/mloss_error_comparison.svg',format='svg', bbox_inches='tight')
plt.show()