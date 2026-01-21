import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tide
from tide import CallbackState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") 

print(f"使用设备: {device}")


ny, nx = 400, 400  
dx = 0.01
model_size_y = ny * dx  
model_size_x = nx * dx  
epsilon_r = torch.ones(ny, nx, device=device) * 6.0  
sigma = torch.zeros(ny, nx, device=device) + 0.001   
mu_r = torch.ones(ny, nx, device=device)             
EP0 = 8.8541878128e-12 
MU0 = 1.2566370614359173e-06  
c0 = 1.0 / (EP0 * MU0)**0.5 
v_medium = c0 / np.sqrt(6.0) 
dt = dx / (v_medium * 2.0)  # CFL 条件
nt = 800  # 时间步数
freq = 500e6  # 500 MHz
t = torch.arange(nt) * dt
t0 = 1.5 / freq  # 子波延迟

source_amplitude = torch.zeros(1, 1, nt, device=device)
pi = 3.14159265359
source_amplitude[0, 0, :] = (1 - 2*(pi*freq*(t - t0))**2) * torch.exp(-(pi*freq*(t - t0))**2)

src_y, src_x = 50, nx // 2  # 距顶部 1m
source_location = torch.tensor([[[src_y, src_x]]], device=device)
n_receivers = 10
receiver_spacing = 20  
receiver_y = src_y  
receiver_x_start = nx // 2 - n_receivers * receiver_spacing // 2
receiver_locations = [[[receiver_y, receiver_x_start + i * receiver_spacing]] for i in range(n_receivers)]
receiver_location = torch.tensor(receiver_locations, device=device).squeeze(1).unsqueeze(0)


model = tide.MaxwellTM(
    epsilon=epsilon_r,
    sigma=sigma,
    mu=mu_r,
    grid_spacing=dx,
)

snapshots = []
snapshot_interval = 10  # 每 10 步保存一次

def save_snapshot(state: CallbackState):
    """回调函数：使用 CallbackState 保存 Ey 场快照
    
    CallbackState 提供标准化接口访问波场、模型和梯度，
    支持 'inner', 'pml', 'full' 三种视图：
    - 'full': 包含 PML 扩展的完整波场 [n_shots, ny+2*pml, nx+2*pml]
    - 'inner': 原始模型区域 [n_shots, ny, nx]
    - 'pml': 包含 PML 但不含 FD padding 的区域
    
    """
    # 使用 get_wavefield 获取内部区域的 Ey 场（原始模型尺寸）
    Ey = state.get_wavefield("Ey", view="inner")
    # Ey shape: [n_shots, ny, nx] - 与输入模型尺寸相同
    snapshots.append(Ey[0].clone().cpu().numpy())
    
    # 打印进度信息
    if state.step % 100 == 0:
        max_amp = Ey.abs().max().item()
        print(f"  Step {state.step:4d}/{state.nt} | "
              f"Time: {state.time*1e9:.2f} ns | "
              f"Progress: {state.progress*100:.1f}% | "
              f"Max |Ey|: {max_amp:.4e}")

# ============== 运行模拟 ==============
print("\n正在运行模拟...")

pml_width = 20  
use_python_backend = False

backend_name = "Python" if use_python_backend else ("CUDA" if device.type == "cuda" else "CPU (C)")
print(f"后端: {backend_name}")

result = model(
    dt=dt,
    source_amplitude=source_amplitude,
    source_location=source_location,
    receiver_location=receiver_location,
    pml_width=pml_width,
    stencil=2,
    python_backend=use_python_backend,
    forward_callback=save_snapshot,
    callback_frequency=snapshot_interval,  # 每 snapshot_interval 步调用一次回调
)

print(f"完成！共保存 {len(snapshots)} 个快照")

# ============== 可视化 ==============
print("\n生成动画...")

fig, ax = plt.subplots(figsize=(10, 10))

# 计算颜色范围 - 使用较小的范围以便观察边界反射
vmax = max(np.abs(s).max() for s in snapshots) * 0.005  # 5% of max to see reflections
vmin = -vmax
print(f"颜色范围: [{vmin:.2e}, {vmax:.2e}]")

inner_ny = ny
inner_nx = nx
inner_size_y = inner_ny * dx
inner_size_x = inner_nx * dx

print(f"波场尺寸: {inner_ny} x {inner_nx} ({inner_size_y:.2f} m x {inner_size_x:.2f} m)")

# 初始图像 - 使用米为单位
im = ax.imshow(
    snapshots[0],
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    extent=(0, inner_size_x, inner_size_y, 0),
)
ax.set_xlabel('x (m)')
ax.set_ylabel('Depth (m)')
title = ax.set_title('Ey Field (Inner Region), t = 0.00 ns')

# 标记震源位置
ax.plot(src_x*dx, src_y*dx, 'k*', markersize=15, label='Source')

# 标记接收器位置
for i in range(n_receivers):
    rx = (receiver_x_start + i * receiver_spacing) * dx
    ry = receiver_y * dx
    ax.plot(rx, ry, 'gv', markersize=8)
ax.plot([], [], 'gv', markersize=8, label='Receivers')

plt.colorbar(im, label='Ey (V/m)')
ax.legend(loc='upper right')

def animate(frame):
    im.set_array(snapshots[frame])
    t_ns = frame * snapshot_interval * dt * 1e9
    title.set_text(f'Ey Field (Inner Region), t = {t_ns:.2f} ns')
    return [im, title]

ani = animation.FuncAnimation(fig, animate, frames=len(snapshots), 
                               interval=50, blit=True)


print("保存动画为 wavefield.gif ...")
ani.save('wavefield.gif', writer='pillow', fps=20)
print("完成！")
