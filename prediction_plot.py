#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 

This code produces the results in the paper titled 
"A Study on SIREN Wavefield Simulation Guided by Frequency Progressive Curriculum"
By Daoxuan Li

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from My_utilities import load_validation_data, sin_activation
from My_CustomLayers import   SirenLayer

dtype = "float32"
tf.keras.backend.set_floatx(dtype)

# ========================Core parameter configuration ========================
frequency = 10  # Frequency in Hz
epoch = 100000
velocity_model = 'marmousi'  # 'simple','marmousi'
use_PML = False  # False,True

model_configs = [
    {
        "path": f"Results/u_model_epoch_{99999 if epoch == 100000 else epoch}.keras",
        "name": "SIREN"
    },
    {
        "path": f"T1Results/u_model_epoch_{99999 if epoch == 100000 else epoch}.keras",
        "name": "SIREN-FCL"
    }
]

# ========================Grid point configuration ========================
if velocity_model == 'simple':
    fig_siz = [10, 4]
    if frequency == 10:
        npts_x_val = 200
        npts_z_val = 200
elif velocity_model == 'marmousi':
    fig_siz = [12, 4]
    if frequency == 10:
        npts_x_val = 150
        npts_z_val = 100

# ========================Load verification data ========================
data = load_validation_data(
    frequency=frequency,
    velocity_model=velocity_model,
    dtype=dtype,
    use_PML=use_PML
)

dU_2d = data['dU_2d']
xz_val = data['xz_val']
s_xz = data['s_xz']
factor = data['factor']
v0 = data['v0']
v_val = data['v_val']
domain_bounds = data['domain_bounds']
a_x, b_x, a_z, b_z = domain_bounds
domain_bounds_valid = domain_bounds

omega = np.float32(frequency * 2 * np.pi)
if use_PML:
    L_PML = 0.5
    omega0 = omega
    a0 = 1.
    print(a0)
    xz_PML = a_x, b_x, a_z, b_z



def model_prediction(model_path, x_in):
    u_model = keras.models.load_model(model_path,
                                      custom_objects={
                                          'SirenLayer': SirenLayer,
                                          'EmbedderLayer': EmbedderLayer,
                                          'sin': tf.sin,
                                          'cos': tf.cos,
                                          'abs': tf.abs,
                                          'sign': tf.sign,
                                          'sin_activation': sin_activation},
                                      compile=False,
                                      safe_mode=False)

    prediction = u_model(x_in)
    u_real = prediction[:, 0]
    u_real_grid = tf.reshape(u_real, (npts_z_val, npts_x_val)).numpy()
    u_imag = prediction[:, 1]
    u_imag_grid = tf.reshape(u_imag, (npts_z_val, npts_x_val)).numpy()
    return u_real_grid, u_imag_grid

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 14,
    "axes.titlesize": 19,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.autolayout": True  
})

# ========================data pre-processing========================
#Reconstruction of FD simulation results
c_lims = [-np.max(np.abs(dU_2d)), np.max(np.abs(dU_2d))]
u_real_fd = tf.reshape(dU_2d[:, 0], (npts_z_val, npts_x_val)).numpy()
u_imag_fd = tf.reshape(dU_2d[:, 1], (npts_z_val, npts_x_val)).numpy()
v_val_grid = tf.reshape(v_val, (npts_z_val, npts_x_val)).numpy()

# Load the prediction results of the two models
pred_results = []
for cfg in model_configs:
    real, imag = model_prediction(cfg["path"], xz_val)
    pred_results.append({
        "name": cfg["name"],
        "real": real,
        "imag": imag
    })


# ======================== Color mapping definition ========================

velocity_cmap = 'terrain' #"terrain"、turbo

wavefield_cmap = "BrBG"#PuOr、BrBG、RdYlBu、turbo

# ======================== visualization ========================

fig1, axes = plt.subplots(1, 3, figsize=[fig_siz[0] * 2.2, fig_siz[1] * 1.5])
#fig1.suptitle("Velocity Model & FD Simulation Results", fontsize=21, y=1.05)


ax1 = axes[0]
im1 = ax1.imshow(v_val_grid, extent=[a_x, b_x, b_z, a_z], origin="upper",
                 cmap=velocity_cmap, aspect="auto")
ax1.set_title("Velocity Model")
ax1.set_ylabel("Depth z (km)") 
ax1.set_xlabel(" x (km)")
cbar1 = plt.colorbar(im1, ax=ax1, label='Velocity (km/s)') 
cbar1.ax.invert_yaxis()


ax2 = axes[1]
im2 = ax2.imshow(u_real_fd, extent=[a_x, b_x, b_z, a_z], origin="upper",
                cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax2.set_title("FD Simulation Real Part")
ax2.set_ylabel("Depth z (km)")  
ax2.set_xlabel(" x (km)")
cbar2 = plt.colorbar(im2, ax=ax2, label='Amplitude') 


ax3 = axes[2]
im3 = ax3.imshow(u_imag_fd, extent=[a_x, b_x, b_z, a_z], origin="upper",
                cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax3.set_title("FD Simulation Imaginary Part")
ax3.set_ylabel("Depth z (km)")  
ax3.set_xlabel(" x (km)")
cbar3 = plt.colorbar(im3, ax=ax3, label='Amplitude')  

plt.tight_layout()

fig1.savefig("Svelocity_fd_results.eps", format='eps', bbox_inches='tight', dpi=300)
fig1.savefig("Svelocity_fd_results.svg", format='svg', bbox_inches='tight')
plt.show()


fig2, axes = plt.subplots(2, 2, figsize=[fig_siz[0] * 1.1, fig_siz[1] * 2.2])
#fig2.suptitle("Model Predictions", fontsize=21, y=.98)


ax_row0_col0 = axes[0, 0]
im3_1 = ax_row0_col0.imshow(pred_results[0]["real"], extent=[a_x, b_x, b_z, a_z], origin="upper",
                            cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax_row0_col0.set_title(f"{pred_results[0]['name']} Real part")
ax_row0_col0.set_ylabel("$z$ (km)")
ax_row0_col0.set_xlabel("$x$ (km)")
plt.colorbar(im3_1, ax=ax_row0_col0)

ax_row0_col1 = axes[0, 1]
im3_2 = ax_row0_col1.imshow(pred_results[0]["imag"], extent=[a_x, b_x, b_z, a_z], origin="upper",
                            cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax_row0_col1.set_title(f"{pred_results[0]['name']} Imaginary part")
ax_row0_col1.set_ylabel("$z$ (km)")
ax_row0_col1.set_xlabel("$x$ (km)")
plt.colorbar(im3_2, ax=ax_row0_col1)


ax_row1_col0 = axes[1, 0]
im4_1 = ax_row1_col0.imshow(pred_results[1]["real"], extent=[a_x, b_x, b_z, a_z], origin="upper",
                            cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax_row1_col0.set_title(f"{pred_results[1]['name']} Real part")
ax_row1_col0.set_ylabel("$z$ (km)")
ax_row1_col0.set_xlabel("$x$ (km)")
plt.colorbar(im4_1, ax=ax_row1_col0)

ax_row1_col1 = axes[1, 1]
im4_2 = ax_row1_col1.imshow(pred_results[1]["imag"], extent=[a_x, b_x, b_z, a_z], origin="upper",
                            cmap=wavefield_cmap, aspect="auto", interpolation='none', clim=c_lims)
ax_row1_col1.set_title(f"{pred_results[1]['name']} Imaginary part")
ax_row1_col1.set_ylabel("$z$ (km)")
ax_row1_col1.set_xlabel("$x$ (km)")
plt.colorbar(im4_2, ax=ax_row1_col1)

plt.tight_layout()

fig2.savefig("Smodel_predictions.eps", format='eps', bbox_inches='tight', dpi=300)
fig2.savefig("Smodel_predictions.svg", format='svg', bbox_inches='tight')
plt.show()
