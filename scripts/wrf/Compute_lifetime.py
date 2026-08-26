import xarray as xr
from netCDF4 import Dataset
import numpy as np
import heapq as hp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import wrf
import scipy
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from tcpyPI import pi

import pandas as pd

# 你提供的 stretching 参数
dx_inner = 2000.0
dx_outer = 15000.0
nos_len = 800000.0
tot_len = 3996000.0

# 计算 nx
zone_len = tot_len - nos_len
avg_dx_stretch = 0.5 * (dx_inner + dx_outer)
nx = int(nos_len / dx_inner + zone_len / avg_dx_stretch)

# 构造 dx array（中间是 dx_inner，两边线性从 dx_outer 到 dx_inner）
half_stretch_len = (tot_len - nos_len) / 2
n_nos = int(nos_len / dx_inner)
n_stretch = int(half_stretch_len / avg_dx_stretch)

# 左半边 dx：从 dx_outer 减小到 dx_inner
dx_left = np.linspace(dx_outer, dx_inner, n_stretch, endpoint=False)
# 中间 constant dx
dx_center = np.full(n_nos, dx_inner)
# 右半边 dx：从 dx_inner 增加到 dx_outer
dx_right = np.linspace(dx_inner, dx_outer, n_stretch)

# 合并为完整 dx 和 dy（对称）
dx_array = np.concatenate([dx_left, dx_center, dx_right])
dy_array = dx_array.copy()

# 构造 2D 网格 dx, dy
dx2d, dy2d = np.meshgrid(dx_array, dy_array)


# 计算权重
weight2d = dx2d * dy2d  # shape: [ny, nx]


# 文件结构建议：
# - d_dx_c.py               # 中心差分函数
# - xy_poisson_greens_fcn.py  # 格林函数解泊松方程
# - yu_chen_main.py         # 主脚本（相当于 YuChen_data.m）

# ----------- d_dx_c.py -----------
import numpy as np

def d_dx_c(V, dx, dim):
    """
    中心差分近似的一维导数，适用于任意维度数组。
    支持边界使用一阶前向/后向差分。
    
    参数：
        V: ndarray，输入数据
        dx: float，间距
        axis: int，导数的方向
    返回：
        dVdx: ndarray，导数数组
    """
    V = np.asarray(V)
    Ndim = V.ndim

    if dim >= Ndim:
        raise ValueError("dim exceeds V's dimensions.")

    dim_order = list(range(Ndim))
    dim_order2 = dim_order.copy()
    dim_order2[0], dim_order2[dim] = dim, 0

    V2 = np.transpose(V, axes=dim_order2)
    shape = V2.shape
    Nx = shape[0]

    dVdx2 = np.zeros_like(V2)

    # 中间：中心差分
    dVdx2[1:-1] = (V2[2:] - V2[:-2]) / (2 * dx)

    # 前端边界：四阶精度前向差分（与MATLAB一致）
    dVdx2[0] = (-11/6 * V2[0] + 3 * V2[1] - 1.5 * V2[2] + 1/3 * V2[3]) / dx

    # 尾端边界：二阶后向差分（保留原公式）
    dx1 = -2 * dx
    dx2 = -dx
    dVdx2[-1] = (V2[-3] * dx2**2 - V2[-2] * dx1**2 - V2[-1] * (dx2**2 - dx1**2)) / (dx1 * dx2 * (dx2 - dx1))

    # 转回原维度顺序
    dVdx = np.transpose(dVdx2, axes=np.argsort(dim_order2))

    return dVdx


# ----------- xy_poisson_greens_fcn.py -----------
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq,fftshift, ifftshift
from scipy.special import j0, j1

def xy_poisson_greens_fcn(x, y, forcing):
    """
    使用格林函数解二维泊松方程 \nabla^2 psi = forcing
    x, y: 1D 坐标数组
    forcing: 2D 源项场 (y, x)
    L: 特征长度，用于贝塞尔函数公式
    返回：psi
    """
    ny, nx = forcing.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Dx = nx*dx
    Dy = ny*dy

    L = np.ceil(1.1 * np.sqrt(Dx**2 + Dy**2))


    Nx_0pad_list = np.arange(4 * nx, 4 * nx + 30 + 1, 2)
    def largest_prime_factor(n):
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
        return n

    large_factors = [largest_prime_factor(n) for n in Nx_0pad_list]
    Nx_0pad = Nx_0pad_list[np.argmin(large_factors)]
    if Nx_0pad % 2 == 1:
        Nx_0pad += 1
    Ny_0pad = Nx_0pad  # 保持相同填充

    Dx_pad = Nx_0pad * dx
    Dy_pad = Ny_0pad * dy

    # Zero-padding
    f_0pad = np.zeros((Ny_0pad, Nx_0pad))  # 注意 shape 顺序 (y, x)
    f_0pad[:ny, :nx] = forcing

    # 频率轴
    fxshift = np.arange(-Nx_0pad//2, Nx_0pad//2) * (2 * np.pi / Dx_pad)
    fyshift = np.arange(-Ny_0pad//2, Ny_0pad//2) * (2 * np.pi / Dy_pad)
    FX, FY = np.meshgrid(fxshift, fyshift, indexing='xy')
    ss_mat = np.sqrt(FX**2 + FY**2)

    # 查找零频处
    fx_zero_ind = np.argmin(np.abs(fxshift))
    fy_zero_ind = np.argmin(np.abs(fyshift))

    # 设置 L
    L = np.ceil(1.1 * np.sqrt(Dx**2 + Dy**2))

    # 构造格林函数频谱
    s_safe = np.where(ss_mat == 0, 1e-10, ss_mat)
    G_fft = -((1 - j0(s_safe * L)) / s_safe**2 - L * np.log(L) * j1(s_safe * L) / s_safe)
    G_fft[fy_zero_ind, fx_zero_ind] = 0  # 设置 s=0 处为 0

    # FFT 求解
    F = fftshift(fft2(f_0pad))
    phi_fft = ifftshift(G_fft * F)
    phi = np.real(ifft2(phi_fft))

    # 裁剪回原始大小
    psi = phi[:ny, :nx]
    return psi


# vorticity: shape (ny, nx)
# X, Y: shape (ny, nx), 网格的 x 和 y 坐标
# x0, y0: 初始猜测中心位置

def compute_centroid(vorticity, X, Y, x0, y0, radius,grid_length):
    # 计算每个点到 (x0, y0) 的距离

    numerator_x = 0;
    numerator_y = 0;
    denominator = 0;
    
    # 构造坐标网格
    xx, yy = np.meshgrid(np.arange(len(X)), np.arange(len(Y)))

    # 计算距离
    dx = xx - x0
    dy = yy - y0
    R = np.sqrt(dx**2 + dy**2) * grid_length  # 假设格距为 2000m

    # 构造 mask，仅保留 R <= radius 区域
    mask = R <= radius

    # 计算加权质心
    weights = -vorticity * mask  # 只取负号加权，且外部置零

    numerator_x = np.sum(dx * weights)
    numerator_y = np.sum(dy * weights)
    denominator = np.sum(vorticity * mask)  # 正数的权重和

    # 防止除以零
    if denominator == 0:
        return x0, y0

    #print('numerator_x, denominator',numerator_x,denominator,numerator_x / denominator)
    #print('numerator_y, denominator',numerator_y,denominator,numerator_y / denominator)
    
    x0_new = x0 + numerator_x / denominator
    y0_new = y0 + numerator_y / denominator

    #print('x0,y0',x0,y0)
    #print('x0_new,y0_new',x0_new,y0_new)

    return x0_new, y0_new


def compute_centroid2(vorticity, X, Y, x0, y0, radius):
    # 计算每个点到 (x0, y0) 的距离

    numerator_x = 0;
    numerator_y = 0;
    denominator = 0;
    
    for x in range(0,len(X)):
        for y in range(0,len(Y)):
            dx = x-x0
            dy = y-y0
            R = np.sqrt(dx**2+dy**2)*2000

            if R <= radius:
                numerator_x +=  dx*( -1* vorticity[y][x]) 
                numerator_y +=  dy*( -1* vorticity[y][x]) 
                denominator +=  (vorticity[y][x])
                #print(x,y,vorticity[y][x])

    #print('numerator_x, denominator',numerator_x,denominator,numerator_x / denominator)
    #print('numerator_y, denominator',numerator_y,denominator,numerator_y / denominator)

    #print('x0,y0 old',x0,y0)    
    x0 += numerator_x/denominator ;
    y0 += numerator_y/denominator ;

    #print('x0,y0',x0,y0)


    return x0 , y0
            
from scipy.interpolate import griddata
def azimuthal_average(field, x_array, y_array, x0, y0, r_bins):
    """
    计算以 (x0, y0) 为中心的 azimuthal 平均
    field: shape (ny, nx)
    x_array, y_array: 1D 数组（网格坐标）
    r_bins: 1D 数组（希望输出的半径 bin 边界）
    """
    ny, nx = field.shape
    X, Y = np.meshgrid(x_array, y_array)
    
    # 1. 计算相对于中心的坐标差
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # 2. 展平数据，准备 bin 平均
    r_flat = r.flatten()
    field_flat = field.flatten()

    # 3. 准备输出
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    azim_avg = np.zeros_like(r_centers)

    for i in range(len(r_centers)):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.any(mask):
            azim_avg[i] = np.mean(field_flat[mask])
        else:
            azim_avg[i] = np.nan  # 没有值就 NaN

    return r_centers, azim_avg


def compute_radial_tangential_masked(u, v, x, y, x0, y0, r_max=None):
    """
    u, v     : 2D arrays at scalar points (e.g., uinterp, vinterp)
    x, y     : 2D coordinate arrays
    x0, y0   : center position
    r_min, r_max : radial range (same units as x, y), optional
    """

    X, Y = np.meshgrid(x, y)

    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)



    mask = r <= r_max



    
    # 单位向量
    rhat_x = dx / (r + 1e-10)
    rhat_y = dy / (r + 1e-10)
    thetahat_x = -dy / (r + 1e-10)
    thetahat_y = dx / (r + 1e-10)

    # 风矢量投影
    ur = u * rhat_x + v * rhat_y
    ut = u * thetahat_x + v * thetahat_y

    # 应用 mask，非目标区域设为 NaN
    ur[~mask] = np.nan
    ut[~mask] = np.nan

    return ur, ut

import numpy as np

def radial_sum_mean(data, x, y, x0, y0, r_max):
    """
    data: 2D array of values
    x, y: 2D coordinate arrays matching data.shape
    x0, y0: center coordinates
    r_max: maximum radius from (x0, y0)
    """
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)

    mask = r <= r_max

    values_in_radius = data[mask]

    total = np.nansum(values_in_radius)
    mean = np.nanmean(values_in_radius)

    return total, mean

def radial_stats_split(data, x, y, x0, y0, r_max):
    """
    data: 2D array of values
    x, y: 2D coordinate arrays matching data.shape
    x0, y0: center coordinates
    r_max: maximum radius from (x0, y0)
    
    Returns:
        pos_total, neg_total,
        pos_mean, neg_mean,
        pos_count, neg_count
    """

    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    mask = r <= r_max

    values_in_radius = data[mask]
    values_in_radius = values_in_radius[~np.isnan(values_in_radius)]  # 去掉 NaN

    pos_values = values_in_radius[values_in_radius > 0]
    neg_values = values_in_radius[values_in_radius < 0]

    pos_total = np.sum(pos_values)
    neg_total = np.sum(neg_values)

    pos_mean = np.mean(pos_values) if len(pos_values) > 0 else np.nan
    neg_mean = np.mean(neg_values) if len(neg_values) > 0 else np.nan

    pos_count = len(pos_values)
    neg_count = len(neg_values)

    return pos_total, neg_total, pos_mean, neg_mean, pos_count, neg_count

def mask_certain_radius(data, x, y, x0, y0, r_max=None):
    """
    data     : 2D arrays at scalar points (e.g., uinterp, vinterp)
    x, y     : 2D coordinate arrays
    x0, y0   : center position
    r_min, r_max : radial range (same units as x, y), optional
    """

    X, Y = np.meshgrid(x, y)

    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)



    mask = r <= r_max

    output = np.copy(data)




    # 应用 mask，非目标区域设为 NaN
    output[~mask] = np.nan

    return output

def get_10m_max_wind(ds,append_list):
    print(ds.filepath())
    times = wrf.getvar(ds, "times", timeidx=wrf.ALL_TIMES)
    number_of_frames = len(times)
    for frame in range (0,number_of_frames):
        print(frame)
        U10m = wrf.getvar(ds, "U10",timeidx=frame,meta=False)
        V10m = wrf.getvar(ds, "V10",timeidx=frame,meta=False)
        wspd10 = np.sqrt(U10m**2 + V10m**2)
    
        append_list.append(np.max(wspd10))
    return append_list



def mean_10m_azimuthal_wind_with_radius(
    wrfout_path: str,
    frames,
    r_bins,
    verbose: bool = True,
):

    ds = Dataset(wrfout_path)
    all_profiles = []

    for frame in frames:
        if verbose:
            print(frame)

        slp = wrf.getvar(ds, "slp", timeidx=frame, meta=False)
        U10m = wrf.getvar(ds, "U10", timeidx=frame, meta=False)
        V10m = wrf.getvar(ds, "V10", timeidx=frame, meta=False)


        ny, nx = slp.shape
        x_arr = np.arange(nx)
        y_arr = np.arange(ny)


        y0, x0 = np.unravel_index(np.argmin(slp), slp.shape)

        wspd10 = np.sqrt(U10m**2 + V10m**2)

        r_centers, u_azim_avg = azimuthal_average(wspd10, x_arr, y_arr, x0, y0, r_bins)

        r_centers_out = np.asarray(r_centers)

        all_profiles.append(np.asarray(u_azim_avg, dtype=float))

    all_profiles = np.vstack(all_profiles)  # shape (Nt, Nr)
    mean_profile = np.nanmean(all_profiles, axis=0)

    ds.close()
    return r_centers_out, mean_profile, all_profiles



def r34_radius_with_time(
    wrfout_path: str,
    frames,
    r_bins,
    verbose: bool = True,
):

    ds = Dataset(wrfout_path)
    radius_list_34knot = []

    for frame in frames:
        if verbose:
            print(frame)

        slp = wrf.getvar(ds, "slp", timeidx=frame, meta=False)
        U10m = wrf.getvar(ds, "U10", timeidx=frame, meta=False)
        V10m = wrf.getvar(ds, "V10", timeidx=frame, meta=False)


        ny, nx = slp.shape
        x_arr = np.arange(nx)
        y_arr = np.arange(ny)

        y0, x0 = np.unravel_index(np.argmin(slp), slp.shape)

        wspd10 = np.sqrt(U10m**2 + V10m**2)

        r_centers, u_azim_avg = azimuthal_average(wspd10, x_arr, y_arr, x0, y0, r_bins)

        idx = np.where(u_azim_avg >= 17)[0]   # 17m/s = 34knot
        if len(idx) == 0:
            radius_34knot = 0.0  # no 34-kt wind found
        else:
            radius_34knot = r_centers[idx[-1]]    #number of grid
    
        radius_list_34knot.append(radius_34knot)



    radius_list_34knot = np.vstack(radius_list_34knot)  # shape (Nt, Nr)

    ds.close()
    return radius_list_34knot

def wrf_air_density_tv(wrfout_path: str, timeidx, meta=False):
    """
    Compute air density (kg/m^3) from WRF output using:
        Tv = T * (1 + 0.608*qv)
        rho = p / (Rd * Tv)

    Requires variables: P, PB, T, QVAPOR (standard in wrfout).
    """
    ds = Dataset(wrfout_path)
    # Total pressure (Pa)
    P  = wrf.getvar(ds, "P",  timeidx=timeidx, meta=False)
    PB = wrf.getvar(ds, "PB", timeidx=timeidx, meta=False)
    Tv = wrf.getvar(ds, "tv", timeidx=frame, meta=False)
    p = P + PB


    Rd = 287.0

    # Density (kg/m^3)
    rho = p / (Rd * Tv)
    return rho

def calculate_TCWC(wrfout_path: str, timeidx, meta=False):
    ds = Dataset(wrfout_path)
    
    qc = wrf.getvar(ds, "QCLOUD", timeidx=timeidx, meta=False)
    qr = wrf.getvar(ds, "QRAIN", timeidx=timeidx, meta=False)
    qi = wrf.getvar(ds, "QICE", timeidx=timeidx, meta=False)
    qs = wrf.getvar(ds, "QSNOW", timeidx=timeidx, meta=False)
    qg = wrf.getvar(ds, "QGRAUP", timeidx=timeidx, meta=False)

    z  = wrf.getvar(ds, "z", timeidx=timeidx, meta=False)

    total_Q = qc+qr+qi+qs+qg
    density = wrf_air_density_tv(wrfout_path, timeidx)

    integrand = density * total_Q
    tcwc = np.trapz(integrand, x=z, axis=0)

    ds.close()
    return tcwc



def azimuthal_average_sector(field, x_array, y_array, x0, y0, r_bins,
                             theta_min, theta_max):
    """
    在给定角度范围 [theta_min, theta_max] 内做 azimuthal mean（扇形平均）

    """

    ny, nx = field.shape
    X, Y = np.meshgrid(x_array, y_array)

    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # [-pi, pi]

    # 展平
    r_flat = r.ravel()
    theta_flat = theta.ravel()
    field_flat = field.ravel()

    # 角度单位处理：统一转为弧度

    tmin = np.deg2rad(theta_min)
    tmax = np.deg2rad(theta_max)


    # 把 theta 归一到 [0, 2pi) 以方便处理跨界区间
    theta_flat_02 = np.mod(theta_flat, 2*np.pi)
    tmin_02 = np.mod(tmin, 2*np.pi)
    tmax_02 = np.mod(tmax, 2*np.pi)

    # 角度 mask（考虑跨越 0 的情况）
    if tmin_02 <= tmax_02:
        theta_mask = (theta_flat_02 >= tmin_02) & (theta_flat_02 <= tmax_02)
    else:
        # 例如 350° 到 20°：theta >= 350° 或 theta <= 20°
        theta_mask = (theta_flat_02 >= tmin_02) | (theta_flat_02 <= tmax_02)

    # 输出
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    azim_avg = np.full_like(r_centers, np.nan, dtype=float)

    for i in range(len(r_centers)):
        r_mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        mask = r_mask & theta_mask
        if np.any(mask):
            azim_avg[i] = np.nanmean(field_flat[mask])

    return r_centers, azim_avg


def diabatic_heating_with_radius_in_certain_angle(
    wrfout_path: str,
    frame,   #just one frame
    r_bins,
    level_range,
    angle_start,
    angle_end
):

    ds = Dataset(wrfout_path)

    all_height_profile = []


    print(frame)

    slp = wrf.getvar(ds, "slp", timeidx=frame, meta=False)
    cumulus_heat = wrf.getvar(ds, 'RTHCUTEN', timeidx=frame, meta=False)
    radiation_heat = wrf.getvar(ds, 'RTHRATEN', timeidx=frame, meta=False)
    pbl_heat = wrf.getvar(ds, 'RTHBLTEN', timeidx=frame, meta=False)
    microphysics_heat = wrf.getvar(ds, 'H_DIABATIC', timeidx=frame, meta=False)


    ny, nx = slp.shape
    x_arr = np.arange(nx)
    y_arr = np.arange(ny)



    total_Q = cumulus_heat+radiation_heat+pbl_heat+microphysics_heat

    
    y0, x0 = np.unravel_index(np.argmin(slp), slp.shape)

    for level in level_range:

        r_centers, Q_azim_avg = azimuthal_average_sector(total_Q[level], x_arr, y_arr, x0, y0, r_bins,angle_start,angle_end)
        r_centers_out = np.asarray(r_centers)
    
        all_height_profile.append(np.asarray(Q_azim_avg, dtype=float))


    ds.close()
    return r_centers_out, all_height_profile


def slp_at_frame(
    wrfout_path: str,
    frame,
):

    ds = Dataset(wrfout_path)
    psfc = wrf.getvar(ds, "PSFC", timeidx=frame, meta=False)

    min_slp = np.min(psfc)/100

    ds.close()
    return min_slp
    

def wind10_max_at_frame(
    wrfout_path: str,
    frame,
):

    ds = Dataset(wrfout_path)
    U10m = wrf.getvar(ds, "U10",timeidx=frame,meta=False)
    V10m = wrf.getvar(ds, "V10",timeidx=frame,meta=False)
    wspd10 = np.sqrt(U10m**2 + V10m**2)

    wind10_max = np.max(wspd10)

    ds.close()
    return wind10_max


def direct_rmw_at_frame(
    wrfout_path: str,
    frame,
    x0,
    y0,
):
    ds = Dataset(wrfout_path)

    U10m = wrf.getvar(ds, "U10", timeidx=frame, meta=False)
    V10m = wrf.getvar(ds, "V10", timeidx=frame, meta=False)

    wspd10 = np.sqrt(U10m**2 + V10m**2)

    # 最大10m风速所在格点
    iy, ix = np.unravel_index(np.nanargmax(wspd10), wspd10.shape)

    # 半径，单位 grid
    rmw = np.sqrt((ix - x0)**2 + (iy - y0)**2)

    ds.close()
    return rmw


def azimuthal_mean_rmw10_at_frame(
    ds,
    frame,
    x0,
    y0,
    rmax_grid=120,
    bin_width_grid=1.0,
):

    U10m = wrf.getvar(ds, "U10", timeidx=frame, meta=False)
    V10m = wrf.getvar(ds, "V10", timeidx=frame, meta=False)

    wspd10 = np.sqrt(U10m**2 + V10m**2)


    ny, nx = wspd10.shape
    yy, xx = np.indices((ny, nx))

    # 每个格点到中心的距离，单位 grid
    r = np.sqrt((xx - x0)**2 + (yy - y0)**2)

    # 径向 bin
    edges = np.arange(0, rmax_grid + bin_width_grid, bin_width_grid)
    r_bin = np.digitize(r.ravel(), edges) - 1

    n_bins = len(edges) - 1

    valid = (
        (r_bin >= 0) &
        (r_bin < n_bins) &
        np.isfinite(wspd10.ravel())
    )

    # 每个半径 bin 内求 azimuthal mean
    sum_wind = np.bincount(
        r_bin[valid],
        weights=wspd10.ravel()[valid],
        minlength=n_bins
    )

    count_wind = np.bincount(
        r_bin[valid],
        minlength=n_bins
    )

    wind_azim_mean = sum_wind / count_wind
    wind_azim_mean[count_wind == 0] = np.nan

    # bin 中心半径，单位 grid
    radius_grid = 0.5 * (edges[:-1] + edges[1:])

    # RMW = azimuthal mean wind 最大的位置
    imax = np.nanargmax(wind_azim_mean)

    rmw_grid = radius_grid[imax]
    max_azim_mean_wind = wind_azim_mean[imax]

    return rmw_grid, max_azim_mean_wind, radius_grid, wind_azim_mean

