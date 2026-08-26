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





def plot_heat_map(x_axis,y_axis,data_profiles,title,colorbar_label,xlabel,ylabel,cmap="RdBu_r",figsize=None,pad=None,
                  vmin_on=False, vmax_on=False, vmin=None, vmax=None):


    kwargs = {"shading": "auto", "cmap": cmap}
    data_profiles=np.asarray(data_profiles)
    
    if vmin_on:
        if vmin is None:
            raise ValueError("vmin_on=True 时必须提供 vmin")
        kwargs["vmin"] = vmin

    if vmax_on:
        if vmax is None:
            raise ValueError("vmax_on=True 时必须提供 vmax")
        kwargs["vmax"] = vmax



    X_plot, Y_plot = np.meshgrid(x_axis, y_axis)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(X_plot, Y_plot, data_profiles, **kwargs)
    if pad is None:
        fig.colorbar(pcm, ax=ax, label=colorbar_label)
    else:
        fig.colorbar(pcm, ax=ax, label=colorbar_label, pad=pad)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    

    return ax


def get_theta_height_time_evolution(
    wrfout_path: str,
    frames_to_use,
    layers_to_use,
    
):
    ds = Dataset(wrfout_path)
    reference_height = ((wrf.getvar(ds, "PH", timeidx=1)+wrf.getvar(ds, "PHB", timeidx=1))/9.81/1000)[:,1,1]
    number_of_layers = len(layers_to_use)


    vertical_theta_anomaly_all_frames = []
    min_slp = []

    for frame in frames_to_use:
        print(f"读取第 {frame} 帧")
        #eth_frame = wrf.getvar(ds, "eth", timeidx=frame, meta=False)  # shape: (z, y, x)
        potential_temp = wrf.getvar(ds, 'theta', timeidx=frame, meta=False)   # T+300
        
        slp = wrf.getvar(ds, "slp",timeidx=frame,meta=False)
        min_pres_idx = np.unravel_index(np.argmin(slp), slp.shape)
        y0, x0 = min_pres_idx

        ny, nx = slp.shape
        x_array = np.arange(nx)
        y_array = np.arange(ny)
        
        
        vertical_theta_anomaly_this_frame = []
        vertical_environment_theta_this_frame = []
        
        for layer in layers_to_use:
            #print(f"Layer {layer}/{number_of_layers}")
    
            r_avg_in = 1
            r_avg_out = 15
    
            r_env_in = 100
            r_env_out = 130        
            
    
            
            X, Y = np.meshgrid(x_array, y_array)  # 注意：先x后y
            r = np.sqrt((X - x0)**2 + (Y - y0)**2)
            mask1 = (r >= r_avg_in) & (r <= r_avg_out)
            mask2 = (r >= r_env_in) & (r <= r_env_out)
    
            
            inner_theta_mean = np.nanmean(potential_temp[layer][mask1])
            environment_theta_mean = np.nanmean(potential_temp[layer][mask2])
    
            vertical_theta_anomaly_this_frame.append(inner_theta_mean-environment_theta_mean)
    
    
        vertical_theta_anomaly_all_frames.append(vertical_theta_anomaly_this_frame)
    
    
    
    vertical_theta_anomaly_all_frames = np.array(vertical_theta_anomaly_all_frames)
    height = reference_height[layers_to_use[0]:layers_to_use[-1]+1]

    ds.close()
    return vertical_theta_anomaly_all_frames, height

def get_diabatic_top_view_certain_level(
    ds,
    frame_to_use,
    r_bins: np.ndarray,
    layer_to_use,
    
):


    reference_height = ((wrf.getvar(ds, "PH", timeidx=1)+wrf.getvar(ds, "PHB", timeidx=1))/9.81/1000)[:,1,1]

    height_of_this_level = reference_height[layer_to_use]


    cumulus_heat = wrf.getvar(ds, 'RTHCUTEN', timeidx=frame_to_use, meta=False)[layer_to_use]
    radiation_heat = wrf.getvar(ds, 'RTHRATEN', timeidx=frame_to_use, meta=False)[layer_to_use]
    pbl_heat = wrf.getvar(ds, 'RTHBLTEN', timeidx=frame_to_use, meta=False)[layer_to_use]
    microphysics_heat = wrf.getvar(ds, 'H_DIABATIC', timeidx=frame_to_use, meta=False)[layer_to_use]

    diabatic = np.asarray(cumulus_heat+radiation_heat+pbl_heat+microphysics_heat, dtype=float)





    return diabatic
    

#################################


def get_precip_at_frame(ds, frame):

    prec_c = ds.variables["PREC_ACC_C"][frame, :, :]
    prec_nc = ds.variables["PREC_ACC_NC"][frame, :, :]

    precip = prec_c + prec_nc


    return precip


######################################

def get_theta_certain_level_at_frame(ds, frame,layer_to_use):

    theta = wrf.getvar(ds,'theta',timeidx=frame,meta=False)[layer_to_use]


    return theta

def get_diabatic_3d_at_frame(ds, frame):
    """
    Return total diabatic heating at one frame.
    Shape: (z, y, x)
    """

    cumulus_heat = wrf.getvar(ds, "RTHCUTEN", timeidx=frame, meta=False)
    radiation_heat = wrf.getvar(ds, "RTHRATEN", timeidx=frame, meta=False)
    pbl_heat = wrf.getvar(ds, "RTHBLTEN", timeidx=frame, meta=False)
    microphysics_heat = wrf.getvar(ds, "H_DIABATIC", timeidx=frame, meta=False)

    diabatic_3d = (
        np.asarray(cumulus_heat, dtype=float)
        + np.asarray(radiation_heat, dtype=float)
        + np.asarray(pbl_heat, dtype=float)
        + np.asarray(microphysics_heat, dtype=float)
    )

    return diabatic_3d


def radial_height_mean(field_3d, x0, y0, dx_km, r_bins_km):
    """
    Convert a 3D field (z, y, x) into radius-height mean.

    Returns
    -------
    r_centers : 1D array
        Radius bin centers, km.
    rz_mean : 2D array
        Shape: (z, r). Azimuthal mean field.
    """

    field_3d = np.asarray(field_3d, dtype=float)

    nz, ny, nx = field_3d.shape

    yy, xx = np.indices((ny, nx))

    r_km = np.sqrt((xx - x0)**2 + (yy - y0)**2) * dx_km

    nr = len(r_bins_km) - 1
    rz_mean = np.full((nz, nr), np.nan)

    for j in range(nr):
        mask = (r_km >= r_bins_km[j]) & (r_km < r_bins_km[j + 1])

        if np.any(mask):
            # field_3d[:, mask] shape = (z, number_of_points_in_annulus)
            rz_mean[:, j] = np.nanmean(field_3d[:, mask], axis=1)

    r_centers = 0.5 * (r_bins_km[:-1] + r_bins_km[1:])

    return r_centers, rz_mean    


def get_region_stats(arr, threshold, prefix):
    """
    对一个 2D 区域 arr 计算常用统计量。

    Parameters
    ----------
    arr : 2D array
        例如 diabatic、rain、box_region、box_large_region
    threshold : float
        阈值。例如 diabatic heating 用 0.005 K/s，rain 可以用 0.1 mm 等
    prefix : str
        输出字典里变量名前缀。例如:
        "heating"
        "heating_box"
        "heating_box_large"
        "rain"
        "rain_box"
        "rain_box_large"

    Returns
    -------
    stats : dict
        包含 max, mean, sum 等统计量
    """

    arr = np.asarray(arr, dtype=float)

    # 防止空区域或全 NaN 报错
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_sum": np.nan,
            f"{prefix}_positive_sum": np.nan,
            f"{prefix}_n_above_threshold": 0,
            f"{prefix}_sum_above_threshold": np.nan,
        }

    above = arr > threshold

    stats = {
        f"{prefix}_max": np.nanmax(arr),
        f"{prefix}_mean": np.nanmean(arr),
        f"{prefix}_sum": np.nansum(arr),
        f"{prefix}_positive_sum": np.nansum(np.where(arr > 0, arr, 0.0)),
        f"{prefix}_n_above_threshold": int(np.sum(above)),
        f"{prefix}_sum_above_threshold": np.nansum(np.where(above, arr, 0.0)),
    }

    return stats



def add_all_region_stats(
    row,
    field,
    name,
    threshold,
    box_y1, box_y2, box_x1, box_x2,
    box_large_y1, box_large_y2, box_large_x1, box_large_x2
):
    """
    对一个变量 field 自动计算:
    1. 整个 crop 区域
    2. RMW box 区域
    3. large box 区域
    """

    # 整个 crop 区域
    row.update(
        get_region_stats(
            field,
            threshold=threshold,
            prefix=name
        )
    )

    # RMW box 区域
    box_region = field[box_y1:box_y2, box_x1:box_x2]

    row.update(
        get_region_stats(
            box_region,
            threshold=threshold,
            prefix=f"{name}_box"
        )
    )

    # large box 区域
    box_large_region = field[
        box_large_y1:box_large_y2,
        box_large_x1:box_large_x2
    ]

    row.update(
        get_region_stats(
            box_large_region,
            threshold=threshold,
            prefix=f"{name}_box_large"
        )
    )

    return row

path = "wrf_ideal_12mshear_restart_at_60h_28sst_minsetup_largeTC_d2.nc"
ds = Dataset(path)

frames = np.arange(0, 24, 1)

fig, axes = plt.subplots(4, 6, figsize=(26, 20))
axes = axes.ravel()

#fig_p, axes_p = plt.subplots(5, 6, figsize=(26, 21))
#axes_p = axes_p.ravel()

mappable = None

consider_level = 5

size_of_large_box = 3

stats_varying = []

# 阈值：超过这个 heating 的点数量
threshold = 0.005   # K/s

# 裁剪区域设置
crop_y1, crop_y2 = 300, 500
crop_x1, crop_x2 = 300, 500

# 裁剪后图像中心
center_y = 100
center_x = 100



thresholds = {
    "heating": 0.005,   # diabatic heating, K/s
    "rain": 0.005,      # 你可以根据 rain 单位改，比如 0.1
    "theta": 0.005,     # theta 用这个阈值未必有物理意义，可之后改
}


# 先计算这些 frame 对应的 RMW
rmw10_dict = {}

for frame in frames:
    
    rmw_this_frame = direct_rmw_at_frame(
        path,
        frame,
        x0=400,
        y0=400
    )
    '''
    rmw_grid, vmax_azim_mean, radius_grid, wind_azim_mean = azimuthal_mean_rmw10_at_frame(
        ds,
        frame,
        x0=400,
        y0=400,
        rmax_grid=120,
        bin_width_grid=1.0
    )
    '''
    rmw10_dict[frame] = rmw_this_frame


for i, frame in enumerate(frames):
    print(frame)
    ax = axes[i]

    diabatic = get_diabatic_top_view_certain_level(
        ds,
        frame,
        np.arange(1, 120),
        consider_level
    )

    diabatic = diabatic[crop_y1:crop_y2, crop_x1:crop_x2]

    rain = get_precip_at_frame(ds, frame)
    rain = rain[crop_y1:crop_y2, crop_x1:crop_x2]

    theta = get_theta_certain_level_at_frame(ds, frame,consider_level)
    theta = theta[crop_y1:crop_y2, crop_x1:crop_x2]

    # =====================================================
    # 根据当前 frame 的 RMW 动态定义 box, box_large
    # =====================================================

    rmw = rmw10_dict[frame]*1
    box_half_width_grid = int(np.round(rmw))


    box_y1 = center_y - box_half_width_grid
    box_y2 = center_y + box_half_width_grid
    box_x1 = center_x - box_half_width_grid
    box_x2 = center_x + box_half_width_grid

    box_large_y1 = center_y - box_half_width_grid*size_of_large_box
    box_large_y2 = center_y + box_half_width_grid*size_of_large_box
    box_large_x1 = center_x - box_half_width_grid*size_of_large_box
    box_large_x2 = center_x + box_half_width_grid*size_of_large_box
    

    # 防止 box 超出 200x200 裁剪范围
    box_y1 = max(box_y1, 0)
    box_y2 = min(box_y2, diabatic.shape[0])
    box_x1 = max(box_x1, 0)
    box_x2 = min(box_x2, diabatic.shape[1])

    box_large_y1 = max(box_large_y1, 0)
    box_large_y2 = min(box_large_y2, diabatic.shape[0])
    box_large_x1 = max(box_large_x1, 0)
    box_large_x2 = min(box_large_x2, diabatic.shape[1])

    # =====================================================
    # 整个裁剪区域统计
    # =====================================================
    row = {
        "frame": frame,
        "rmw": rmw,
        "box_half_width_grid": box_half_width_grid,

        "box_y1": box_y1,
        "box_y2": box_y2,
        "box_x1": box_x1,
        "box_x2": box_x2,

        "box_large_y1": box_large_y1,
        "box_large_y2": box_large_y2,
        "box_large_x1": box_large_x1,
        "box_large_x2": box_large_x2,
    }


    # =====================================================
    # 自动统计 heating / rain / theta
    # =====================================================

    row = add_all_region_stats(
        row=row,
        field=diabatic,
        name="heating",
        threshold=thresholds["heating"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    row = add_all_region_stats(
        row=row,
        field=rain,
        name="rain",
        threshold=thresholds["rain"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    row = add_all_region_stats(
        row=row,
        field=theta,
        name="theta",
        threshold=thresholds["theta"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    stats_varying.append(row)

    # =====================================================
    # 画图
    # =====================================================

    mappable = ax.pcolormesh(
        rain,
        vmin=0,
        vmax=0.020,
        shading="auto",
        cmap="RdBu_r"
    )

    # 把动态 RMW box 画出来
    rect = plt.Rectangle(
        (box_x1, box_y1),
        box_x2 - box_x1,
        box_y2 - box_y1,
        fill=False,
        linewidth=1.5,
        edgecolor="yellow",
    )
    ax.add_patch(rect)

    rect_large = plt.Rectangle(
        (box_large_x1, box_large_y1),
        box_large_x2 - box_large_x1,
        box_large_y2 - box_large_y1,
        fill=False,
        linewidth=1.5,
        edgecolor="magenta",
    )
    ax.add_patch(rect_large)

    tick_pos = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    tick_lab = np.array([0, 75, 150, 225, 300, 375, 450, 525, 600]) - 300

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)
    ax.tick_params(axis='both', labelsize=8)

    ax.set_xlabel("r (km)", fontsize=8)
    ax.set_ylabel("r (km)", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{frame} h",
        fontsize=15
    )



# 如果 subplot 多于 frame 数，隐藏多余 panel
for j in range(len(frames), len(axes)):
    axes[j].axis("off")


fig.colorbar(
    mappable,
    ax=axes,
    orientation="vertical",
    pad=0.02,
    label="Diabatic heating (K/s)",
    fraction=0.02,
    aspect=30
)

fig.subplots_adjust(top=0.88,bottom=0.15,hspace=0.4, right=0.84)
fig.suptitle("Weak TC group example diabatic heating", fontsize=20, y=0.93)



stats_varying = pd.DataFrame(stats_varying)

path = "wrf_ideal_12mshear_restart_at_96h_28sst_minsetup_largeTC_d2.nc"
ds = Dataset(path)

frames = np.arange(0, 24, 1)

fig, axes = plt.subplots(4, 6, figsize=(26, 20))
axes = axes.ravel()

#fig_p, axes_p = plt.subplots(5, 6, figsize=(26, 21))
#axes_p = axes_p.ravel()

mappable = None

consider_level = 5

size_of_large_box = 3

stats_control = []

# 阈值：超过这个 heating 的点数量
threshold = 0.005   # K/s

# 裁剪区域设置
crop_y1, crop_y2 = 300, 500
crop_x1, crop_x2 = 300, 500

# 裁剪后图像中心
center_y = 100
center_x = 100



thresholds = {
    "heating": 0.005,   # diabatic heating, K/s
    "rain": 0.005,      # 你可以根据 rain 单位改，比如 0.1
    "theta": 0.005,     # theta 用这个阈值未必有物理意义，可之后改
}


# 先计算这些 frame 对应的 RMW
rmw10_dict = {}

for frame in frames:
    
    rmw_this_frame = direct_rmw_at_frame(
        path,
        frame,
        x0=400,
        y0=400
    )
    '''
    rmw_grid, vmax_azim_mean, radius_grid, wind_azim_mean = azimuthal_mean_rmw10_at_frame(
        ds,
        frame,
        x0=400,
        y0=400,
        rmax_grid=120,
        bin_width_grid=1.0
    )
    '''
    rmw10_dict[frame] = rmw_this_frame


for i, frame in enumerate(frames):
    print(frame)
    ax = axes[i]

    diabatic = get_diabatic_top_view_certain_level(
        ds,
        frame,
        np.arange(1, 120),
        consider_level
    )

    diabatic = diabatic[crop_y1:crop_y2, crop_x1:crop_x2]

    rain = get_precip_at_frame(ds, frame)
    rain = rain[crop_y1:crop_y2, crop_x1:crop_x2]

    theta = get_theta_certain_level_at_frame(ds, frame,consider_level)
    theta = theta[crop_y1:crop_y2, crop_x1:crop_x2]

    # =====================================================
    # 根据当前 frame 的 RMW 动态定义 box, box_large
    # =====================================================

    rmw = rmw10_dict[frame]*1
    box_half_width_grid = int(np.round(rmw))


    box_y1 = center_y - box_half_width_grid
    box_y2 = center_y + box_half_width_grid
    box_x1 = center_x - box_half_width_grid
    box_x2 = center_x + box_half_width_grid

    box_large_y1 = center_y - box_half_width_grid*size_of_large_box
    box_large_y2 = center_y + box_half_width_grid*size_of_large_box
    box_large_x1 = center_x - box_half_width_grid*size_of_large_box
    box_large_x2 = center_x + box_half_width_grid*size_of_large_box
    

    # 防止 box 超出 200x200 裁剪范围
    box_y1 = max(box_y1, 0)
    box_y2 = min(box_y2, diabatic.shape[0])
    box_x1 = max(box_x1, 0)
    box_x2 = min(box_x2, diabatic.shape[1])

    box_large_y1 = max(box_large_y1, 0)
    box_large_y2 = min(box_large_y2, diabatic.shape[0])
    box_large_x1 = max(box_large_x1, 0)
    box_large_x2 = min(box_large_x2, diabatic.shape[1])

    # =====================================================
    # 整个裁剪区域统计
    # =====================================================
    row = {
        "frame": frame,
        "rmw": rmw,
        "box_half_width_grid": box_half_width_grid,

        "box_y1": box_y1,
        "box_y2": box_y2,
        "box_x1": box_x1,
        "box_x2": box_x2,

        "box_large_y1": box_large_y1,
        "box_large_y2": box_large_y2,
        "box_large_x1": box_large_x1,
        "box_large_x2": box_large_x2,
    }


    # =====================================================
    # 自动统计 heating / rain / theta
    # =====================================================

    row = add_all_region_stats(
        row=row,
        field=diabatic,
        name="heating",
        threshold=thresholds["heating"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    row = add_all_region_stats(
        row=row,
        field=rain,
        name="rain",
        threshold=thresholds["rain"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    row = add_all_region_stats(
        row=row,
        field=theta,
        name="theta",
        threshold=thresholds["theta"],
        box_y1=box_y1,
        box_y2=box_y2,
        box_x1=box_x1,
        box_x2=box_x2,
        box_large_y1=box_large_y1,
        box_large_y2=box_large_y2,
        box_large_x1=box_large_x1,
        box_large_x2=box_large_x2
    )

    stats_control.append(row)

    # =====================================================
    # 画图
    # =====================================================

    mappable = ax.pcolormesh(
        rain,
        vmin=0,
        vmax=0.020,
        shading="auto",
        cmap="RdBu_r"
    )

    # 把动态 RMW box 画出来
    rect = plt.Rectangle(
        (box_x1, box_y1),
        box_x2 - box_x1,
        box_y2 - box_y1,
        fill=False,
        linewidth=1.5,
        edgecolor="yellow",
    )
    ax.add_patch(rect)

    rect_large = plt.Rectangle(
        (box_large_x1, box_large_y1),
        box_large_x2 - box_large_x1,
        box_large_y2 - box_large_y1,
        fill=False,
        linewidth=1.5,
        edgecolor="magenta",
    )
    ax.add_patch(rect_large)

    tick_pos = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    tick_lab = np.array([0, 75, 150, 225, 300, 375, 450, 525, 600]) - 300

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)
    ax.tick_params(axis='both', labelsize=8)

    ax.set_xlabel("r (km)", fontsize=8)
    ax.set_ylabel("r (km)", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{frame} h",
        fontsize=15
    )



# 如果 subplot 多于 frame 数，隐藏多余 panel
for j in range(len(frames), len(axes)):
    axes[j].axis("off")


cbar = fig.colorbar(
    mappable,
    ax=axes,
    orientation="vertical",
    pad=0.02,
    label="Diabatic heating (K s$^{-1}$)",
    fraction=0.02,
    aspect=30
)

# 手动把 colorbar 往上移动
pos = cbar.ax.get_position()
cbar.ax.set_position([
    pos.x0,          # 左右位置不变
    pos.y0,  # 往上移动；0.05 可以自己调
    pos.width,      # 宽度不变
    pos.height      # 高度不变
])

fig.subplots_adjust(top=0.88,bottom=0.12,hspace=0.4, right=0.84)
fig.suptitle("Strong TC group example diabatic heating", fontsize=20, y=0.93)



stats_control = pd.DataFrame(stats_control)


stats_control_1 = stats_control.copy()
stats_varying_1 = stats_varying.copy()

stats_control_mean = (
    stats_control_1.set_index("frame")
#    + stats_control_2.set_index("frame")
#    + stats_control_3.set_index("frame")
#    + stats_control_4.set_index("frame")
#    + stats_control_5.set_index("frame")
#    + stats_control_6.set_index("frame")
) / 1.0

stats_control_mean = stats_control_mean.reset_index()

stats_varying_mean = (
    stats_varying_1.set_index("frame")
#    + stats_varying_2.set_index("frame")
#    + stats_varying_3.set_index("frame")
#    + stats_varying_4.set_index("frame")
#    + stats_varying_5.set_index("frame")
#    + stats_varying_6.set_index("frame")
) / `.0

stats_varying_mean = stats_varying_mean.reset_index()

variable_to_consider = 'heating_box_mean'
variable_to_consider2 = 'heating_box_large_mean'

plt.figure(figsize=(6, 4))

plt.plot(
    stats_control_mean["frame"][1:15],
    stats_control_mean[variable_to_consider][1:15],
    marker="o",
    label='Within 1 RMW'
)

plt.plot(
    stats_control_mean["frame"][1:15],
    stats_control_mean[variable_to_consider2][1:15],
    marker='o',
    linestyle='--',
    color='red',
    label='Within 3 RMW'
)



plt.xlabel("Time (h)")
plt.ylabel(f"Mean heating (k/s)")
plt.title(f"Mean heating over time for strong TC group")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


#plt.plot(slp_list)
#plt.show()

#########################

plt.figure(figsize=(6, 4))

plt.plot(
    stats_varying_mean["frame"][1:15],
    stats_varying_mean[variable_to_consider][1:15],
    marker="o",
    label='Within 1 RMW'
)

plt.plot(
    stats_varying_mean["frame"][1:15],
    stats_varying_mean[variable_to_consider2][1:15],
    marker='o',
    linestyle='--',
    color='red',
    label='Within 3 RMW'
)

plt.xlabel("Time (h)")
plt.ylabel(f"Mean heating (k/s) ")
plt.title(f"Mean heating over time for weak TC group")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

plt.plot(
    stats_varying_mean["frame"][1:15],
    stats_varying_mean["heating_box_sum"][1:15] / stats_varying_mean["heating_box_large_sum"][1:15],
    marker="o",
    label="Weak intensity group"
)

plt.plot(
    stats_control_mean["frame"][1:15],
    stats_control_mean["heating_box_sum"][1:15] / stats_control_mean["heating_box_large_sum"][1:15],
    marker="o",
    label="Strong intensity group"
)

plt.xlabel("Time (h)")
plt.ylabel(f"H(r ≤ 1*RMW) / H(r ≤ 3*RMW)")
plt.title(f"Heating ratio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()










path = "wrf_ideal_12mshear_restart_at_60h_28sst_minsetup_largeTC_d2.nc"
ds = Dataset(path)

#frames = np.arange(0, 24, 1)
#fig, axes = plt.subplots(4, 6, figsize=(26, 20))

frames = [1,6,9,13,16,19]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

mappable = None

# 你的 d02 网格间距
dx_km = 3.0

# TC center
x0 = 400
y0 = 400

# 半径范围，单位 km
r_bins_km = np.arange(0, 301, 5)

# 高度：沿用你原来的 reference_height 写法
reference_height = (
    (wrf.getvar(ds, "PH", timeidx=1) + wrf.getvar(ds, "PHB", timeidx=1))
    / 9.81
    / 1000.0
)[:, 1, 1]

# 先计算 RMW
rmw10_dict = {}

for frame in frames:
    rmw_this_frame = direct_rmw_at_frame(
        path,
        frame,
        x0=x0,
        y0=y0
    )
    rmw10_dict[frame] = rmw_this_frame


for i, frame in enumerate(frames):
    print(frame)
    ax = axes[i]

    # =====================================================
    # 读取 3D diabatic heating
    # =====================================================

    diabatic_3d = get_diabatic_3d_at_frame(ds, frame)

    # PH/PHB 是 w-level，通常比 diabatic 多一层
    # 转成 mass-level height
    if len(reference_height) == diabatic_3d.shape[0] + 1:
        height_km = 0.5 * (reference_height[:-1] + reference_height[1:])
    else:
        height_km = reference_height

    # =====================================================
    # 计算 radius-height azimuthal mean
    # =====================================================

    r_centers, diabatic_rz = radial_height_mean(
        diabatic_3d,
        x0=x0,
        y0=y0,
        dx_km=dx_km,
        r_bins_km=r_bins_km
    )

    # =====================================================
    # 画 radial-height plot
    # =====================================================

    R, Z = np.meshgrid(r_centers, height_km)

    mappable = ax.contourf(
        R,
        Z,
        diabatic_rz,
        levels=np.linspace(-0.005, 0.020, 26),
        cmap="RdBu_r",
        extend="both"
    )

    # 画 RMW 和 3RMW
    rmw_grid = rmw10_dict[frame]
    rmw_km = rmw_grid * dx_km

    ax.axvline(
        rmw_km,
        color="yellow",
        linewidth=1.6,
        linestyle="-",
        label="RMW"
    )

    ax.axvline(
        3 * rmw_km,
        color="magenta",
        linewidth=1.6,
        linestyle="--",
        label="3RMW"
    )

    ax.set_xlim(0, 280)
    ax.set_ylim(0.5, 18)

    ax.set_xlabel("Radius (km)", fontsize=9)
    ax.set_ylabel("Height (km)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.set_title(
        f"{frame} h",
        fontsize=14
    )

    # 避免每个 panel 都有 legend 太乱，只给第一个加
    if i == 0:
        ax.legend(fontsize=8, loc="upper right")


# 多余 panel 隐藏
for j in range(len(frames), len(axes)):
    axes[j].axis("off")


fig.colorbar(
    mappable,
    ax=axes,
    orientation="vertical",
    pad=0.02,
    label="Diabatic heating (K/s)",
    fraction=0.02,
    aspect=30
)

fig.subplots_adjust(
    top=0.88,
    bottom=0.08,
    hspace=0.35,
    right=0.84
)

fig.suptitle(
    "Weak TC group example radial-height diabatic heating",
    fontsize=20,
    y=0.93
)

plt.show()

ds.close()

path = "wrf_ideal_12mshear_restart_at_96h_28sst_minsetup_largeTC_d2.nc"
ds = Dataset(path)

#frames = np.arange(0, 24, 1)
#fig, axes = plt.subplots(4, 6, figsize=(26, 20))

frames = [1,6,9,13,16,19]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes = axes.ravel()

mappable = None

# 你的 d02 网格间距
dx_km = 3.0

# TC center
x0 = 400
y0 = 400

# 半径范围，单位 km
r_bins_km = np.arange(0, 301, 5)

# 高度：沿用你原来的 reference_height 写法
reference_height = (
    (wrf.getvar(ds, "PH", timeidx=1) + wrf.getvar(ds, "PHB", timeidx=1))
    / 9.81
    / 1000.0
)[:, 1, 1]

# 先计算 RMW
rmw10_dict = {}

for frame in frames:
    rmw_this_frame = direct_rmw_at_frame(
        path,
        frame,
        x0=x0,
        y0=y0
    )
    rmw10_dict[frame] = rmw_this_frame


for i, frame in enumerate(frames):
    print(frame)
    ax = axes[i]

    # =====================================================
    # 读取 3D diabatic heating
    # =====================================================

    diabatic_3d = get_diabatic_3d_at_frame(ds, frame)

    # PH/PHB 是 w-level，通常比 diabatic 多一层
    # 转成 mass-level height
    if len(reference_height) == diabatic_3d.shape[0] + 1:
        height_km = 0.5 * (reference_height[:-1] + reference_height[1:])
    else:
        height_km = reference_height

    # =====================================================
    # 计算 radius-height azimuthal mean
    # =====================================================

    r_centers, diabatic_rz = radial_height_mean(
        diabatic_3d,
        x0=x0,
        y0=y0,
        dx_km=dx_km,
        r_bins_km=r_bins_km
    )

    # =====================================================
    # 画 radial-height plot
    # =====================================================

    R, Z = np.meshgrid(r_centers, height_km)

    mappable = ax.contourf(
        R,
        Z,
        diabatic_rz,
        levels=np.linspace(-0.005, 0.020, 26),
        cmap="RdBu_r",
        extend="both"
    )

    # 画 RMW 和 3RMW
    rmw_grid = rmw10_dict[frame]
    rmw_km = rmw_grid * dx_km

    ax.axvline(
        rmw_km,
        color="yellow",
        linewidth=1.6,
        linestyle="-",
        label="RMW"
    )

    ax.axvline(
        3 * rmw_km,
        color="magenta",
        linewidth=1.6,
        linestyle="--",
        label="3RMW"
    )

    ax.set_xlim(0, 280)
    ax.set_ylim(0.5, 18)

    ax.set_xlabel("Radius (km)", fontsize=9)
    ax.set_ylabel("Height (km)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.set_title(
        f"{frame} h",
        fontsize=14
    )

    # 避免每个 panel 都有 legend 太乱，只给第一个加
    if i == 0:
        ax.legend(fontsize=8, loc="upper right")


# 多余 panel 隐藏
for j in range(len(frames), len(axes)):
    axes[j].axis("off")


fig.colorbar(
    mappable,
    ax=axes,
    orientation="vertical",
    pad=0.02,
    label="Diabatic heating (K/s)",
    fraction=0.02,
    aspect=30
)

fig.subplots_adjust(
    top=0.88,
    bottom=0.08,
    hspace=0.35,
    right=0.84
)

fig.suptitle(
    "Strong TC group example radial-height diabatic heating",
    fontsize=20,
    y=0.93
)

plt.show()

ds.close()

path = "wrf_ideal_12mshear_restart_at_60h_28sst_minsetup_largeTC_d2.nc"
#frames = [0,5,7,8,9,11,12,13,18,19]
frames = [0,9,18]

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

axes = axes.ravel()

vmin, vmax = -40, 40
mappable = None

consider_level=23

for i, frame in enumerate(frames):
    ax = axes[i]
    
    U_level,V_level = get_uv_vector_top_view_certain_level(path, frame, None, consider_level)
    U_r,V_r,vr,ix,iy = get_vr_top_view_certain_level(path, frame, np.arange(1,150), consider_level)

    # 只画局部区域
    data = vr[300:500, 300:500]
    u2d = U_level[300:500, 300:500]
    v2d = V_level[300:500, 300:500]

    skip = 8
    u_q = u2d[::skip, ::skip]
    v_q = v2d[::skip, ::skip]
    
    

    mappable = ax.pcolormesh(data, shading="auto", vmin=vmin, vmax=vmax,cmap="RdBu_r")

    ny, nx = u2d.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    X_q = X[::skip, ::skip]
    Y_q = Y[::skip, ::skip]

    '''
    ax.quiver(X_q, Y_q, u_q, v_q,
          angles="xy", scale_units="xy", scale=1.0,  # scale 你可能要调
          width=0.002, alpha=0.8)

    '''

    
    
    ax.set_title(f"{frame} h",fontsize=8)  # 标题只要数字

    # 可选：不显示坐标刻度，让图更清爽
    tick_pos = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    tick_lab = np.array([0, 75, 150, 225, 300, 375, 450, 525, 600])-300

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
    ax.tick_params(axis='both', labelsize=8)

    ax.set_xlabel('x (km)', fontsize=8)
    ax.set_ylabel('y (km)', fontsize=8)
    #ax.axvline(100, linestyle="--", linewidth=1.0, color="red")
    #ax.axhline(100, linestyle="--", linewidth=1.0, color="red")

# 共享 colorbar（用最后一次的 mappable 即可）

fig.colorbar(mappable, ax=axes, orientation="vertical", pad=0.02, label="Radial flow (m/s)", fraction=0.02,aspect=30 )
fig.subplots_adjust(hspace=0.2,right=0.84)
fig.suptitle("Weaker cyclone: radial flow at 14.5 km", fontsize=12,y=1.02)

plt.show()


path = "wrf_ideal_12mshear_restart_at_96h_28sst_minsetup_largeTC_d2.nc"
#frames = [0,5,7,8,9,11,12,13,18,19]
frames = [0,9,18]

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

axes = axes.ravel()

vmin, vmax = -40, 40
mappable = None

consider_level=23

for i, frame in enumerate(frames):
    ax = axes[i]
    
    U_level,V_level = get_uv_vector_top_view_certain_level(path, frame, None, consider_level)
    U_r,V_r,vr,ix,iy = get_vr_top_view_certain_level(path, frame, np.arange(1,150), consider_level)

    # 只画局部区域
    data = vr[300:500, 300:500]
    u2d = U_level[300:500, 300:500]
    v2d = V_level[300:500, 300:500]

    skip = 8
    u_q = u2d[::skip, ::skip]
    v_q = v2d[::skip, ::skip]
    
    

    mappable = ax.pcolormesh(data, shading="auto", vmin=vmin, vmax=vmax,cmap="RdBu_r")

    ny, nx = u2d.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    X_q = X[::skip, ::skip]
    Y_q = Y[::skip, ::skip]

    '''
    ax.quiver(X_q, Y_q, u_q, v_q,
          angles="xy", scale_units="xy", scale=1.0,  # scale 你可能要调
          width=0.002, alpha=0.8)

    '''

    
    
    ax.set_title(f"{frame} h",fontsize=8)  # 标题只要数字

    # 可选：不显示坐标刻度，让图更清爽
    tick_pos = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    tick_lab = np.array([0, 75, 150, 225, 300, 375, 450, 525, 600])-300

    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
    ax.tick_params(axis='both', labelsize=8)

    ax.set_xlabel('x (km)', fontsize=8)
    ax.set_ylabel('y (km)', fontsize=8)
    #ax.axvline(100, linestyle="--", linewidth=1.0, color="red")
    #ax.axhline(100, linestyle="--", linewidth=1.0, color="red")

# 共享 colorbar（用最后一次的 mappable 即可）

fig.colorbar(mappable, ax=axes, orientation="vertical", pad=0.02, label="Radial flow (m/s)", fraction=0.02,aspect=30 )
fig.subplots_adjust(hspace=0.2,right=0.84)
fig.suptitle("Stronger cyclone: radial flow at 14.5 km", fontsize=12,y=1.02)

plt.show()






























































