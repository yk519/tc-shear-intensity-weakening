from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import pandas as pd

ds = Dataset("IBTrACS.ALL.v04r01.nc")

def analyze_ibtracs_wind(ds):
    """
    分析 IBTrACS 中台风风速的可用性情况。
    输出：
    1. 哪些台风有风速数据（任意机构）
    2. 哪些台风完全没有风速数据
    3. 每个台风的有效风速数量
    4. 每个台风的风速来源机构
    """



    # 各机构风速变量（可能并非所有都存在，按实际文件检查）
    wind_sources = [
        "usa_wind", "tokyo_wind", "cma_wind", "wmo_wind",
        "hko_wind", "newdelhi_wind", "bom_wind"
    ]

    available_sources = [w for w in wind_sources if w in ds.variables]

    n_storm = len(ds["sid"])
    results = []

    for i in range(n_storm):
        storm_sid = ds["sid"][i].tobytes().decode("utf-8").strip()
        storm_name = ds["name"][i].tobytes().decode("utf-8").strip()

        storm_result = {
            "index": i,
            "sid": storm_sid,
            "name": storm_name,
            "sources": [],
            "valid_count": 0
        }

        valid_total = 0
        sources_used = []

        # 遍历每个机构
        for src in available_sources:
            arr = ds[src][i, :]

            # masked → 缺测
            if hasattr(arr, "mask"):
                valid_mask = ~arr.mask
            else:
                valid_mask = ~np.isnan(arr)

            valid_n = np.sum(valid_mask)

            if valid_n > 0:         # 该机构提供了风速
                sources_used.append(src)
                valid_total += valid_n

        storm_result["sources"] = sources_used
        storm_result["valid_count"] = valid_total

        results.append(storm_result)

    # 分类统计
    storms_with_wind = [r for r in results if r["valid_count"] > 0]
    storms_without_wind = [r for r in results if r["valid_count"] == 0]

    print("==============================================")
    print(f"总台风数: {n_storm}")
    print(f"有风速记录的台风数量: {len(storms_with_wind)}")
    print(f"完全没有风速记录的台风数量: {len(storms_without_wind)}")
    print("==============================================")
    
    print("\n📌 有风速记录的台风示例（前 10 个）：")
    for r in storms_with_wind[:10]:
        print(f"{r['sid']:15s}  {r['name']:15s}  valid={r['valid_count']:3d}  sources={r['sources']}")

    print("\n⚠️ 完全没有风速记录的台风示例（前 10 个）：")
    for r in storms_without_wind[:10]:
        print(f"{r['sid']:15s}  {r['name']:15s}")

    return results, storms_with_wind, storms_without_wind



#################


def decode_char_array(arr):
    """用于解码 IBTrACS 的字符数组"""
    return "".join(x.decode("utf-8") for x in arr if isinstance(x, bytes)).strip()


#########################

def show_storm_info(ds, i=None, sid=None):
    """
    显示单个台风的详细信息
    ds  : Dataset("IBTrACS....nc")
    i   : 风暴序号（storm index）
    sid : 台风的唯一 ID（例如 '2019186N09123'）
    """

    # 1. 查找 storm index
    if sid is not None:
        all_sid = ["".join(c.tobytes().decode("utf-8").strip()) 
                   for c in ds["sid"][:]]
        if sid in all_sid:
            i = all_sid.index(sid)
        else:
            print(f"找不到 storm ID: {sid}")
            return

    if i is None:
        print("请提供 storm index i= 或唯一 sid=")
        return
    
    # 2. 获取基本字段
    storm_sid  = decode_char_array(ds["sid"][i])
    storm_name = decode_char_array(ds["name"][i])
    basin      = decode_char_array(ds["basin"][i])

    lat  = ds["lat"][i, :]
    lon  = ds["lon"][i, :]
    time_raw = ds["time"][i, :]

    # 转换时间
    try:
        times = num2date(time_raw, ds["time"].units)
    except:
        times = time_raw  # 某些版本不带 units

    # 3. 支持的风速来源
    wind_sources = [
        "usa_wind", "tokyo_wind", "cma_wind", "wmo_wind",
        "hko_wind", "newdelhi_wind", "bom_wind"
    ]
    wind_sources = [w for w in wind_sources if w in ds.variables]

    wind_info = {}

    for src in wind_sources:
        arr = ds[src][i, :]
        # masked → 缺失
        if hasattr(arr, "mask"):
            valid = arr[~arr.mask]
        else:
            valid = arr[~np.isnan(arr)]

        if len(valid) > 0:
            max_kt = np.max(valid)
            max_ms = max_kt * 0.514444
            wind_info[src] = (max_kt, max_ms)

    # 4. 轨迹点数量
    valid_lat = lat[~lat.mask]
    n_points = len(valid_lat)

    # 5. 输出信息
    print("======================================")
    print(f" Storm Index   : {i}")
    print(f" Storm SID     : {storm_sid}")
    print(f" Storm Name    : {storm_name}")
    print(f" Basin         : {basin}")
    print("--------------------------------------")
    print(f" Track Points  : {n_points}")
    if n_points > 0:
        print(f" Start Time    : {times[0]}")
        print(f" End Time      : {times[n_points-1]}")
    print("--------------------------------------")
    print(" Wind Sources with Data:")
    for src, (kt, ms) in wind_info.items():
        print(f"  - {src:<12}  max = {kt:5.1f} kt  ({ms:5.1f} m/s)")
    print("--------------------------------------")
    print(" Variables available for this storm:")
    print("  lat, lon, time, wind (multi-agency), pressure, basin…")
    print("======================================")

    return {
        "index": i,
        "sid": storm_sid,
        "name": storm_name,
        "basin": basin,
        "wind_info": wind_info,
        "lat": lat,
        "lon": lon,
        "time": times,
        "n_points": n_points
    }


#######################

def find_cat4_cat5_storms(ds):
    """
    在 IBTrACS 数据中查找 Cat4 和 Cat5 台风
    使用多机构风速（usa, wmo, tokyo, cma 等）
    返回两个列表：cat4_list, cat5_list
    """

    # 机构风速优先顺序（可改动）
    wind_sources = [
        "usa_wind", "wmo_wind", "tokyo_wind", "cma_wind",
        "hko_wind", "newdelhi_wind", "bom_wind"
    ]
    wind_sources = [w for w in wind_sources if w in ds.variables]

    n_storm = len(ds["sid"])

    cat4 = []
    cat5 = []

    for i in range(n_storm):

        # 找这个台风的最大风速（跨所有机构）
        peak_kt = None

        for src in wind_sources:
            arr = ds[src][i, :]

            # masked array → 去除缺测
            if hasattr(arr, "mask"):
                valid = arr[~arr.mask]
            else:
                valid = arr[~np.isnan(arr)]

            if len(valid) > 0:
                max_this = np.max(valid)
                if (peak_kt is None) or (max_this > peak_kt):
                    peak_kt = max_this

        if peak_kt is None:
            continue  # 没有风速资料

        # 分类
        if peak_kt >= 137:    # Cat5
            cat5.append((i, peak_kt))
        elif peak_kt >= 115:  # Cat4
            cat4.append((i, peak_kt))

    # 格式化输出
    def format_list(lst):
        out = []
        for i, w in lst:
            sid = "".join(ds["sid"][i].tobytes().decode("utf-8").strip())
            name = "".join(ds["name"][i].tobytes().decode("utf-8").strip())
            out.append({"index": i, "sid": sid, "name": name, "peak_wind": w})
        return out

    return format_list(cat4), format_list(cat5)

def classify_storms_by_category(ds):
    # 支持的风速来源（按优先级）
    wind_sources = [
        "usa_wind"
    ]
    wind_sources = [w for w in wind_sources if w in ds.variables]

    n_storm = len(ds["sid"])
    peak_intensities = []

    for i in range(n_storm):
        peak = None

        for src in wind_sources:
            arr = ds[src][i, :]

            # masked array → 去缺测
            if hasattr(arr, "mask"):
                valid = arr[~arr.mask]
            else:
                valid = arr[~np.isnan(arr)]

            if len(valid) > 0:
                max_this = np.max(valid)
                if peak is None or max_this > peak:
                    peak = max_this

        if peak is not None:
            peak_intensities.append(peak)

    peak_intensities = np.array(peak_intensities)

    # Category bins (upper-stream)
    bins = [64, 83, 96, 113, 137, 200]  # Cat1–5
    labels = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"]

    categories = np.digitize(peak_intensities, bins, right=False)

    # Count categories
    counts = [np.sum(categories == i+1) for i in range(len(labels))]

    return peak_intensities, counts, labels


def plot_category_histogram(counts, labels):
    plt.figure(figsize=(8,5))
    plt.bar(labels, counts, color="royalblue")
    plt.xlabel("Storm Category (Saffir–Simpson)")
    plt.ylabel("Number of Storms")
    plt.title("IBTrACS Global Tropical Cyclone Category Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.show()




###############################

def get_storm_sizes_by_category(ds, categories=[4,5], size_var="usa_r34", method="mean"):
    """
    返回指定 category 的所有台风的大小（如 r34）
    
    categories: list，例如 [4,5] 想查看 Cat4 + Cat5
    size_var: 'r34', 'r50', 'r64', 'roci', 'rmw'
    method: 合成方法
            - 'mean' : 四象限平均
            - 'max'  : 四象限最大
            - 'life_mean': 台风生命周期平均
            - 'life_max' : 生命周期最大
    """

    # 台风风速来源
    wind_sources = [s for s in [
        "usa_wind", "wmo_wind", "tokyo_wind", "cma_wind",
        "hko_wind", "newdelhi_wind", "bom_wind"
    ] if s in ds.variables]

    bins = [0,64,83,96,113,137,200]  # Cat1-5
    n_storm = len(ds["sid"])

    size_values = []

    for i in range(n_storm):
        # ---- 获取 peak wind ----
        peak = None
        for src in wind_sources:
            arr = ds[src][i,:]
            valid = arr[~arr.mask] if hasattr(arr,"mask") else arr[~np.isnan(arr)]
            if len(valid) > 0:
                max_val = np.max(valid)
                if peak is None or max_val > peak:
                    peak = max_val
        if peak is None:
            continue

        # ---- 计算类别 ----
        storm_cat = np.digitize([peak], bins)[0]  # 1~5

        # ---- 如果属于筛选的 category ----
        if storm_cat in categories:
            sv = ds[size_var][i]  # (time, quad) OR (time)
            
            # ---- 合成风圈大小 ----
            if size_var in ["r34","r50","r64"]:
                # 四象限
                if method == "mean":
                    v = np.nanmean(sv)
                elif method == "max":
                    v = np.nanmax(sv)
                elif method == "life_mean":
                    v = np.nanmean(np.nanmean(sv, axis=1))
                elif method == "life_max":
                    v = np.nanmax(np.nanmean(sv, axis=1))
            
            else:
                # 单一变量：roci, rmw
                if method in ["mean","life_mean"]:
                    v = np.nanmean(sv)
                elif method in ["max","life_max"]:
                    v = np.nanmax(sv)

            size_values.append(v)

    return np.array(size_values)

def extract_radius_value(arr, method="max"):
    """
    arr shape: (time, quadrant) or (time)
    method: 'max', 'mean', 'life_mean', 'life_max'
    """
    # 去除缺测值
    if hasattr(arr, "mask"):
        valid = arr[~arr.mask]
    else:
        valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return None

    if method == "max":
        return np.nanmax(valid)
    elif method == "mean":
        return np.nanmean(valid)
    elif method == "life_mean":
        return np.nanmean(valid)
    elif method == "life_max":
        return np.nanmax(valid)



def get_radius_list(ds, categories, target_radius, method="life_max"):

    # Category boundaries (kt)
    bins = [0,64,83,96,113,137,400]

    # 1. 找风速来源（多机构）
    wind_sources = [
        "usa_wind"
    ]
    wind_sources = [w for w in wind_sources if w in ds.variables]

    # 2. 找所有 target radius 变量
    radius_vars = [target_radius]

    n_storm = len(ds["sid"])
    size_list = []

    for i in range(n_storm):

        # ===== 找 peak wind =====
        peak = None
        for src in wind_sources:
            arr = ds[src][i,:]
            valid = arr[~arr.mask] if hasattr(arr,"mask") else arr[~np.isnan(arr)]
            if len(valid) > 0:
                mx = np.max(valid)
                if peak is None or mx > peak:
                    peak = mx
        if peak is None:
            continue

        # ===== 计算类别 =====
        storm_cat = np.digitize([peak], bins)[0]

        # ===== 不在筛选范围 =====
        if storm_cat not in categories:
            continue

        # ===== 寻找所有机构的 radius SIZE =====
        for rvar in radius_vars:
            arr = ds[rvar][i,:]   # shape may be (time) or (time, quad)
            val = extract_radius_value(arr, method=method)
            if val is not None:
                size_list.append(val)

    return np.array(size_list)




def get_cat_sids(ds,target_cat):
    bins = [0,64,83,96,113,137,400]   # knots

    sids = []
    n = len(ds["sid"])
    usa = ds["usa_wind"]

    for i in range(n):

        arr = usa[i,:]
        valid = arr[~arr.mask] if hasattr(arr,"mask") else arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue

        peak = np.max(valid)
        cat = np.digitize([peak], bins)[0]-1

        if cat in target_cat:
            
            id_bytes = ds["usa_atcf_id"][i]
            sid = None
            for t in range(id_bytes.shape[0]):
                chars = id_bytes[t]

                s = decode_atcf(chars)  # <--- 使用新的 decode
                if s != "":
                    sid = s
                    break

            if sid:
                sids.append(sid)

    return sids


def find_ibtracs_index_from_sid(sid_to_index, sid):
    return sid_to_index.get(sid, None)

def get_r34_for_sid(ds, sid, sid_to_index, method="life_max"):

    idx = find_ibtracs_index_from_sid(sid_to_index, sid)
    if idx is None:
        return None

    arr = ds["usa_r34"][idx,:,:]
    return extract_radius_value(arr, method=method)

def get_rmw_for_sid(ds, sid, sid_to_index, method="life_max"):

    idx = find_ibtracs_index_from_sid(sid_to_index, sid)
    if idx is None:
        return None

    arr = ds["usa_rmw"][idx,:]
    return extract_radius_value(arr, method=method)


def build_sid_index_map(ds):
    sid_map = {}
    ids = ds["usa_atcf_id"][:]   # (storm, time, char)

    for i in range(len(ids)):
        row = ids[i]
        sid = None

        for t in range(row.shape[0]):
            decoded = decode_atcf(row[t])
            if decoded != "":
                sid = decoded
                break

        if sid:
            sid_map[sid] = i

    return sid_map



def decode_atcf(chars):
    raw = b"".join(chars).split(b'\x00')[0]
    return raw.decode().strip()

def find_ships_entry_directly(sid, ships_file="ships_5day/SHIPS_5day_ALL.txt"):
    matches = []
    with open(ships_file, "r") as f:
        for lineno, line in enumerate(f):
            if sid in line:
                matches.append((lineno, line.strip()))
    return matches

def find_ships_entry_in_function(sid, all_lines):
    matches = []
    for lineno, line in enumerate(all_lines):
        if sid in line:
            matches.append((lineno, line.strip()))
    return matches

SEPARATOR = "9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 LAST"

def extract_block_from_entry(lineno, all_lines, separator=SEPARATOR):
    block = []
    for i in range(lineno, len(all_lines)):
        if separator in all_lines[i]:
            break
        block.append(all_lines[i].rstrip("\n"))
    return block



def get_ships_blocks_for_sid(sid, all_lines):
    entries = find_ships_entry_in_function(sid, all_lines)
    blocks = []

    for lineno, _ in entries:
        block = extract_block_from_entry(lineno, all_lines)
        blocks.append(block)

    return blocks


def extract_shrd_from_block(block):
    for line in block:
        if "SHRD" in line:          # 找到 SHRD 所在的行
            cols = line.split()

            # SHRD 行格式大致如下：
            #   251 216 186 221 243 ... SHRD
            #   ↑   ↑   ↑
            #  -12 -6  0   <--- 你要第3列（索引2）

            if len(cols) >= 3:
                try:
                    val = float(cols[2])   # time=0 列
                    if val == 9999:
                        return None        # 忽略缺测
                    return val
                except:
                    return None

    return None

ds = Dataset("IBTrACS.ALL.v04r01.nc")
sid_to_index = build_sid_index_map(ds)
cat345_sids = get_cat_sids(ds,[3,4,5])
print("Total Cat3+4+5:", len(cat345_sids))

cat12_sids = get_cat_sids(ds,[0,1,2])
print("Total Cat0+1+2:", len(cat12_sids))

with open("ships_5day/SHIPS_5day_ALL.txt", "r") as f:
    all_lines = f.readlines()

sid_to_shrd345 = {}

total345 = len(cat345_sids)

for idx, sid in enumerate(cat345_sids, start=1):

    print(f"Processing {idx}/{total345}: {sid}")

    blocks = get_ships_blocks_for_sid(sid,all_lines)

    shrd_list = []
    for block in blocks:
        val = extract_shrd_from_block(block)
        if val is not None:
            shrd_list.append(val)

    if len(shrd_list) == 0:
        sid_to_shrd345[sid] = None
    else:
        sid_to_shrd345[sid] = float(np.nanmean(shrd_list))   # 多个 block 平均
    print(sid_to_shrd345[sid])


sid_to_radius345 = {}

for idx, sid in enumerate(cat345_sids, start=1):
    print(f"Processing {idx}/{total345}: {sid}")
    radius = get_rmw_for_sid(ds, sid, sid_to_index)
    sid_to_radius345[sid] = radius

sid_to_shrd12 = {}

total12 = len(cat12_sids)

for idx, sid in enumerate(cat12_sids, start=1):

    print(f"Processing {idx}/{total12}: {sid}")

    blocks = get_ships_blocks_for_sid(sid,all_lines)

    shrd_list = []
    for block in blocks:
        val = extract_shrd_from_block(block)
        if val is not None:
            shrd_list.append(val)

    if len(shrd_list) == 0:
        sid_to_shrd12[sid] = None
    else:
        sid_to_shrd12[sid] = float(np.nanmean(shrd_list))   # 多个 block 平均
    print(sid_to_shrd12[sid])


sid_to_radius12 = {}

for idx, sid in enumerate(cat12_sids, start=1):
    print(f"Processing {idx}/{total12}: {sid}")
    radius = get_rmw_for_sid(ds, sid, sid_to_index)
    sid_to_radius12[sid] = radius


sid_to_r34_12 = {}

for idx, sid in enumerate(cat12_sids, start=1):
    print(f"Processing {idx}/{total12}: {sid}")
    radius = get_r34_for_sid(ds, sid, sid_to_index)
    sid_to_r34_12[sid] = radius

sid_to_r34_345 = {}

for idx, sid in enumerate(cat345_sids, start=1):
    print(f"Processing {idx}/{total345}: {sid}")
    radius = get_r34_for_sid(ds, sid, sid_to_index)
    sid_to_r34_345[sid] = radius

none_sids345 = [sid for sid, v in sid_to_radius345.items() if v is None]
sid_to_radius_clean345 = {sid: v for sid, v in sid_to_radius345.items() if v is not None}
sid_to_shrd_clean345 = {sid: sid_to_shrd345[sid] 
                     for sid in sid_to_radius_clean345.keys()
                     if sid_to_shrd345.get(sid) is not None}

valid_sids345 = set(sid_to_radius_clean345.keys()) & set(sid_to_shrd_clean345.keys())

none_sids12 = [sid for sid, v in sid_to_radius12.items() if v is None]
sid_to_radius_clean12 = {sid: v for sid, v in sid_to_radius12.items() if v is not None}
sid_to_shrd_clean12 = {sid: sid_to_shrd12[sid] 
                     for sid in sid_to_radius_clean12.keys()
                     if sid_to_shrd12.get(sid) is not None}

valid_sids12 = set(sid_to_radius_clean12.keys()) & set(sid_to_shrd_clean12.keys())

none_r34_sids345 = [sid for sid, v in sid_to_r34_345.items() if v is None]
sid_to_r34_clean345 = {sid: v for sid, v in sid_to_r34_345.items() if v is not None}
sid_to_shrd_clean345 = {sid: sid_to_shrd345[sid] 
                     for sid in sid_to_r34_clean345.keys()
                     if sid_to_shrd345.get(sid) is not None}

valid_r34_sids345 = set(sid_to_r34_clean345.keys()) & set(sid_to_shrd_clean345.keys())


none_r34_sids12 = [sid for sid, v in sid_to_r34_12.items() if v is None]
sid_to_r34_clean12 = {sid: v for sid, v in sid_to_r34_12.items() if v is not None}
sid_to_shrd_clean12 = {sid: sid_to_shrd12[sid] 
                     for sid in sid_to_r34_clean12.keys()
                     if sid_to_shrd12.get(sid) is not None}

valid_r34_sids12 = set(sid_to_r34_clean12.keys()) & set(sid_to_shrd_clean12.keys())

def decode_time(chars):
    raw = b"".join(chars).split(b'\x00')[0]
    return raw.decode().strip()


def decode_str(chars):
    raw = b"".join(chars).split(b'\x00')[0]
    return raw.decode().strip()

def has_landfall_within_24h(ds, idx, lmi_tidx, hours=24, mode="after"):
    """
    判断某个 storm 在 LMI 附近是否发生 landfall。

    这里使用 IBTrACS 的 landfall 变量定义：
    landfall == 0 表示在当前观测到下一个观测之间发生了登陆/穿岸。

    mode:
        "before" -> [LMI-24h, LMI]
        "both"   -> [LMI-24h, LMI+24h]
        "after"  -> [LMI, LMI+24h]
    """
    lmi_time_str = decode_time(ds["iso_time"][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        return False

    arr = ds["landfall"][idx, :]
    nt = arr.shape[0]

    for t in range(nt):
        # 跳过缺测的 landfall
        if hasattr(arr, "mask") and arr.mask[t]:
            continue

        time_str = decode_time(ds["iso_time"][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        dt_hours = (t_time - lmi_time).total_seconds() / 3600.0

        if mode == "before":
            in_window = (-hours <= dt_hours <= 0)
        elif mode == "both":
            in_window = (-hours <= dt_hours <= hours)
        elif mode == "after":
            in_window = (0 <= dt_hours <= hours)
        else:
            raise ValueError("mode must be 'before', 'both', or 'after'")

        if not in_window:
            continue

        # landfall == 0 才视为登陆
        if float(arr[t]) == 0:
            return True

    return False


def get_lmi_info_for_sid(ds, sid, sid_to_index,
                         wind_var="usa_wind", time_var="iso_time"):
    idx = find_ibtracs_index_from_sid(sid_to_index, sid)
    if idx is None:
        return None

    wind = ds[wind_var][idx, :]

    if hasattr(wind, "mask"):
        valid_mask = ~wind.mask
    else:
        valid_mask = ~np.isnan(wind)

    if np.sum(valid_mask) == 0:
        return None

    wind_filled = np.array(wind, dtype=float)
    wind_filled[~valid_mask] = -np.inf

    max_wind = np.max(wind_filled)
    if np.isneginf(max_wind):
        return None

    max_indices = np.where(np.isclose(wind_filled, max_wind))[0]
    lmi_tidx = int(max_indices[-1])

    time_str = decode_time(ds[time_var][idx, lmi_tidx])
    lmi_time = pd.to_datetime(time_str, errors="coerce")

    return {
        "sid": sid,
        "idx": idx,
        "lmi_tidx": lmi_tidx,
        "lmi_wind": float(max_wind),
        "lmi_time_str": time_str,
        "lmi_time": lmi_time
    }


#######################################################################################################


def extract_shrd_24h_stats_from_block(block):
    """
    从 SHIPS block 的 SHRD 行中提取 0-24h 的 shear，
    返回 mean 和 max（单位：m/s）。

    SHIPS TIME 列通常对应：
    -12, -6, 0, 6, 12, 18, 24, ...

    所以：
    cols[2] -> 0h
    cols[3] -> 6h
    cols[4] -> 12h
    cols[5] -> 18h
    cols[6] -> 24h
    """
    for line in block:
        if "SHRD" in line:
            cols = line.split()

            # 至少要能取到 24h 那一列
            if len(cols) >= 7:
                vals = []
                for i in range(2, 7):   # 0h, 6h, 12h, 18h, 24h
                    try:
                        v = float(cols[i])
                        if v != 9999:
                            vals.append(v / 20.0)   # 与 shear_0h 一样，转成 m/s
                    except ValueError:
                        continue

                if len(vals) == 0:
                    return {
                        "shear_24h_mean": np.nan,
                        "shear_24h_max": np.nan
                    }

                return {
                    "shear_24h_mean": float(np.mean(vals)),
                    "shear_24h_max": float(np.max(vals))
                }

    return {
        "shear_24h_mean": np.nan,
        "shear_24h_max": np.nan
    }

SEPARATOR = "9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 9999 LAST"


def parse_ships_head_line(line):
    parts = line.split()
    if len(parts) < 8:
        return None

    return {
        "basin": parts[0],
        "date_str": parts[1],   # YYMMDD
        "hour_str": parts[2],   # HH
        "sid": parts[7]
    }


def build_ships_block_map(ships_file="ships_5day/SHIPS_5day_ALL.txt", separator=SEPARATOR):
    with open(ships_file, "r") as f:
        all_lines = [line.rstrip("\n") for line in f]

    ships_map = {}
    i = 0
    n = len(all_lines)

    while i < n:
        line = all_lines[i]

        if "HEAD" in line:
            head_info = parse_ships_head_line(line)
            if head_info is not None:
                sid = head_info["sid"]
                timekey = f"{head_info['date_str']}{head_info['hour_str']}"

                block = []
                j = i
                while j < n and separator not in all_lines[j]:
                    block.append(all_lines[j])
                    j += 1

                ships_map[(sid, timekey)] = block
                i = j + 1
                continue

        i += 1

    return ships_map

def ibtracs_time_to_ships_timekey(ts):
    if pd.isna(ts):
        return None
    return ts.strftime("%y%m%d%H")

def get_r34_at_lmi_for_sid(ds, sid, sid_to_index,
                           wind_var="usa_wind", time_var="iso_time",
                           reduce_method="mean"):
    lmi_info = get_lmi_info_for_sid(ds, sid, sid_to_index,
                                    wind_var=wind_var, time_var=time_var)
    if lmi_info is None:
        return None

    idx = lmi_info["idx"]
    t = lmi_info["lmi_tidx"]

    arr = ds["usa_r34"][idx, t, :]

    if hasattr(arr, "mask"):
        valid = arr[~arr.mask]
    else:
        valid = arr[~np.isnan(arr)]

    if len(valid) == 0:
        r34_mean = np.nan
        r34_max = np.nan
    else:
        r34_mean = float(np.nanmean(valid)) * 1.852     # nmi → km  
        r34_max = float(np.nanmax(valid)) * 1.852     # nmi → km  

    if reduce_method == "mean":
        r34_value = r34_mean
    elif reduce_method == "max":
        r34_value = r34_max
    else:
        raise ValueError("reduce_method must be 'mean' or 'max'")

    return {
        **lmi_info,
        "r34_value": r34_value,
        "r34_mean": r34_mean,
        "r34_max": r34_max
    }


def get_lmi_r34_shear_for_sid(ds, sid, sid_to_index, ships_map,
                              wind_var="usa_wind", time_var="iso_time",
                              reduce_method="mean",exclude_non_tropical=False,require_cat1_24h=False):
    """
    对一个 sid：
    1. 在 IBTrACS 中求 LMI
    2. 取 LMI 时次的 R34
    3. 把 LMI 时间转成 SHIPS timekey
    4. 在 SHIPS 中严格匹配同一时刻的 block
    5. 提取该 block 的 0h shear
    """
    lmi_info = get_lmi_info_for_sid(ds, sid, sid_to_index,
                                    wind_var=wind_var, time_var=time_var)
    if lmi_info is None:
        return None
        
    # 先排除 landfall case
    if has_landfall_within_24h(ds,
                               idx=lmi_info["idx"],
                               lmi_tidx=lmi_info["lmi_tidx"],
                               hours=24):
        return None

    if exclude_non_tropical:
        if has_non_tropical_within_24h(ds,
                                       idx=lmi_info["idx"],
                                       lmi_tidx=lmi_info["lmi_tidx"],
                                       hours=24):
            return None

    if require_cat1_24h:
        if has_wind_below_threshold_within_24h(ds,
                                               idx=lmi_info["idx"],
                                               lmi_tidx=lmi_info["lmi_tidx"],
                                               wind_var=wind_var,
                                               time_var=time_var,
                                               hours=24,
                                               threshold=33.0,
                                               mode="after"):
            return None


    
    r34_info = get_r34_at_lmi_for_sid(ds, sid, sid_to_index,
                                      wind_var=wind_var, time_var=time_var,
                                      reduce_method=reduce_method)
    if r34_info is None:
        return None

    timekey = ibtracs_time_to_ships_timekey(lmi_info["lmi_time"])
    if timekey is None:
        return None

    block = ships_map.get((sid, timekey), None)
    if block is None:
        return None

    shrd = extract_shrd_from_block(block)
    if shrd is None:
        return None

    shrd = shrd/20      # 转成 m/s
    shrd_24h = extract_shrd_24h_stats_from_block(block)

    return {
        "sid": sid,
        "idx": lmi_info["idx"],
        "lmi_tidx": lmi_info["lmi_tidx"],
        "lmi_wind": lmi_info["lmi_wind"] * 0.514444,
        "lmi_time_str": lmi_info["lmi_time_str"],
        "lmi_time": lmi_info["lmi_time"],
        "r34_value": r34_info["r34_value"],
        "r34_mean": r34_info["r34_mean"],
        "r34_max": r34_info["r34_max"],
        "shear_0h": shrd,
        "shear_24h_mean": shrd_24h["shear_24h_mean"],
        "shear_24h_max": shrd_24h["shear_24h_max"]
    }


def get_lmi_shear_for_sid(ds, sid, sid_to_index, ships_map,
                          wind_var="usa_wind", time_var="iso_time",
                          exclude_non_tropical=False,
                          require_cat1_24h=False):
    """
    对一个 sid：
    1. 在 IBTrACS 中求 LMI
    2. 把 LMI 时间转成 SHIPS timekey
    3. 在 SHIPS 中严格匹配同一时刻的 block
    4. 提取该 block 的 0h shear 和 24h shear statistics

    注意：
    不再要求 R34 存在。
    """

    lmi_info = get_lmi_info_for_sid(
        ds, sid, sid_to_index,
        wind_var=wind_var,
        time_var=time_var
    )

    if lmi_info is None:
        return None

    # 仍然保留你之前的 landfall 筛选
    if has_landfall_within_24h(
        ds,
        idx=lmi_info["idx"],
        lmi_tidx=lmi_info["lmi_tidx"],
        hours=24
    ):
        return None

    if exclude_non_tropical:
        if has_non_tropical_within_24h(
            ds,
            idx=lmi_info["idx"],
            lmi_tidx=lmi_info["lmi_tidx"],
            hours=24
        ):
            return None

    if require_cat1_24h:
        if has_wind_below_threshold_within_24h(
            ds,
            idx=lmi_info["idx"],
            lmi_tidx=lmi_info["lmi_tidx"],
            wind_var=wind_var,
            time_var=time_var,
            hours=24,
            threshold=33.0,
            mode="after"
        ):
            return None

    timekey = ibtracs_time_to_ships_timekey(lmi_info["lmi_time"])

    if timekey is None:
        return None

    block = ships_map.get((sid, timekey), None)

    if block is None:
        return None

    shrd = extract_shrd_from_block(block)

    if shrd is None:
        return None

    shrd = shrd / 20.0   # 你原来就是这样转成 m/s

    shrd_24h = extract_shrd_24h_stats_from_block(block)

    return {
        "sid": sid,
        "idx": lmi_info["idx"],
        "lmi_tidx": lmi_info["lmi_tidx"],
        "lmi_wind": lmi_info["lmi_wind"] * 0.514444,  # kt -> m/s
        "lmi_time_str": lmi_info["lmi_time_str"],
        "lmi_time": lmi_info["lmi_time"],
        "shear_0h": shrd,
        "shear_24h_mean": shrd_24h["shear_24h_mean"],
        "shear_24h_max": shrd_24h["shear_24h_max"]
    }




def has_non_tropical_within_24h(ds, idx, lmi_tidx, hours=24, mode="after",
                                bad_types=("ET", "SS", "DS", "NR", "MX")):
    """
    判断某个 storm 在 LMI 附近是否转为非纯热带系统。

    bad_types:
        你想排除的 nature 类型
    mode:
        "before" -> [LMI-24h, LMI]
        "both"   -> [LMI-24h, LMI+24h]
        "after"  -> [LMI, LMI+24h]
    """
    lmi_time_str = decode_time(ds["iso_time"][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        return False

    nt = ds["nature"].shape[1]

    for t in range(nt):
        time_str = decode_time(ds["iso_time"][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        dt_hours = (t_time - lmi_time).total_seconds() / 3600.0

        if mode == "before":
            in_window = (-hours <= dt_hours <= 0)
        elif mode == "both":
            in_window = (-hours <= dt_hours <= hours)
        elif mode == "after":
            in_window = (0 <= dt_hours <= hours)
        else:
            raise ValueError("mode must be 'before', 'both', or 'after'")

        if not in_window:
            continue

        try:
            nature_t = decode_str(ds["nature"][idx, t])
        except:
            continue

        if nature_t in bad_types:
            return True

    return False


def has_wind_below_threshold_within_24h(ds, idx, lmi_tidx,
                                        wind_var="usa_wind",
                                        time_var="iso_time",
                                        hours=24,
                                        threshold=33.0,
                                        mode="after"):
    """
    判断某个 storm 在 LMI 附近的时间窗内，是否出现风速 <= threshold 的时次。

    返回：
    - True  : 发现有任一有效时次 wind <= threshold
    - False : 所有有效时次 wind > threshold

    mode:
        "before" -> [LMI-24h, LMI]
        "both"   -> [LMI-24h, LMI+24h]
        "after"  -> [LMI, LMI+24h]
    """

    lmi_time_str = decode_time(ds[time_var][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        return True   # 时间读不出来时，保守起见当作不满足条件

    nt = ds[time_var].shape[1]
    found_any_valid = False

    for t in range(nt):
        time_str = decode_time(ds[time_var][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        dt_hours = (t_time - lmi_time).total_seconds() / 3600.0

        if mode == "before":
            in_window = (-hours <= dt_hours <= 0)
        elif mode == "both":
            in_window = (-hours <= dt_hours <= hours)
        elif mode == "after":
            in_window = (0 <= dt_hours <= hours)
        else:
            raise ValueError("mode must be 'before', 'both', or 'after'")

        if not in_window:
            continue

        v = get_scalar_value(ds[wind_var][idx, t]) * 0.514444
        if pd.isna(v):
            continue

        found_any_valid = True

        # 如果你的 usa_wind 仍然是 kt，这里就要改 threshold
        # 如果你已经统一成 m/s，则 threshold=35 就直接可用
        if v <= threshold:
            return True

    # 如果窗口内一个有效风速都没找到，保守起见也当作不满足
    if not found_any_valid:
        return True

    return False

def get_scalar_value(x):
    if np.ma.is_masked(x):
        return np.nan
    try:
        return float(x)
    except:
        return np.nan


def get_wind_24h_after_lmi(ds, idx, lmi_tidx,
                           wind_var="usa_wind", time_var="iso_time",
                           target_hours=24, tol_hours=1.5):
    """
    返回 LMI 后24h的风速值，以及对应时间索引
    """
    lmi_time_str = decode_time(ds[time_var][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        return np.nan, None, pd.NaT

    target_time = lmi_time + pd.Timedelta(hours=target_hours)

    nt = ds[time_var].shape[1]
    best_t = None
    best_diff = np.inf
    best_time = pd.NaT

    for t in range(nt):
        time_str = decode_time(ds[time_var][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        diff = abs((t_time - target_time).total_seconds()) / 3600.0
        if diff < best_diff:
            best_diff = diff
            best_t = t
            best_time = t_time

    if best_t is None or best_diff > tol_hours:
        return np.nan, None, pd.NaT

    v24 = get_scalar_value(ds[wind_var][idx, best_t])
    return v24 * 0.514444, best_t, best_time

def calc_decay24(v0, v24, hours=24):
    """
    decay coefficient:
        k = ln(v0 / v24) / 24

    返回:
    - k > 0: weakening
    - k = 0: unchanged
    - k < 0: intensifying
    """
    if pd.isna(v0) or pd.isna(v24):
        return np.nan
    if v0 <= 0 or v24 <= 0:
        return np.nan

    return np.log(v0 / v24) / hours




def get_post_lmi_wind_series(ds, idx, lmi_tidx,
                             wind_var="usa_wind", time_var="iso_time",
                             max_hours=24):
    """
    返回 LMI 之后 max_hours 内的:
    - t_hours: 相对 LMI 的小时数
    - winds: 对应风速（保持 ds 原单位；只要前后一致即可）
    """
    lmi_time_str = decode_time(ds[time_var][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        return np.array([]), np.array([])

    nt = ds[time_var].shape[1]

    t_list = []
    v_list = []

    for t in range(nt):
        time_str = decode_time(ds[time_var][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        dt_hours = (t_time - lmi_time).total_seconds() / 3600.0

        if dt_hours < 0:
            continue
        if dt_hours > max_hours:
            continue

        v = get_scalar_value(ds[wind_var][idx, t])
        if pd.isna(v):
            continue
        if v <= 0:
            continue

        t_list.append(dt_hours)
        v_list.append(v)

    return np.array(t_list, dtype=float), np.array(v_list, dtype=float)



def fit_exponential_decay_from_lmi(ds, idx, lmi_tidx,
                                   wind_var="usa_wind", time_var="iso_time",
                                   max_hours=24, min_points=2):
    """
    用 LMI 后 max_hours 内的所有有效点拟合:
        v(t) = A * exp(-k t)

    返回:
    - k_fit
    - tau_fit = 1/k
    - A_fit
    - n_points
    - t_hours
    - winds
    """
    t_hours, winds = get_post_lmi_wind_series(
        ds, idx, lmi_tidx,
        wind_var=wind_var, time_var=time_var,
        max_hours=max_hours
    )

    if len(t_hours) < min_points:
        return {
            "k_fit": np.nan,
            "tau_fit": np.nan,
            "A_fit": np.nan,
            "n_points": len(t_hours),
            "t_hours": t_hours,
            "winds": winds
        }

    # 对 ln(v) 做线性拟合
    logv = np.log(winds)
    slope, intercept = np.polyfit(t_hours, logv, 1)

    k_fit = -slope
    A_fit = np.exp(intercept)

    # 如果 k<=0，说明不是衰减，tau 没法按 decay timescale 解释
    if k_fit <= 0:
        tau_fit = np.nan
    else:
        tau_fit = 1.0 / k_fit

    return {
        "k_fit": k_fit,
        "tau_fit": tau_fit,
        "A_fit": A_fit,
        "n_points": len(t_hours),
        "t_hours": t_hours,
        "winds": winds
    }


def show_post_lmi_wind_series(ds, idx, lmi_tidx,
                              wind_var="usa_wind", time_var="iso_time",
                              max_hours=24, convert_to_ms=False):
    """
    打印某个 storm 在 LMI 后 max_hours 内的时间和风速
    """
    lmi_time_str = decode_time(ds[time_var][idx, lmi_tidx])
    lmi_time = pd.to_datetime(lmi_time_str, errors="coerce")
    if pd.isna(lmi_time):
        print("LMI time 无法读取")
        return None

    rows = []
    nt = ds[time_var].shape[1]

    for t in range(nt):
        time_str = decode_time(ds[time_var][idx, t])
        t_time = pd.to_datetime(time_str, errors="coerce")
        if pd.isna(t_time):
            continue

        dt_hours = (t_time - lmi_time).total_seconds() / 3600.0

        if dt_hours < 0 or dt_hours > max_hours:
            continue

        v = get_scalar_value(ds[wind_var][idx, t])
        if pd.isna(v):
            continue

        if convert_to_ms:
            v = v * 0.514444

        rows.append({
            "tidx": t,
            "time": t_time,
            "dt_hours": dt_hours,
            "wind": v
        })

    df_series = pd.DataFrame(rows)

    if len(df_series) == 0:
        print("没有找到 LMI 后的有效风速点")
        return None

    print(df_series)
    return df_series

ships_map = build_ships_block_map("ships_5day/SHIPS_5day_ALL.txt")

lmi_info = get_lmi_info_for_sid(ds, sid, sid_to_index)
if lmi_info is not None:
    timekey = ibtracs_time_to_ships_timekey(lmi_info["lmi_time"])
    block = ships_map.get((sid, timekey), None)

    if block is not None:
        shear = extract_shrd_from_block(block)
    else:
        shear = None

ships_map = build_ships_block_map("ships_5day/SHIPS_5day_ALL.txt")

catall_sids = list(dict.fromkeys(cat12_sids + cat345_sids))

def extract_vmpi_from_block(block):
    """
    Extract 0h VMPI from one SHIPS block.
    Return unit: kt.
    """
    if block is None:
        return None

    lines = block.splitlines() if isinstance(block, str) else block

    for line in lines:
        if line.strip().endswith("VMPI"):
            parts = line.split()

            if len(parts) < 2:
                return None

            try:
                vmpi = float(parts[0])   # 0h VMPI
            except ValueError:
                return None

            if vmpi in [999, 9999, -999, -9999]:
                return None

            return vmpi

    return None


results = []
failed_sids = []


results_exET = []
failed_sids_exET = []

considered_data = catall_sids

n_total = len(considered_data)

for i, sid in enumerate(considered_data):
    out = get_lmi_r34_shear_for_sid(ds, sid, sid_to_index, ships_map,exclude_non_tropical=True)
    out_ex = get_lmi_r34_shear_for_sid(ds, sid, sid_to_index, ships_map,exclude_non_tropical=True,require_cat1_24h=True)

    if out is None:
        failed_sids.append(sid)
    else:
        results.append(out)


    if out_ex is None:
        failed_sids_exET.append(sid)
    else:
        results_exET.append(out_ex)

    # 可选：每处理 50 个打印一次进度
    if (i + 1) % 50 == 0 or (i + 1) == n_total:
        print(f"{i+1}/{n_total} done")

df_result = pd.DataFrame(results)
df_result_exET = pd.DataFrame(results_exET)

print("总 sid 数:", n_total)
print("成功匹配数:", len(df_result))
print("失败数:", len(failed_sids))
print(df_result.head())

print("总 sid 数:", n_total)
print("exclude ET 成功匹配数:", len(df_result_exET))
print("失败数:", len(failed_sids_exET))
print(df_result_exET.head())

df_valid = df_result[
    df_result["shear_0h"].notna() 

].copy()
#    df_result["r34_value"].notna()

df_valid_exET = df_result_exET[
    df_result_exET["shear_0h"].notna() 

].copy()
#    df_result_exET["r34_value"].notna()

print("同时有有效 shear 和 r34 的样本数:", len(df_valid))
print("exclude ET 同时有有效 shear 和 r34 的样本数:", len(df_valid_exET))

wind_24h_list = []
t24_idx_list = []
t24_time_list = []
delta_v24_list = []
decay24_list = []
lmi_pres_list = []

tau_fit_list = []
k_fit_list = []
nfit_list = []


lmi_lat_list = []
lmi_lon_list = []
lmi_rmw_list = []



n_total = len(df_valid)
for i, (_, row) in enumerate(df_valid.iterrows()):
    idx = int(row["idx"])
    lmi_tidx = int(row["lmi_tidx"])
    v0 = row["lmi_wind"]
    

    p = get_scalar_value(ds["usa_pres"][idx, lmi_tidx])
    lmi_pres_list.append(p)

    v24, t24_idx, t24_time = get_wind_24h_after_lmi(ds, idx, lmi_tidx,target_hours=24)
    decay24 = calc_decay24(v0, v24, hours=24)


    out = fit_exponential_decay_from_lmi(
        ds, idx, lmi_tidx,
        wind_var="usa_wind",
        time_var="iso_time",
        max_hours=48,     # 你自己改，比如 48 / 72
        min_points=3
    )

    tau_fit_list.append(out["tau_fit"])
    k_fit_list.append(out["k_fit"])
    nfit_list.append(out["n_points"])

    

    wind_24h_list.append(v24)
    t24_idx_list.append(t24_idx)
    t24_time_list.append(t24_time)
    delta_v24_list.append(v0 - v24 if pd.notna(v24) else np.nan)
    decay24_list.append(decay24)


    lat = get_scalar_value(ds["usa_lat"][idx, lmi_tidx])
    lmi_lat_list.append(lat)

    lon = get_scalar_value(ds["usa_lon"][idx, lmi_tidx])
    lmi_lon_list.append(lon)
    

    rmw = get_scalar_value(ds["usa_rmw"][idx, lmi_tidx])
    lmi_rmw_list.append(rmw)

    if (i + 1) % 50 == 0 or (i + 1) == n_total:
        print(f"{i+1}/{n_total}")
    



df_valid["wind_24h"] = wind_24h_list
df_valid["t24_idx"] = t24_idx_list
df_valid["t24_time"] = t24_time_list
df_valid["delta_v24"] = delta_v24_list
df_valid["decay24"] = decay24_list
df_valid["lmi_pres"] = lmi_pres_list

df_valid["tau_fit"] = tau_fit_list
df_valid["k_fit"] = k_fit_list
df_valid["nfit"] = nfit_list


df_valid["lmi_lat"] = lmi_lat_list
df_valid["lmi_lon"] = lmi_lon_list
df_valid["lmi_rmw"] = lmi_rmw_list

wind_24h_list_exET = []
t24_idx_list_exET = []
t24_time_list_exET = []
delta_v24_list_exET = []
decay24_list_exET = []
lmi_pres_list_exET = []

tau_fit_list_exET = []
k_fit_list_exET = []
nfit_list_exET = []


lmi_lat_list_exET = []
lmi_lon_list_exET = []
lmi_rmw_list_exET = []



n_total_exET = len(df_valid_exET)
for i, (_, row) in enumerate(df_valid_exET.iterrows()):
    idx_exET = int(row["idx"])
    lmi_tidx_exET = int(row["lmi_tidx"])
    v0_exET = row["lmi_wind"]
    

    p_exET = get_scalar_value(ds["usa_pres"][idx_exET, lmi_tidx_exET])
    lmi_pres_list_exET.append(p_exET)

    v24_exET, t24_idx_exET, t24_time_exET = get_wind_24h_after_lmi(ds, idx_exET, lmi_tidx_exET,target_hours=24)
    decay24_exET = calc_decay24(v0_exET, v24_exET, hours=24)


    out_exET = fit_exponential_decay_from_lmi(
        ds, idx_exET, lmi_tidx_exET,
        wind_var="usa_wind",
        time_var="iso_time",
        max_hours=48,     # 你自己改，比如 48 / 72
        min_points=3
    )

    tau_fit_list_exET.append(out_exET["tau_fit"])
    k_fit_list_exET.append(out_exET["k_fit"])
    nfit_list_exET.append(out_exET["n_points"])

    

    wind_24h_list_exET.append(v24_exET)
    t24_idx_list_exET.append(t24_idx_exET)
    t24_time_list_exET.append(t24_time_exET)
    delta_v24_list_exET.append(v0_exET - v24_exET if pd.notna(v24_exET) else np.nan)
    decay24_list_exET.append(decay24_exET)


    lat_exET = get_scalar_value(ds["usa_lat"][idx_exET, lmi_tidx_exET])
    lmi_lat_list_exET.append(lat_exET)

    lon_exET = get_scalar_value(ds["usa_lon"][idx_exET, lmi_tidx_exET])
    lmi_lon_list_exET.append(lon_exET)

    

    rmw_exET = get_scalar_value(ds["usa_rmw"][idx_exET, lmi_tidx_exET])
    lmi_rmw_list_exET.append(rmw_exET)

    if (i + 1) % 50 == 0 or (i + 1) == n_total_exET:
        print(f"{i+1}/{n_total_exET}")
    



df_valid_exET["wind_24h"] = wind_24h_list_exET
df_valid_exET["t24_idx"] = t24_idx_list_exET
df_valid_exET["t24_time"] = t24_time_list_exET
df_valid_exET["delta_v24"] = delta_v24_list_exET
df_valid_exET["decay24"] = decay24_list_exET
df_valid_exET["lmi_pres"] = lmi_pres_list_exET

df_valid_exET["tau_fit"] = tau_fit_list_exET
df_valid_exET["k_fit"] = k_fit_list_exET
df_valid_exET["nfit"] = nfit_list_exET


df_valid_exET["lmi_lat"] = lmi_lat_list_exET
df_valid_exET["lmi_lon"] = lmi_lon_list_exET
df_valid_exET["lmi_rmw"] = lmi_rmw_list_exET

def get_pi_vmax_for_lmi(lmi_time, lmi_lat, lmi_lon,
                        year_arr, month_arr, lat_arr, lon_arr, vmax_arr):
    """
    根据 LMI 时间和位置，从 monthly mean PI 文件中读取 PI vmax
    文件结构: vmax(year, month, lat, lon)
    """

    if pd.isna(lmi_time) or pd.isna(lmi_lat) or pd.isna(lmi_lon):
        return np.nan

    t = pd.Timestamp(lmi_time)
    year_val = t.year
    month_val = t.month

    # year index
    year_matches = np.where(year_arr == year_val)[0]
    if len(year_matches) == 0:
        return np.nan
    iy = int(year_matches[0])

    # month index
    month_matches = np.where(month_arr == month_val)[0]
    if len(month_matches) == 0:
        return np.nan
    im = int(month_matches[0])

    # nearest lat
    ilat = int(np.abs(lat_arr - lmi_lat).argmin())

    # convert lon from -180~180 to 0~360
    lon_use = float(lmi_lon)
    if lon_use < 0:
        lon_use = lon_use + 360.0

    ilon = int(np.abs(lon_arr - lon_use).argmin())

    val = vmax_arr[iy, im, ilat, ilon]

    if np.ma.is_masked(val):
        return np.nan

    try:
        return float(val)
    except:
        return np.nan

pi_vmax_list = []
pi_vmax_200km_list = []

n_total = len(df_valid)

for i, (_, row) in enumerate(df_valid.iterrows()):
    pi_val_1 = get_pi_vmax_for_lmi(
        lmi_time=row["lmi_time_str"],   # 这里直接用 lmi_time_str
        lmi_lat=row["lmi_lat"],
        lmi_lon=row["lmi_lon"],
        year_arr=pi1_year,
        month_arr=pi1_month,
        
        lat_arr=pi1_lat,
        lon_arr=pi1_lon,
        vmax_arr=pi1_vmax
    )

    pi_val_2 = get_pi_vmax_for_lmi(
        lmi_time=row["lmi_time_str"],
        lmi_lat=row["lmi_lat"],
        lmi_lon=row["lmi_lon"],
        year_arr=pi2_year,
        month_arr=pi2_month,
        lat_arr=pi2_lat,
        lon_arr=pi2_lon,
        vmax_arr=pi2_vmax
    )

    pi_vmax_list.append(pi_val_1)
    pi_vmax_200km_list.append(pi_val_2)

    if (i + 1) % 50 == 0 or (i + 1) == n_total:
        print(f"{i+1}/{n_total}")

df_valid["pi_vmax"] = pi_vmax_list
df_valid["pi_vmax_200km"] = pi_vmax_200km_list

pi_vmax_list_exET = []
pi_vmax_200km_list_exET = []

n_total_exET = len(df_valid_exET)

for i, (_, row) in enumerate(df_valid_exET.iterrows()):
    pi_val_1_exET = get_pi_vmax_for_lmi(
        lmi_time=row["lmi_time_str"],   # 这里直接用 lmi_time_str
        lmi_lat=row["lmi_lat"],
        lmi_lon=row["lmi_lon"],
        year_arr=pi1_year,
        month_arr=pi1_month,
        lat_arr=pi1_lat,
        lon_arr=pi1_lon,
        vmax_arr=pi1_vmax
    )

    pi_val_2_exET = get_pi_vmax_for_lmi(
        lmi_time=row["lmi_time_str"],
        lmi_lat=row["lmi_lat"],
        lmi_lon=row["lmi_lon"],
        year_arr=pi2_year,
        month_arr=pi2_month,
        lat_arr=pi2_lat,
        lon_arr=pi2_lon,
        vmax_arr=pi2_vmax
    )

    pi_vmax_list_exET.append(pi_val_1_exET)
    pi_vmax_200km_list_exET.append(pi_val_2_exET)

    if (i + 1) % 50 == 0 or (i + 1) == n_total_exET:
        print(f"{i+1}/{n_total_exET}")

df_valid_exET["pi_vmax"] = pi_vmax_list_exET
df_valid_exET["pi_vmax_200km"] = pi_vmax_200km_list_exET

df_plot = df_valid[
#    df_valid["r34_value"].notna() &
    df_valid["shear_0h"].notna() &
    df_valid["lmi_wind"].notna() &
    (df_valid["lmi_wind"] >= 33) &
    df_valid["wind_24h"].notna() &
    df_valid["decay24"].notna() &
    df_valid["lmi_pres"].notna() &

    df_valid["lmi_lat"].notna()&
    df_valid["pi_vmax"].notna()
].copy()

df_plot2 = df_valid[
#    df_valid["r34_value"].notna() &
    df_valid["shear_0h"].notna() &
    df_valid["lmi_wind"].notna() &
#    (df_valid["lmi_wind"] > 35) &
    df_valid["wind_24h"].notna() &
    df_valid["decay24"].notna() &
    df_valid["lmi_pres"].notna()
].copy()

df_fit = df_valid[
#    df_valid["r34_value"].notna() &
    df_valid["shear_0h"].notna() &
    df_valid["lmi_wind"].notna() &
#    (df_valid["lmi_wind"] > 35) &
    df_valid["wind_24h"].notna() &
    df_valid["decay24"].notna() &
    df_valid["lmi_pres"].notna() &
    df_valid['tau_fit'].notna() &
    df_valid["lmi_lat"].notna()
].copy()

df_plot["efold_time"] = np.where(
    df_plot["decay24"] > 0.0005,
    1.0 / df_plot["decay24"],
    np.nan
)

df_plot2["efold_time"] = np.where(
    df_plot2["decay24"] > 0.005,
    1.0 / df_plot2["decay24"],
    np.nan
)

df_fit["efold_time"] = np.where(
    df_fit["decay24"] > 0.0005,
    1.0 / df_fit["decay24"],
    np.nan
)

df_plot_exET = df_valid_exET[
#    df_valid_exET["r34_value"].notna() &
    df_valid_exET["shear_0h"].notna() &
    df_valid_exET["lmi_wind"].notna() &
    (df_valid_exET["lmi_wind"] > 33) &
    df_valid_exET["wind_24h"].notna() &
    df_valid_exET["decay24"].notna() &
    df_valid_exET["lmi_pres"].notna() &

    df_valid_exET["lmi_lat"].notna()&
    df_valid_exET["pi_vmax"].notna()
].copy()

df_plot_exET["efold_time"] = np.where(
    df_plot_exET["decay24"] > 0.0005,
    1.0 / df_plot_exET["decay24"],
    np.nan
)

from scipy import stats

df_reg = df_plot[[
    "efold_time",
    "shear_24h_mean",   # 这里可改成 "shear_0h"
    "lmi_wind","lmi_lat",'lmi_rmw','pi_vmax'
]].copy()

df_reg["log_tau"] = np.log(df_reg["efold_time"])


df_reg["vmax_shear"] = df_reg["lmi_wind"] * df_reg["shear_24h_mean"]
df_reg["RI"] = df_reg["lmi_wind"] / df_reg["pi_vmax"]

df_reg['RI_shear'] = df_reg['RI'] * df_reg['shear_24h_mean']
df_reg['PI_shear'] = df_reg['pi_vmax'] * df_reg['shear_24h_mean']
df_reg["REI"] = (df_reg["lmi_wind"] - 33.0) / np.maximum(df_reg["pi_vmax"] - 33.0,33.0)
df_reg['REI_shear'] = df_reg['REI'] * df_reg['shear_24h_mean']

df_reg["diff_PI_lmi"] = df_reg["pi_vmax"] - df_reg["lmi_wind"]
df_reg['diff_shear'] = df_reg['diff_PI_lmi'] * df_reg['shear_24h_mean']

# 因变量
y = df_reg["efold_time"].values.astype(float)

# 自变量
X_raw = df_reg[[
    "shear_24h_mean",


    'REI',

    'REI_shear'


]].values.astype(float)

# 加截距项
X = np.column_stack([np.ones(len(X_raw)), X_raw])

# 最小二乘拟合
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

# 预测值
y_hat = X @ beta

# 样本数、参数数
n = len(y)
p = X.shape[1]   # 包括截距

# 残差
resid = y - y_hat

# SSE, SST, R^2
SSE = np.sum(resid**2)
SST = np.sum((y - np.mean(y))**2)
R2 = 1 - SSE / SST
adj_R2 = 1 - (SSE / (n - p)) / (SST / (n - 1))

# 残差方差
sigma2 = SSE / (n - p)

# 系数协方差矩阵
XtX_inv = np.linalg.inv(X.T @ X)
cov_beta = sigma2 * XtX_inv

# 标准误
se_beta = np.sqrt(np.diag(cov_beta))

# t统计量
t_stats = beta / se_beta

# 双侧p值
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

param_names = [
    "const",
    "shear_24h_mean",


    'REI',

    'REI_shear'



]

print("Multiple linear regression with interactions")
print("-" * 60)
for name, coef, se, tval, pval in zip(param_names, beta, se_beta, t_stats, p_values):
    print(f"{name:15s} coef = {coef:10.4f}   SE = {se:10.4f}   t = {tval:9.3f}   p = {pval:.5f}")

print("-" * 60)
print(f"N = {n}")
print(f"R^2 = {R2:.4f}")
print(f"Adjusted R^2 = {adj_R2:.4f}")

def plot_raw_binned_relation(
    x,
    y,
    bin_edges=None,
    bins=8,
    bin_method="quantile",   # "quantile" 或 "manual"
    xlabel="x",
    ylabel="log(e-folding time)",
    title=None
):
    data = pd.DataFrame({"x": x, "y": y})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 分 bin
    if bin_method == "manual":
        data["bin"] = pd.cut(data["x"], bins=bin_edges, include_lowest=True)
    else:
        data["bin"] = pd.qcut(data["x"], q=bins, duplicates="drop")

    # 每个 bin 求平均
    summary = data.groupby("bin", observed=True).agg(
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        y_std=("y", "std"),
        count=("y", "count")
    ).reset_index()

    # 去掉空 bin
    fit_data = summary[["x_mean", "y_mean"]].dropna()

    # linear approximation + Pearson R
    slope, intercept = np.polyfit(fit_data["x_mean"], fit_data["y_mean"], 1)
    r, p = stats.pearsonr(fit_data["x_mean"], fit_data["y_mean"])

    x_fit = np.linspace(fit_data["x_mean"].min(), fit_data["x_mean"].max(), 100)
    y_fit = slope * x_fit + intercept

    plt.figure(figsize=(6, 4))
    plt.scatter(data["x"], data["y"], alpha=0.18, label="Raw data")
    plt.scatter(summary["x_mean"], summary["y_mean"], color="red", label="Binned mean")

    plt.plot(
        x_fit, y_fit,
        "--",
        color="purple",
        linewidth=2.5,
        label=f"Fit: R={r:.2f}, p={p:.3g}"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title is not None else f"{ylabel} vs {xlabel}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    print(f"slope = {slope:.4f}")
    print(f"intercept = {intercept:.4f}")
    print(f"R = {r:.4f}")
    print(f"p = {p:.4g}")
    print(f"N raw = {len(data)}")
    print(f"N bins = {len(fit_data)}")

    return summary, slope, intercept, r, p

y = df_plot["efold_time"]
#shear_bins = [1,5,9,14,19,21,25,29]

# 1. lifetime - shear
summary_shear, slope_shear, intercept_shear, r_shear, p_shear = plot_raw_binned_relation(
    x=df_reg["shear_24h_mean"],
    y=y,
    bins=8,
    bin_method="quantile",
    xlabel="Wind shear",
    ylabel="Lifetime (h)",
    title="Lifetime vs shear"
)

# 2. lifetime - wind speed
lmi_bins = [33,36,40,45,50,55,60,65,70]

summary_vmax, slope_vmax, intercept_vmax, r_vmax, p_vmax = plot_raw_binned_relation(
    x=df_reg["lmi_wind"],
    y=y,
    bin_edges=lmi_bins,
    bin_method="manual",
    xlabel="LMI wind",
    ylabel="Lifetime (h)",
    title="Lifetime vs LMI"
)

# 3. lifetime - wind speed × shear
x_interaction = df_reg["vmax_shear"]

lmi_shear_bins = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]

summary_inter, slope_inter, intercept_inter, r_inter, p_inter = plot_raw_binned_relation(
    x=x_interaction,
    y=y,
    bin_edges=lmi_shear_bins,
    bin_method="manual",
    xlabel="Wind shear × LMI wind",
    ylabel="Lifetime (h)",
    title="Lifetime vs wind shear * LMI"
)





















































