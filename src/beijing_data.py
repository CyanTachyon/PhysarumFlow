"""
Beijing major location dataset, parallel to tokyo_data.py.

We use 36 representative points across the Beijing metropolitan area:
- Major railway stations (Beijing, Beijing West/South/North)
- Major subway transfer hubs
- Key urban districts (Zhongguancun, Wangjing, Sanlitun, Financial Street...)
- Outer districts (Tongzhou, Yizhuang, Daxing, Shijingshan...)
- Airport

This mirrors the paper's setup (36 nodes around Greater Tokyo). All
coordinates come from the approximate lat/lon of well-known locations.
The set is intentionally chosen to span the area that the real Beijing
subway network covers, so the Physarum solution can be compared against
the real network topology.
"""

from __future__ import annotations

import numpy as np

_RAW_BEIFING_LOCATIONS = [
    ("Tiananmen",          39.9087, 116.3974),
    ("Beijing Station",    39.9027, 116.4271),
    ("Beijing West",       39.8949, 116.3226),
    ("Beijing South",      39.8651, 116.3786),
    ("Beijing North",      39.9438, 116.3539),
    ("Xizhimen",           39.9409, 116.3551),
    ("Dongzhimen",         39.9417, 116.4339),
    ("Jianguomen",         39.9083, 116.4336),
    ("Fuxingmen",          39.9083, 116.3539),
    ("Guomao",             39.9094, 116.4589),
    ("Zhongguancun",       39.9841, 116.3160),
    ("Peking Univ",        39.9921, 116.3054),
    ("Tsinghua",           40.0028, 116.3253),
    ("Olympic Park",       39.9928, 116.3936),
    ("Sanlitun",           39.9355, 116.4500),
    ("Wangfujing",         39.9143, 116.4109),
    ("Qianmen",            39.8995, 116.3956),
    ("Yonghegong",         39.9475, 116.4127),
    ("Huixinxijie",        39.9847, 116.4117),
    ("Shaoyaoju",          39.9824, 116.4502),
    ("Wangjing",           39.9972, 116.4759),
    ("Tiantan",            39.8822, 116.4066),
    ("Financial St",       39.9183, 116.3589),
    ("Nanluoguxiang",      39.9356, 116.4033),
    ("Andingmen",          39.9527, 116.4069),
    ("Hepingli",           39.9613, 116.4163),
    ("Sihui",              39.9089, 116.4897),
    ("Tongzhou",           39.9087, 116.6566),
    ("Yizhuang",           39.7984, 116.5105),
    ("Daxing",             39.7297, 116.3416),
    ("Capital Airport",    40.0801, 116.5846),
    ("Shijingshan",        39.9167, 116.1872),
    ("Fragrant Hills",     39.9921, 116.1600),
    ("Wukesong",           39.9094, 116.2754),
    ("Gongzhufen",         39.9090, 116.3207),
    ("Lize",               39.8683, 116.3256),
]


def beijing_coords() -> tuple[np.ndarray, list[str]]:
    names = [r[0] for r in _RAW_BEIFING_LOCATIONS]
    lat = np.array([r[1] for r in _RAW_BEIFING_LOCATIONS])
    lon = np.array([r[2] for r in _RAW_BEIFING_LOCATIONS])

    lat0, lon0 = lat.mean(), lon.mean()
    R = 6371.0
    deg = np.pi / 180.0
    x = R * (lon - lon0) * deg * np.cos(lat0 * deg)
    y = R * (lat - lat0) * deg
    coords = np.stack([x, y], axis=1)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    extent = max(x_max - x_min, y_max - y_min)
    coords = 0.05 + 0.9 * (coords - coords.min(axis=0)) / extent

    return coords, names


if __name__ == "__main__":
    coords, names = beijing_coords()
    print(f"Beijing dataset: {len(names)} nodes")
    print(f"x range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"y range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
