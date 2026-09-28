"""
Greater Tokyo region city coordinates for the Physarum network experiment.

The original paper (Tero et al. 2010) used a set of ~36 major cities/towns in
the Greater Tokyo (Kanto) area as the food sources. The exact coordinates are
not published in the main paper, but the geographic layout is well documented
in the SOM and in many open-source reproductions.

We approximate the locations using real latitude/longitude of major Kanto
cities (Greater Tokyo, Kanagawa, Saitama, Chiba, and southern Ibaraki).
Coordinates are converted to a flat Cartesian plane using an equirectangular
projection centred on Tokyo, then normalised to a [0, 1] box.

The set includes 36 nodes - matching the paper's setup.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Raw data: name, latitude (deg N), longitude (deg E)
# Approximate city-hall locations of 36 cities/towns across the Kanto plain.
# ---------------------------------------------------------------------------
_RAW_KANTO_CITIES = [
    # name, lat, lon
    ("Tokyo",         35.6812, 139.7671),   # Tokyo Station
    ("Shinjuku",      35.6896, 139.6917),
    ("Shibuya",       35.6640, 139.6981),
    ("Ikebukuro",     35.7295, 139.7109),
    ("Urawa",         35.8616, 139.6455),
    ("Omiya",         35.9053, 139.6243),
    ("Kawaguchi",     35.8050, 139.7310),
    ("Koshigaya",     35.8910, 139.7900),
    ("Soka",          35.8260, 139.8080),
    ("Matsudo",       35.7830, 139.9030),
    ("Kashiwa",       35.8610, 139.9680),
    ("Ichikawa",      35.7200, 139.9290),
    ("Funabashi",     35.6930, 139.9830),
    ("Chiba",         35.6070, 140.1220),
    ("Kisarazu",      35.3730, 139.9220),  # across Tokyo bay
    ("Kawasaki",      35.5310, 139.6960),
    ("Yokohama",      35.4437, 139.6380),
    ("Yokosuka",      35.2860, 139.6670),
    ("Kamakura",      35.3190, 139.5470),
    ("Fujisawa",      35.3420, 139.4740),
    ("Tama",          35.6370, 139.4460),
    ("Hachioji",      35.6550, 139.3240),
    ("Sagamihara",    35.5590, 139.3720),
    ("Machida",       35.5440, 139.4480),
    ("Akishima",      35.7020, 139.3500),
    ("Tachikawa",     35.6960, 139.4130),
    ("Kokubunji",     35.7050, 139.4750),
    ("Fuchu",         35.6720, 139.4800),
    ("Higashi-Murayama",35.7540, 139.4680),
    ("Kawagoe",       35.9280, 139.4850),
    ("Kuki",          36.0680, 139.6680),
    ("Kazo",          36.1210, 139.7040),
    ("Gyoda",         36.1400, 139.4550),
    ("Kasukabe",      35.9760, 139.7490),
    ("Noda",          35.9440, 139.8660),
    ("Urayasu",       35.6610, 139.8940),
]


def tokyo_coords() -> tuple[np.ndarray, list[str]]:
    """Return (coords, names) for the 36-node Greater Tokyo dataset.

    Coordinates are in arbitrary units, normalised to a [0, 1] bounding box
    that preserves geographic aspect ratio.
    """
    names = [r[0] for r in _RAW_KANTO_CITIES]
    lat = np.array([r[1] for r in _RAW_KANTO_CITIES])
    lon = np.array([r[2] for r in _RAW_KANTO_CITIES])

    # Equirectangular projection around the centroid.
    lat0, lon0 = lat.mean(), lon.mean()
    R = 6371.0  # Earth radius km
    deg = np.pi / 180.0
    x = R * (lon - lon0) * deg * np.cos(lat0 * deg)
    y = R * (lat - lat0) * deg

    coords = np.stack([x, y], axis=1)

    # Normalise to [0.05, 0.95] preserving aspect ratio.
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    extent = max(x_max - x_min, y_max - y_min)
    coords = 0.05 + 0.9 * (coords - coords.min(axis=0)) / extent

    return coords, names


if __name__ == "__main__":
    coords, names = tokyo_coords()
    print(f"Tokyo dataset: {len(names)} nodes")
    print(f"x range: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    print(f"y range: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    print("First 5 cities:")
    for i in range(5):
        print(f"  {names[i]:20s} ({coords[i,0]:.3f}, {coords[i,1]:.3f})")
