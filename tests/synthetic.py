"""Synthetic listings that mimic the public Zameen CSV layout. FOR TESTS ONLY, never for results."""
import numpy as np
import pandas as pd

LOCATIONS = {  # name: (price per sqft PKR, lat, lon)
    "DHA Defence": (21000, 31.47, 74.40), "Gulberg": (25000, 31.51, 74.34),
    "Model Town": (22000, 31.48, 74.32), "Bahria Town": (9500, 31.37, 74.18),
    "Johar Town": (12500, 31.47, 74.28), "Wapda Town": (10000, 31.44, 74.26),
    "Valencia Housing Society": (11500, 31.44, 74.26), "Allama Iqbal Town": (10500, 31.51, 74.29),
    "Samanabad": (8000, 31.54, 74.29), "Township": (7500, 31.45, 74.31),
}


def make_raw(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(LOCATIONS)
    rows = []
    for i in range(n):
        loc = names[rng.integers(len(names))]
        ppsf, lat, lon = LOCATIONS[loc]
        marla = float(rng.choice([3, 5, 7, 10, 20, 40], p=[0.1, 0.3, 0.2, 0.2, 0.15, 0.05]))
        beds = int(np.clip(round(marla ** 0.5 * 1.3 + rng.normal(0, 0.7)), 1, 10))
        baths = int(np.clip(beds + rng.integers(-1, 2), 1, 10))
        year = int(rng.choice([2018, 2019, 2020]))
        price = ppsf * marla * 225 * np.exp(rng.normal(0, 0.18)) * (1 + 0.08 * (year - 2018))
        rows.append({
            "property_id": 1000 + i, "location": loc, "city": "Lahore", "province_name": "Punjab",
            "property_type": "House", "purpose": "For Sale", "price": round(price, -3),
            "area": f"{marla:g} Marla" if marla < 20 else f"{marla / 20:g} Kanal",
            "Total_Area": marla * 225, "bedrooms": beds, "baths": baths,
            "latitude": lat + rng.normal(0, 0.015), "longitude": lon + rng.normal(0, 0.015),
            "date_added": f"{year}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}",
        })
    df = pd.DataFrame(rows)
    noise = df.sample(120, random_state=seed).copy()           # rows that must be filtered out
    noise.loc[noise.index[:30], "city"] = "Karachi"
    noise.loc[noise.index[30:60], "purpose"] = "For Rent"
    noise.loc[noise.index[60:90], "property_type"] = "Flat"
    noise.loc[noise.index[90:], "price"] = 5000                  # absurd price (typo)
    noise["property_id"] = noise["property_id"] + 50000
    dupes = df.sample(60, random_state=seed + 1)                 # exact duplicate listings
    return pd.concat([df, noise, dupes], ignore_index=True).sample(frac=1, random_state=seed)
