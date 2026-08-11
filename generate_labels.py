import pandas as pd
from label_generation import *

print("Loading Features...")

df = pd.read_parquet(
    "feature_dataset.parquet"
)

print(df.shape)

print("Generating Labels...")

df = generate_labels(df)

label_diagnostics(df)

df.to_parquet(
    "labeled_dataset.parquet",
    index=False
)
print(
    f"delta_t < 0 : "
    f"{100*(df['delta_t']<0).mean():.2f}%"
)

print(
    f"rhi > 100 : "
    f"{100*(df['rhi']>100).mean():.2f}%"
)
print("\nSaved!")
print(df.head())