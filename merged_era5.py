import xarray as xr

files = [
    "era5_2023_01.nc",
    "era5_2023_02.nc",
    "era5_2023_03.nc",
    "era5_2023_04.nc",
    "era5_2023_05.nc",
    "era5_2023_06.nc",
    "era5_2023_07.nc",
    "era5_2023_08.nc",
    "era5_2023_09.nc",
    "era5_2023_10.nc",
    "era5_2023_11.nc",
    "era5_2023_12.nc"
]

print("Opening ERA5 files...")

ds = xr.open_mfdataset(
    files,
    combine="by_coords"
)

print("Files merged successfully!")

print("\nDataset Summary:")
print(ds)

print("\nSaving merged dataset...")

ds.to_netcdf("era5_2023_full.nc")

print("\nMerged dataset saved as era5_2023_full.nc")