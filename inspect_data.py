import xarray as xr

print("Opening merged ERA5 dataset...")

ds = xr.open_dataset("era5_2023_full.nc")

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")
print(ds)

print("\n==============================")
print("VARIABLES")
print("==============================")
for var in ds.data_vars:
    print(f"{var} : {ds[var].attrs}")

print("\n==============================")
print("DIMENSIONS")
print("==============================")
print(ds.dims)

print("\n==============================")
print("PRESSURE LEVELS")
print("==============================")
print(ds.pressure_level.values)

print("\n==============================")
print("TIME")
print("==============================")
print(ds.valid_time.values[:5])
print("...")
print(ds.valid_time.values[-5:])

print("\n==============================")
print("LATITUDE")
print("==============================")
print(ds.latitude.values[:5])
print("...")
print(ds.latitude.values[-5:])

print("\n==============================")
print("LONGITUDE")
print("==============================")
print(ds.longitude.values[:5])
print("...")
print(ds.longitude.values[-5:])