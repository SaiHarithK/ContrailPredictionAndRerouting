import cdsapi

client = cdsapi.Client()

months = [
    '02', '03', '04', '05', '06',
    '07', '08', '09', '10', '11', '12'
]

for month in months:

    output_file = f"era5_2023_{month}.nc"

    print(f"\nDownloading Month: {month}")

    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',

            'variable': [
                'temperature',
                'relative_humidity',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
                'geopotential'
            ],

            'pressure_level': [
                '150',
                '175',
                '200',
                '225',
                '250',
                '275',
                '300'
            ],

            'year': '2023',

            'month': month,

            'day': [
                '01','02','03','04','05','06','07','08','09','10',
                '11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','25','26','27','28','29','30','31'
            ],

            'time': [
                '00:00',
                '06:00',
                '12:00',
                '18:00'
            ],

            'area': [
                60,
                -10,
                35,
                30
            ],

            'format': 'netcdf'
        },

        output_file
    )

    print(f"Finished {month} -> {output_file}")

print("\n===================================")
print("ALL DOWNLOADS COMPLETED!")
print("===================================")