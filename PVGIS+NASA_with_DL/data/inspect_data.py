import pandas as pd

print("=== PVGIS HOURLY 2019-2020 ===")
for skip in range(10):
    try:
        df = pd.read_csv('raw_pvgis_hourly_2019_2020.csv', skiprows=skip)
        if any(c in df.columns for c in ['time','P','Gb(i)']):
            print(f"skiprows={skip}, shape={df.shape}")
            print("cols:", df.columns.tolist())
            print(df.head(3).to_string())
            print(df.tail(2).to_string())
            print("nulls:", df.isnull().sum().to_dict())
            print("dtypes:", df.dtypes.to_dict())
            break
    except Exception as e:
        pass

print("\n=== NASA 2019 ===")
for skip in range(10):
    try:
        df = pd.read_csv('raw_nasa_2019.csv', skiprows=skip)
        if any(c in df.columns for c in ['T2M','RH2M','YEAR','YYYYMMDD']):
            print(f"skiprows={skip}, shape={df.shape}")
            print("cols:", df.columns.tolist())
            print(df.head(3).to_string())
            print(df.tail(2).to_string())
            print("nulls:", df.isnull().sum().to_dict())
            break
    except Exception as e:
        pass

print("\n=== NASA 2020 ===")
for skip in range(10):
    try:
        df = pd.read_csv('raw_nasa_2020.csv', skiprows=skip)
        if any(c in df.columns for c in ['T2M','RH2M','YEAR','YYYYMMDD']):
            print(f"skiprows={skip}, shape={df.shape}")
            print("cols:", df.columns.tolist())
            print(df.head(3).to_string())
            print(df.tail(2).to_string())
            print("nulls:", df.isnull().sum().to_dict())
            break
    except Exception as e:
        pass

print("\n=== PVGIS TMY ===")
for skip in range(10):
    try:
        df = pd.read_csv('raw_pvgis_tmy.csv', skiprows=skip)
        if any(c in df.columns for c in ['time(UTC)','T2m','G(h)']):
            print(f"skiprows={skip}, shape={df.shape}")
            print("cols:", df.columns.tolist())
            print(df.head(3).to_string())
            print(df.tail(2).to_string())
            print("nulls:", df.isnull().sum().to_dict())
            break
    except Exception as e:
        pass
