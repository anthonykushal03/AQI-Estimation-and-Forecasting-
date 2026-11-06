# Data

The raw data files are **not included** in this repository. They can be downloaded free of charge from the Central Pollution Control Board (CPCB).

## Source

- **Portal:** CPCB Central Control Room for Air Quality Management — https://airquality.cpcb.gov.in/ccr/
- **Station:** Anand Vihar, Delhi - DPCC
- **Period:** 1 January 2020 – 31 December 2024
- **Frequency:** daily (24-hour average)

## How to reproduce the dataset

1. Open the CPCB portal and go to the data download section (Advanced Search / Data Repository).
2. Select **State:** Delhi, **City:** Delhi, **Station:** Anand Vihar, Delhi - DPCC.
3. Select the parameters listed below and **Report format:** 24 Hours.
4. Download one calendar year at a time (2020 to 2024) and save each as a CSV file.
5. Tidy each file so it has the column names below, add a `Month` column (1–12), and name the files:

```
data/
├── Data_2020(1).csv
├── Data_2021(1).csv
├── Data_2022(1).csv
├── Data_2023(1).csv
└── Data_2024(1).csv
```

The portal layout may change over time, so the exact menu names may differ slightly.

## Expected columns

| Column | Unit | Description |
|---|---|---|
| `Timestamp` | date (M/D/YYYY) | Day of measurement |
| `PM2.5 (µg/m³)` | µg/m³ | Fine particulate matter |
| `PM10 (µg/m³)` | µg/m³ | Coarse particulate matter |
| `NO (µg/m³)` | µg/m³ | Nitric oxide |
| `NO2 (µg/m³)` | µg/m³ | Nitrogen dioxide |
| `NOx (ppb)` | ppb | Nitrogen oxides |
| `NH3 (µg/m³)` | µg/m³ | Ammonia |
| `SO2 (µg/m³)` | µg/m³ | Sulphur dioxide |
| `CO (mg/m³)` | mg/m³ | Carbon monoxide |
| `Ozone (µg/m³)` | µg/m³ | Ozone |
| `Benzene (µg/m³)` | µg/m³ | Benzene |
| `Toluene (µg/m³)` | µg/m³ | Toluene |
| `AT (°C)` | °C | Air temperature |
| `RH (%)` | % | Relative humidity |
| `WS (m/s)` | m/s | Wind speed |
| `WD (deg)` | degrees | Wind direction (see data-quality note in the notebook) |
| `SR (W/mt2)` | W/m² | Solar radiation |
| `BP (mmHg)` | mmHg | Barometric pressure |
| `Month` | 1–12 | Month number |
| `AQI` | index | CPCB Air Quality Index |

## Data notes

- 1,827 daily records; missing values are left as blanks in the CSV files.
- About 13% of AQI values are missing (176 days in 2020–2023, 55 days in 2024). The ML models use only days with measured AQI. Interpolated AQI is used only for the continuous ARIMA series.
- Benzene and Toluene contain a gap of 174 consecutive days (16 Feb – 8 Aug 2023), so neither is used as a model input.
- The daily AQI appears to cover a 24-hour window of roughly 4 PM to 4 PM, while the pollutant values are calendar-day averages. Check the CPCB documentation for the exact definition before using the data for forecasting.
