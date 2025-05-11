import os
import numpy as np
import pandas as pd
from prefect import flow, task
from common import engine


@task(retries=0)
def extract_data_for_year(year: int, data_path: str) -> None:
    print(f'Processing {year}...')
    try:
        data_file_path = os.path.join(data_path, f"{year}VAERSDATA.csv")
        vax_file_path = os.path.join(data_path, f"{year}VAERSVAX.csv")
        symptoms_file_path = os.path.join(data_path, f"{year}VAERSSYMPTOMS.csv")

        if os.path.exists(data_file_path) and os.path.exists(vax_file_path) and os.path.exists(symptoms_file_path):
            df_data = pd.read_csv(data_file_path, low_memory=False, encoding='latin1')
            df_vax = pd.read_csv(vax_file_path, low_memory=False, encoding='latin1')
            df_symptoms = pd.read_csv(symptoms_file_path, low_memory=False, encoding='latin1')

            merged = df_data.merge(df_vax, on='VAERS_ID', how='left')
            merged = merged.merge(df_symptoms, on='VAERS_ID', how='left')

            merged['YEAR'] = year

            merged = merged.drop_duplicates()
            merged = merged.dropna(subset=['VAERS_ID', 'RECVDATE', 'AGE_YRS'])

            merged['RECVDATE'] = pd.to_datetime(merged['RECVDATE'], errors='coerce')
            merged['DATEDIED'] = pd.to_datetime(merged['DATEDIED'], errors='coerce')
            merged['VAX_DATE'] = pd.to_datetime(merged['DATEDIED'], errors='coerce')
            merged['ONSET_DATE'] = pd.to_datetime(merged['DATEDIED'], errors='coerce')

            merged['AGE_YRS'] = pd.to_numeric(merged['AGE_YRS'], errors='coerce')

            merged = merged.drop(columns=['V_ADMINBY', 'V_FUNDBY', 'FORM_VERS', 'LAB_DATA', 'SPLTTYPE', 'TODAYS_DATE', 'OFC_VISIT'],
                                         errors='ignore')

            conditions = [
                merged['HOSPITAL'] == 'Y',
                (merged['ER_ED_VISIT'] == 'Y') | (merged['ER_VISIT'] == 'Y'),
                merged['DIED'] == 'Y'
            ]
            choices = ['Hospitalization', 'ER Visit', 'Death']
            merged['OUTCOME'] = np.select(conditions, choices, default='No hospitalization, ER visit, or death')

            merged.to_sql('vaers_data', engine, if_exists='append', index=False)

        else:
            print(f'Missing files for {year}')

    except Exception as e:
        print(f'Error in {year}: {e}')

@flow(retries=0)
def extract_data(years: list[int]):
    for year in years:
        extract_data_for_year(year=year, data_path='AllVAERSDataCSVS')

if __name__ == '__main__':
    extract_data(list(range(1990, 2026)))