import requests
import pandas as pd
import json

base_url = "https://envrio.org/era5_api/"

response = requests.post(f'{base_url}auth', json={"username": "xylopodaros@yahoo.gr", "password": "TestPass123@123"})

if response.json().get('access_token'):
    access_token = response.json().get('access_token')
    headers = {'Authorization': f'Bearer {access_token}'}


variables = requests.get(f'{base_url}variables', headers=headers)


print()