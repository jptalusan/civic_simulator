import requests
import pandas as pd

def get_census_data(api_key, api_url, table_name, state_fips, county_fips, block_groups, county_only):
    # print(block_groups)
    url = api_url

    if county_only:
        params = {
            "get": f"NAME,group({table_name})",
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}",
            "key": api_key,
        }
    else:
        params = {
            "get": f"NAME,group({table_name})",
            "for": f"block group:{block_groups}",
            "in": f"state:{state_fips} county:{county_fips}",
            "key": api_key,
        }
    # print(url)
    # print(params)
    response = requests.get(url, params=params)
    # print(response.url)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    else:
        print("Failed to retrieve data:", response.status_code)