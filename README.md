# Setup
* `git clone`
* `cd civic_simulator`
* `source ./set_env.sh`
* Generate requests based on population: `./bin/by_block_population.ipynb`

# Required Files
* Census Block Shapefiles
    * [Link](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
    * Year: 2022
    * Layer type: Blocks or Block Groups
* Building Data
    * [Link](https://github.com/microsoft/USBuildingFootprints?tab=readme-ov-file)
    * Scroll down and download the [Tennessee geojson file.](https://usbuildingdata.blob.core.windows.net/usbuildings-v2/Tennessee.geojson.zip)
    * 890MB file that covers the entire Tennesse will need to be `sjoin` into the corresponding cblocks.
* Census Data
    * Downloaded using an API. API should be placed in `.env` file:
    * API should be free.
    * `CENSUS_API_KEY=CENSUS_GOV_API_KEY`
* Census Variable names
    * [Link](https://www.census.gov/data/developers/data-sets/acs-5year.html)
    * This will interpret the column names generated by the census dataset.
    * Select `Detailed Tables > Json`
    * [HTML Version](https://api.census.gov/data/2022/acs/acs5/variables.html)
* BOC.geojson
    * This was manually generated for now.
    * I will add it to the data folder.
* County codes:
    * [TN](https://www2.census.gov/geo/docs/reference/codes2020/cou/st47_tn_cou2020.txt)