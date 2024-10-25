## Potential groups for analysis:
acs_groups = {"age_sex_educational_attainment": "B15001",
              "age_sex": "B01001",
              "estimated_yearly_income": "B25121"
              }

STATEFP = '47' # TN

# 11 counties around BOC
# https://www.census.gov/library/reference/code-lists/ansi.html

COUNTY_FPS_MAP = {"DYER":"045",
                  "GIBSON":"053",
                  "CROCKETT": "033",
                  "LAUDERDALE": "097",
                  "TIPTON": "167",
                  "FAYETTE":"047",
                  "SHELBY":"157",
                  "HARDEMAN":"069",
                  "CHESTER":"023",
                  "MADISON":"113",
                  "HAYWOOD":"075"
                  }
FPS_COUNTY_MAP = {v: k for k, v in COUNTY_FPS_MAP.items()}
# COUNTYFPS = ['097', '075', '023', '069', '157', '167', '033', '113', '045', '053', '047']
COUNTYFPS = [COUNTY_FPS_MAP[k] for k in COUNTY_FPS_MAP.keys()]
BOC_TOTAL_POPULATION = 10_000

ACS5 = "https://api.census.gov/data/2022/acs/acs5"

HRA_MAP = {
    "MATA": ["SHELBY"],
    "DELTA": ["TIPTON", "FAYETTE", "LAUDERDALE"],
    "SOUTHWEST": ["MADISON", "HARDEMAN", "CHESTER", "HAYWOOD"],
    "NORTHWEST": ["GIBSON", "DYER", "CROCKETT"],
}