# This is for running offline solvers in parallel.
# TODO: Change all 2024-10-01 references to the config file params.current_date
from keplergl import KeplerGl
import multiprocessing as mp
import pandas as pd
import geopandas as gpd
import dateparser
from shapely.geometry import LineString
import json
import os
from distutils.dir_util import copy_tree
import datetime as dt
from pathlib import Path
from rapidroutesim.common.time import Time
import rapidroutesim.simulator.manifest as mn
from rapidroutesim.simulator.events.event_queue import EventQueue
from rapidroutesim.simulator.logistics.logistics import Logistics
import requests
from rapidroutesim.simulator.solvers.smarttransit.smarttransit_bridge import offline_solve
from rapidroutesim.simulator.routers.router import Router
import argparse
import rapidroutesim.fixedline_simulator.constants as constants
from rapidroutesim.fixedline_simulator.event import Event
from rapidroutesim.fixedline_simulator.vehicle import ExpressBus
from rapidroutesim.fixedline_simulator.utils import generate_metrics_for_on_demand, get_linestring_duration_distance_for_OD_pair_dict, get_stops_df
import logging
from gtfs_functions import Feed

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to console handler
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

#####
pd.options.mode.chained_assignment = None  # default='warn'

note = "buses"
# all requests
all_employee_requests = pd.read_csv("data/employee_demand_basic_pop_models_with_transit_and_city_labels_memphis_clustered.csv", index_col=0)

def read_json_edit_key(fp, key, val):
    # Read the JSON file
    with open(fp, "r") as f:
        data = json.load(f)
    # Edit a key in the JSON data
    data[key] = val
    # Write the updated JSON data back to the file
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)
    return True

def read_results_csv(root_dir, filename):
    all_data = []
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            file_path = os.path.join(root, filename)
            df = pd.read_csv(file_path, index_col=0)

            if filename == 'results.csv':
                df['COUNTYFP'] = df['COUNTYFP'].bfill()
                df['COUNTYFP'] = df['COUNTYFP'].ffill()
            all_data.append(df)
    all_data = pd.concat(all_data)
    return all_data

def create_vehicles_list(count, capacity, out_dir):
    data = {}
    vehicles = []
    for vid in range(count):
        vehicles.append({"vehicle_id": vid, "am_capacity": capacity, "wc_capacity": 0})
        
    data["vehicle_list"] = vehicles
    # Write the updated JSON data back to the file
    with open(f"{out_dir}/cali_vehicles.json", 'w') as f:
        json.dump(data, f, indent=2)
        
    return True

def create_vehicle_schedules(count, offine_solve_start, start_time, end_time, out_dir):
    data = {}
    vehicles = []
    for vid in range(count):
        vehicles.append({"run_id": vid, 
                         "vehicle_id": vid, 
                         "start_time": start_time, 
                         "end_time": end_time})
        
    data["offline_solve_time"] = offine_solve_start
    data["schedule_list"] = vehicles
    # Write the updated JSON data back to the file
    with open(f"{out_dir}/cali_schedule.json", 'w') as f:
        json.dump(data, f, indent=2)
        
    return True

# Assuming people are ok leaving and arriving 5am to 8am
def convert_BOC_requests_to_payload(requests_df):
    requests_json = []
    travel_time_matrix = []
    for i, (k, v) in enumerate(requests_df.iterrows()):
        r = {}
        r["event"] = "client"
        r["type"] = "request"
        r["request_id"] = v["user_id"]
        r["event_time"] = "2024-09-29T07:15:00Z"
        r["requested_pickup"] = {
            "time": "2024-10-01T05:00:00Z",
            "location": {"lat": v["h_lat"], "lon": v["h_lon"]},
            "name": "Home",
        }

        r["requested_dropoff"] = {
            "time": "2024-10-01T05:00:00Z",
            "location": {"lat": v["w_lat"], "lon": v["w_lon"]},
            "name": "Work",
        }

        r["passenger_count"] = {"planned": 1, "actual": 1}

        r["wheelchair_count"] = {"planned": 0, "actual": 0}

        r["load"] = {"no_show": False, "time": 120}
        r["unload"] = {"no_show": False, "time": 60}

        travel_time_matrix.append(v["travel_time_s"])
        requests_json.append(r)
    return requests_json


def generate_results(manifest, county_requests, COUNTY="CHESTER", config={}):
    EXPERIMENT_NAME = config["params"]["output_name"]

    BASE_DATA = f"data/{EXPERIMENT_NAME}/{COUNTY}"
    OUTPUT_DIR = f"output/{EXPERIMENT_NAME}/{COUNTY}"
    Path(f"{OUTPUT_DIR}").mkdir(parents=True, exist_ok=True)

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(f"{OUTPUT_DIR}/manifest.csv")
    # Setup car's depot start and end

    # Read the JSON data from a file
    with open(f"{BASE_DATA}/logistics/cali_depot.json", "r") as f:
        depot_data = json.load(f)

    # Read the JSON data from a file
    with open(f"{BASE_DATA}/logistics/cali_schedule.json", "r") as f:
        schedule_data = json.load(f)

    depot_lat = depot_data["depot_location"]["lat"]
    depot_lon = depot_data["depot_location"]["lon"]
    schedule_list = schedule_data["schedule_list"]

    runs_start_end = []
    for run in manifest_df.run_id.unique().tolist():
        run_time_start = Time(schedule_list[run]["start_time"])
        run_time_end = Time(schedule_list[run]["end_time"])

        # Start
        run_dict = {
            "run_id": run,
            "h_lat": depot_lat,
            "h_lon": depot_lon,
            "building_id": "depot",
            "booking_id": -1,
            "action": "depot_start",
            "scheduled_time": run_time_start.seconds_from_midnight,
        }
        runs_start_end.append(run_dict)

        # End
        run_dict = {
            "run_id": run,
            "h_lat": depot_lat,
            "h_lon": depot_lon,
            "building_id": "depot",
            "booking_id": -1,
            "action": "depot_end",
            "scheduled_time": run_time_end.seconds_from_midnight,
        }
        runs_start_end.append(run_dict)

    driver_runs_df = pd.DataFrame(runs_start_end)

    served_users = pd.merge(county_requests, manifest_df, left_on=["user_id"], right_on=["booking_id"])
    unserved_users = served_users.user_id.unique()
    unserved_requests = county_requests[~county_requests["user_id"].isin(unserved_users)]
    logger.debug(f"served {len(county_requests.index)-len(unserved_requests.index)}/{len(county_requests.index)}")
    logger.debug(f"unserved {len(unserved_requests.index)}/{len(county_requests.index)}")
    results_df = pd.concat([served_users, driver_runs_df], join="outer", ignore_index=True).sort_values(
        by=["run_id", "scheduled_time"]
    )

    # Creating lat/lon column based on action
    results_df["lat"] = 0.0
    results_df["lon"] = 0.0
    results_df.loc[results_df["action"] == "pickup", "lat"] = results_df["h_lat"]
    results_df.loc[results_df["action"] == "pickup", "lon"] = results_df["h_lon"]

    results_df.loc[results_df["action"] == "dropoff", "lat"] = results_df["w_lat"]
    results_df.loc[results_df["action"] == "dropoff", "lon"] = results_df["w_lon"]

    results_df.to_csv(f"{OUTPUT_DIR}/results.csv")
    county_requests.to_csv(f"{OUTPUT_DIR}/requests.csv")

    # Write to JSON file
    with open(f"{OUTPUT_DIR}/configs.json", 'w') as f:
        json.dump(config, f, indent=2)

    return True


def solve_requests(COUNTY="CHESTER", county_requests=None, config={}):
    EXPERIMENT_NAME = config["params"]["output_name"]
    BASE_DATA = f"data/{EXPERIMENT_NAME}/{COUNTY}"
    queue = EventQueue(f"{BASE_DATA}/events/")
    logistics = Logistics(f"{BASE_DATA}/logistics/")
    router = Router(f"{BASE_DATA}/logistics/solver_sample.json")

    driver_runs = mn.DriverRuns(logistics.vehicles.init_driver_runs_dict())
    logistics_solve_time_dt = dateparser.parse(logistics.offline_solve_time).replace(tzinfo=None)

    request_events = []
    for e in queue.queue:
        e_type = e.type
        e_time = e.event_time
        if (e_type == "request") and (e_time.time_obj < logistics_solve_time_dt):
            request_events.append(e)
    logger.debug(f"Total requests: {len(request_events)}")
    payload = mn.generate_offline_payload(request_events, logistics, driver_runs)
    # pprint(payload)
    manifest = offline_solve(payload)

    # TODO: Improve res (on-demand) to be able to merge with fixed_line in another function before returning
    on_demand_result = generate_results(manifest, county_requests, COUNTY=COUNTY, config=config)

    ON_DEMAND_TO_TRANSIT_HUB = config["params"]["to_transit_hub"]
    fixed_line_result = None
    if ON_DEMAND_TO_TRANSIT_HUB:
        employee_arrivals = convert_on_demand_results_to_transit_arrivals(config, COUNTY.upper())
        fixed_line_result = run_fixed_line_simulator(employee_arrivals, config, COUNTY.upper())

    fixed_line_metrics, on_demand_metrics, per_user_metrics = compute_final_metrics(config=config, county=COUNTY, 
                                           county_requests=county_requests,
                                           on_demand_result=on_demand_result,
                                           fixed_line_result=fixed_line_result)
    return fixed_line_metrics, on_demand_metrics, per_user_metrics

def get_requests_per_county(countyname, county):
    '''
    requests should have COUNTYFP, shift, transit_taker, in_city, h_lat, h_lon, w_lat, w_lon
    '''
    named_stops = config["named_stops"]
    counties = config["counties"]
    on_demand_depot_to_boc = config["on_demand_depot_to_boc"]
    
    MEMPHIS_CLUSTERED = config["params"]["memphis_clustered"]
    SHIFT = config["params"]["shift"]
    USE_CITY_BUSES_TO_TRANSIT_HUB = config["params"]["use_city_buses"]
    ON_DEMAND_TO_TRANSIT_HUB = config["params"]["to_transit_hub"]
    ONLY_TRANSIT_TAKERS = config["params"]["transit_takers"]

    if MEMPHIS_CLUSTERED:
        county_requests = all_employee_requests[(all_employee_requests["COUNTYFP"] == int(county)) & \
                                                (all_employee_requests["shift"] == SHIFT)]
    else:
        copy_requests = all_employee_requests.copy()
        copy_requests['COUNTYFP'] = copy_requests['COUNTYFP'].astype('str')
        copy_requests.loc[copy_requests['COUNTYFP'].str.contains('157'), 'COUNTYFP'] = '157'
        copy_requests['COUNTYFP'] = copy_requests['COUNTYFP'].astype('int')
        county_requests = copy_requests[(copy_requests["COUNTYFP"] == int(county)) & (copy_requests["shift"] == SHIFT)]
    
    if ONLY_TRANSIT_TAKERS:
        county_requests = county_requests[(county_requests["transit_taker"])]

    if USE_CITY_BUSES_TO_TRANSIT_HUB:
        county_requests = county_requests[~county_requests['in_city']]

    if ON_DEMAND_TO_TRANSIT_HUB:
        # Depot
        county_depot = on_demand_depot_to_boc[countyname][0]
        county_requests.loc[:, "w_lat"] = named_stops[county_depot]["lat"]
        county_requests.loc[:, "w_lon"] = named_stops[county_depot]["lon"]
    else:
        # BOC
        county_depot = on_demand_depot_to_boc[countyname][-1]
        county_requests.loc[:, "w_lat"] = named_stops[county_depot]["lat"]
        county_requests.loc[:, "w_lon"] = named_stops[county_depot]["lon"]
    
    return county_requests

def setup_input_jsons(config):
    EXPERIMENT_NAME = config["params"]["output_name"]

    _add = ''
    MEMPHIS_CLUSTERED = config["params"]["memphis_clustered"]
    OUTPUT_NAME = config["params"]["output_name"]

    if MEMPHIS_CLUSTERED:
        _add = "_memphis_clustered"
    named_stops = config["named_stops"]
    counties = config["counties"]
    on_demand_depot_to_boc = config["on_demand_depot_to_boc"]
    on_demand_assets = config["on_demand_assets"]
    
    on_demand_start_time = config["schedules"]["on_demand_start_time"]
    on_demand_end_time = config["schedules"]["on_demand_end_time"]
    logger.info(f"Start of on-demand service: {on_demand_start_time}")

    WORK_DIR = "/Users/jose/Developer/git/RapidRouteSim"
    DATA_DIR = f"{WORK_DIR}/data/{EXPERIMENT_NAME}"
    BASE_EVENTS = f"{WORK_DIR}/data/events/"
    BASE_LOGISTICS = f"{WORK_DIR}/data/logistics/"

    county_requests_dict = {}

    for k, v in counties.items():
        countyname = k
        county = v
        if MEMPHIS_CLUSTERED and (countyname == "SHELBY"):
            continue

        Path(f"{DATA_DIR}/{countyname}").mkdir(parents=True, exist_ok=True)
        Path(f"{DATA_DIR}/{countyname}/events").mkdir(parents=True, exist_ok=True)
        Path(f"{DATA_DIR}/{countyname}/logistics").mkdir(parents=True, exist_ok=True)

        # Copy base files
        copy_tree(BASE_EVENTS, f"{DATA_DIR}/{countyname}/events/")
        copy_tree(BASE_LOGISTICS, f"{DATA_DIR}/{countyname}/logistics/")

        # Update events based on the requests.

        county_requests = get_requests_per_county(county=county, countyname=countyname)

        payload = convert_BOC_requests_to_payload(county_requests)

        clients_path = f"{DATA_DIR}/{countyname}/events/cali_clients.json"
        with open(clients_path, "w") as f:
            json.dump(payload, f, indent=4)

        county_depot = on_demand_depot_to_boc[countyname][0]
        # Update logistics based on the depot placement.
        depot_path = f"{DATA_DIR}/{countyname}/logistics/cali_depot.json"
        read_json_edit_key(depot_path, "depot_location", {"lat": named_stops[county_depot]["lat"], 
                                                          "lon": named_stops[county_depot]["lon"]})
    
        outdir = f"{DATA_DIR}/{countyname}/logistics/"
        county_on_demand_bus_count = on_demand_assets[countyname]["count"]
        county_on_demand_bus_capacity = on_demand_assets[countyname]["capacity"]
        create_vehicles_list(county_on_demand_bus_count, county_on_demand_bus_capacity, outdir)
        create_vehicle_schedules(county_on_demand_bus_count, "2024-09-30T00:00:01Z", 
                                 f"2024-10-01T{on_demand_start_time}:06Z", 
                                 f"2024-10-01T{on_demand_end_time}:29Z", outdir)
        
        county_requests_dict[countyname] = county_requests

    return county_requests_dict

def run_fixed_line_simulator(employee_arrivals=None, config={}, county="CHESTER"):
    EXPERIMENT_NAME = config["params"]["output_name"]

    logger.info(f"Starting fixed line simulations for {county.title()}")

    travel_time_distance_matrix_file = config["params"]["travel_time_distance_matrix_file"]
    tt_d_matrix = gpd.read_file(f"./data/{EXPERIMENT_NAME}/{travel_time_distance_matrix_file}")
    county_stops = tt_d_matrix[tt_d_matrix['county'] == county.upper()]

    buses, events = recreate_fixed_line_bus_events(config, county)
    config_date_str = config["params"]["current_date"]
    start_time_str = config["schedules"]["fixed_line_start_time"]
    start_time_dt = dateparser.parse(f"{config_date_str} {start_time_str}")

    end_time_str = config["schedules"]["fixed_line_end_time"]
    end_time_dt = dateparser.parse(f"{config_date_str} {end_time_str}")
    
    start_of_day_dt = start_time_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    logger.info(f"Start and end times:{start_time_dt} and {end_time_dt}")

    people_at_destination = 0
    total_people_served = 0
    total_people_picked = 0
    total_passengers_from_county = employee_arrivals['count'].sum()

    logger.debug(f"Need to transfer: {total_passengers_from_county} people")
    number_of_stops = len(county_stops)
    logger.debug(f"Total stops: {number_of_stops}")
    logger.debug(f"End time: {end_time_dt + dt.timedelta(hours=1)}")

    current_time_dt = start_time_dt

    fixed_line_results = []
    while (len(events) > 0) & (current_time_dt < (end_time_dt + dt.timedelta(hours=1))):
        new_events = []
        event = events.pop(0)
        event_type = event.event_type

        current_time_dt = event.time
        tempdf = employee_arrivals.loc[employee_arrivals['adjusted_scheduled_time_dt'] <= current_time_dt]
        
        people_currently_at_stop = 0
        people_id_currently_at_stop = []
        if not tempdf.empty:
            people_currently_at_stop = tempdf['count'].sum() - total_people_picked
            people_id_currently_at_stop = tempdf['list'].explode().tolist()[total_people_picked:]

        event_info = event.type_specific_information
        current_stop = event_info['current_stop']
        bus_id = event_info['bus_id']
        # logger.debug(f"@{current_time_dt} and {people_currently_at_stop} people")
        current_stop = current_stop % number_of_stops
        
        bus = buses[bus_id]
        bus.current_stop = current_stop

        if event_type == constants.EVENT_PICKUP:
            free_seats = bus.capacity - bus.load
            stop_name = county_stops.iloc[current_stop]['source']
            next_stop = county_stops.iloc[current_stop]['target']

            picked_up = 0
            if people_currently_at_stop > 0:
                picked_up = min(free_seats, people_currently_at_stop)
                bus.load += picked_up
                total_people_picked += picked_up
                bus.current_passenger_ids = people_id_currently_at_stop

            travel_time_to_next_stop = county_stops.iloc[current_stop]['duration_s']
            distance_to_next_stop = county_stops.iloc[current_stop]['distance_m']
            
            time_at_next_stop = current_time_dt + dt.timedelta(seconds=travel_time_to_next_stop)
            new_event = Event(event_type=constants.EVENT_DROPOFF, time=time_at_next_stop, 
                        type_specific_information={"bus_id":bus_id, "current_stop": current_stop + 1})
            new_events.append(new_event)

            # Results
            res = {"bus_id": bus_id, "current_stop": stop_name, "next_stop": next_stop, "action": "pickup",
                   "ons": picked_up, "offs":0, "current_time": (current_time_dt - start_of_day_dt).total_seconds(),
                   "served_user_ids": bus.current_passenger_ids}
            fixed_line_results.append(res)

            if "boc_conex" not in stop_name:
                logger.debug(f"bus: {bus_id}, pick {picked_up} people @ {current_time_dt} on {stop_name}")
                # print(f"remaining: {people_at_stop}")

        elif event_type == constants.EVENT_DROPOFF:
            stop_name = county_stops.iloc[current_stop]['source']
            next_stop = county_stops.iloc[current_stop]['target']
            
            new_event = Event(event_type=constants.EVENT_PICKUP, time=current_time_dt, 
                        type_specific_information={"bus_id":bus_id, "current_stop": current_stop})
            new_events.append(new_event)
            # if 'boc_conex' == stop_name:
            logger.debug(f"bus: {bus_id}, drop {bus.load} people @ {current_time_dt} on {stop_name}")
            dropped_off = bus.load
            passenger_ids = bus.current_passenger_ids
            bus.load = 0
            bus.current_passenger_ids = []
            people_at_destination += dropped_off
            total_people_served += dropped_off

            res = {"bus_id": bus_id, "current_stop": stop_name, "next_stop": next_stop, "action": "dropoff",
                   "ons": 0, "offs":dropped_off, "current_time": (current_time_dt - start_of_day_dt).total_seconds(),
                   "served_user_ids": passenger_ids}
            
            fixed_line_results.append(res)

        for ne in new_events:
            events.append(ne)
        events.sort(key=lambda x: (x.time, x.event_type), reverse=False)

    logger.info(f"Done simulating fixed line for {county.title()}")
    tdf = employee_arrivals.loc[(employee_arrivals['countyname'] == county.upper())]
    served_users_in_county = tdf['list'].explode().tolist()[:total_people_served]
    logger.debug(f"Served {len(served_users_in_county)}. {served_users_in_county}")

    fixed_line_results = pd.DataFrame(fixed_line_results)

    result = {
        "countyname": county.upper(),
        "served": total_people_served,
        "requests": total_passengers_from_county,
        "served_ids": served_users_in_county,
        "pct_served": total_people_served / total_passengers_from_county * 100
    }

    return fixed_line_results

def get_datetime(current_date_dt, x):
    return current_date_dt + dt.timedelta(seconds=x)

def main(config, county_requests):
    '''
    Runs multiple counties in parallel then returns aggregate datasets.
    Returns (counties, fixed_line_results, on_demand_results, per_user_results)
    '''
    counties = list(config["counties"].keys())
    # counties = ["CHESTER"]
    if "SHELBY" in counties:
        counties.remove("SHELBY")
    print(f"Starting offline solver for: {counties}")

    args = []
    for county in counties:
        if len(county_requests[county].index) > 0:
            logger.debug(f"Total requests for {county} is {len(county_requests[county].index)}")
            args.append((county, county_requests[county], config))

    # # Create a pool of worker processes
    with mp.Pool() as pool:
        # Apply the function to the arguments in parallel
        simulation_results = pool.starmap(solve_requests, args)

    fixed_line_results = []
    on_demand_results = []
    per_user_results = []
    # Save the results separately per process
    for i, (f, o, u) in enumerate(simulation_results):
        fixed_line_results.append(f)
        on_demand_results.append(o)
        per_user_results.append(u)
    
    fixed_line_results = pd.concat(fixed_line_results)
    on_demand_results = pd.concat(on_demand_results)
    per_user_results = pd.concat(per_user_results)

    return counties, fixed_line_results, on_demand_results, per_user_results

def compute_final_metrics(config, county, county_requests, on_demand_result, fixed_line_result):
    '''
    Returns (fixed_line_metrics, on_demand_metrics, per_user_metrics)
    '''
    EXPERIMENT_NAME = config["params"]["output_name"]

    travel_time_distance_matrix_file = config["params"]["travel_time_distance_matrix_file"]
    tt_d_df = gpd.read_file(f"./data/{EXPERIMENT_NAME}/{travel_time_distance_matrix_file}")

    def get_travel_time_distance_for_fixed_line(row):
        data = tt_d_df[(tt_d_df['source'] == row['current_stop']) & \
                       (tt_d_df['target'] == row['next_stop'])]
        if data.empty:
            raise f"County: {county}: Source and target: {row['current_stop']}, {row['next_stop']} not found."
        data = data.iloc[0]
        return data['duration_s'], data['distance_m']
    
    # per fixed line bus metrics
    fixed_line_metrics = []

    if not fixed_line_result.empty:
        # Adding distance and travel times?
        fixed_line_result[["duration_s", "distance_m"]] = fixed_line_result.apply(get_travel_time_distance_for_fixed_line, axis=1, result_type="expand")
        fixed_line_result["type"] = constants.EXPRESS_BUS

        for bus_id, df in fixed_line_result.groupby("bus_id"):
            fixed_line_served = df["offs"].sum()
            fixed_line_duration = df['duration_s'].sum()
            tdf = df.drop_duplicates(subset=["current_time"])
            fixed_line_VMT = tdf['distance_m'].sum() / constants.METER_IN_MILES
            fixed_line_PMT = fixed_line_served * tdf['distance_m'].iloc[0] / constants.METER_IN_MILES
            fixed_line_metric= {"county": county, 
                                 "run_id": bus_id, 
                                 "served": fixed_line_served,
                                 "requests": -1,
                                 "vmt": fixed_line_VMT, 
                                 "pmt": fixed_line_PMT
                                 }
            fixed_line_metrics.append(fixed_line_metric)

    fixed_line_metrics = pd.DataFrame(fixed_line_metrics)

    # TODO: Remove when done debugging
    on_demand_result = pd.read_csv(f"./output/TEST/{county}/results.csv")

    if (on_demand_result is not None):
        if (not on_demand_result.empty):
            on_demand_metrics = []
            per_user_on_demand_metrics = []
            for (countyfp, run_id), v in on_demand_result.groupby(["COUNTYFP", "run_id"]):
                v = v.reset_index(drop=True).sort_values(by="scheduled_time")

                # Served count
                served_count = v.query("action == 'dropoff'").shape[0]

                v.loc[0, "lat"] = v.iloc[0].h_lat
                v.loc[0, "lon"] = v.iloc[0].h_lon

                v.loc[v.shape[0]-1, "lat"] = v.iloc[0].h_lat
                v.loc[v.shape[0]-1, "lon"] = v.iloc[0].h_lon

                v['next_lat'] = v['lat'].shift(-1)
                v['next_lon'] = v['lon'].shift(-1)
                # drop the last row since it is redundant.
                v = v[:-1]
                # For vehicle miles traveled.
                v[['_', 'duration', 'distance']] = v.apply(generate_metrics_for_on_demand, axis=1, result_type="expand")
                
                # For passenger miles traveled
                user_df = v.query("action == 'pickup'").copy()
                user_df = user_df.drop(columns=["lat", "lon", "next_lat", "next_lon"])
                user_df = user_df.rename(columns={"h_lat": "lat", "h_lon": "lon", "w_lat": "next_lat", "w_lon": "next_lon"})
                user_df[['_', 'duration', 'distance']] = user_df.apply(generate_metrics_for_on_demand, axis=1, result_type="expand")

                # For gathering per user metrics
                tempdf = on_demand_result[on_demand_result['action'].isin(["pickup", "dropoff"])][['user_id', 'scheduled_time']]
                tempdf = tempdf.merge(user_df, on=["user_id", "scheduled_time"], suffixes=["_x", "_user"], how="outer")
                for user_id, df in tempdf.groupby("user_id"):
                    arrival_time = df.iloc[-1]['scheduled_time']
                    departure_time = df.iloc[0]['scheduled_time']
                    travel_duration_s = arrival_time - departure_time
                    travel_distance_m = df.iloc[0]['distance']
                    res = {
                            "county": county,
                            "bus_id": run_id,
                            "user_id": user_id, 
                            "departure_time": departure_time,
                            "arrival_time": arrival_time,
                            "travel_duration_s":travel_duration_s, 
                            "distance_m":travel_distance_m,
                            "type": constants.ON_DEMAND
                            }
                    per_user_on_demand_metrics.append(res)

                county_run_metric = {
                                    "county": county,
                                    "run_id": run_id, 
                                    "served": served_count,
                                    "requests": -1,
                                    "vmt": v['distance'].sum() / constants.METER_IN_MILES, 
                                    "pmt": user_df['distance'].sum() / constants.METER_IN_MILES,
                                    "countyfp": countyfp, 
                                    }
                on_demand_metrics.append(county_run_metric)
            on_demand_metrics = pd.DataFrame(on_demand_metrics)
    else:
        logger.error("No on demand result provided.")

    # per user metrics
    per_user_metrics = []

    # Fixed line user metrics: Only if fixed_line was used
    tdf = fixed_line_result.copy()
    tdf = tdf.explode('served_user_ids')
    tdf = tdf.reset_index(drop=True)
    tdf = tdf.dropna(subset=["served_user_ids"])
    for user_id, user_df in tdf.groupby("served_user_ids"):
        arrival_time = user_df.iloc[-1]['current_time']
        departure_time = user_df.iloc[0]['current_time']
        travel_duration_s = arrival_time - departure_time
        travel_distance_m = user_df.iloc[0]['distance_m']
        fixed_line_user = {"county": county,
                           "bus_id": user_df.iloc[0].bus_id,
                           "user_id": user_id, 
                           "departure_time": departure_time,
                           "arrival_time": arrival_time,
                           "travel_duration_s":travel_duration_s, 
                           "distance_m":travel_distance_m,
                           "type": constants.EXPRESS_BUS}

        per_user_metrics.append(fixed_line_user)
    
    per_user_metrics.extend(per_user_on_demand_metrics)
    per_user_metrics = pd.DataFrame(per_user_metrics)

    return fixed_line_metrics, on_demand_metrics, per_user_metrics

# TODO: Parameterize the times for GTFS etc...
# TODO: DST is messing up my time zones (kepler is not helping)
def generate_kepler_geojson(config, counties=["CHESTER"], per_user_metrics=None, county_requests=None, all_results=None):
    '''
    returns (mata_geo_json, on_demand_geo_json, express_geo_json, requests_with_isServed_bool)

    '''
    USE_CITY_BUSES_TO_TRANSIT_HUB = config["params"]["use_city_buses"]
    ON_DEMAND_TO_TRANSIT_HUB = config["params"]["to_transit_hub"]

    EXPERIMENT_NAME = config["params"]["output_name"]
    SCENARIO_OUTPUT = f"./output/{EXPERIMENT_NAME}"

    TRAVEL_TIME_DISTANCE_MATRIX_FILE = config["params"]["travel_time_distance_matrix_file"]
    TRAVEL_TIME_DISTANCE_MATRIX_FILE = f"./data/{EXPERIMENT_NAME}/{TRAVEL_TIME_DISTANCE_MATRIX_FILE}"
    FIXED_LINE_FREQUENCY_MIN = int(config["schedules"]["fixed_line_frequency_min"])
    window_list = ['0:00-6:00', '6:00-9:00']
    earliest_run = 18000 # 5 am
    date_str = config["params"]["current_date"]
    date = dateparser.parse(date_str)
    date = date.replace(hour=1, minute=0)

    # Fixed line
    fixed_line_start_time = config["schedules"]["fixed_line_start_time"]
    fixed_line_start_time = dateparser.parse(f"{date_str} {fixed_line_start_time}") - dt.timedelta(hours=4)

    ######## MATA GEOJSON ########
    mata_geo_json = {}
    if USE_CITY_BUSES_TO_TRANSIT_HUB:
        # It also works with URL's
        gtfs_path = './data/GTFS_MATA.zip'

        feed = Feed(gtfs_path, busiest_date=False,)
        lines_freq = feed.lines_freq
        stop_freq = feed.stops_freq

        condition_window = lines_freq.window.isin(window_list)
        gdf = lines_freq.loc[(condition_window),:].reset_index()
        segments_gdf = feed.segments

        route_ids = gdf.route_id.unique().tolist()

        all_routes_and_trips = []
        for route_id in route_ids:
        # go through each trip_id group and create their segment geojson
            sdf = get_stops_df(feed=feed, route_id=route_id, window_list=window_list, earliest_time=earliest_run)
            route_segments_df = segments_gdf[segments_gdf['route_id'] == route_id]
            for k, v in sdf.groupby("trip_id"):
                v['start_stop_id'] = v['stop_id']
                v['end_stop_id'] = v['stop_id'].shift(-1)
                v['travel_time'] = v['arrival_time'].shift(-1) - v['arrival_time']
                v['travel_time'] = v['travel_time'].fillna(0.0)
                v = v.merge(route_segments_df[['shape_id', 'stop_sequence', 'start_stop_id', 'distance_m', 'geometry']], on=['shape_id', 'stop_sequence', 'start_stop_id'])
                all_routes_and_trips.append(v)
                
        all_routes_and_trips_df = pd.concat(all_routes_and_trips)
        all_routes_and_trips_df = gpd.GeoDataFrame(all_routes_and_trips_df, geometry='geometry')
        all_routes_and_trips_df.plot(figsize=(2, 2))

        mata_geo_json = dict(type="FeatureCollection", features=[])

        for (route_id, trip_id), v in all_routes_and_trips_df.groupby(["route_id", "trip_id"]):
            v = v.sort_values(by="arrival_time", ascending=True)
            feature = dict(type="Feature", geometry=None, properties=dict(route_id=str(route_id), trip_id=str(trip_id)))
            
            data = []
            for j, w in v.iterrows():
                geom = w['geometry']
                duration = w['travel_time']
                time_per_seg = duration/len(geom.coords)
                timestamp = (date + dt.timedelta(seconds=w['arrival_time']) - dt.timedelta(hours=5)).timestamp()
                if timestamp > fixed_line_start_time.timestamp():
                    continue
                for coord in geom.coords:
                    timestamp += time_per_seg
                    data.append([coord[0], coord[1], 0, int(timestamp)])
                feature["geometry"] = dict(type="LineString", coordinates=data)
            mata_geo_json["features"].append(feature)
    
    ######## ON DEMAND GEOJSON ########
    combined_result = read_results_csv(SCENARIO_OUTPUT, filename='results.csv')
    combined_result['scheduled_time_dt'] = combined_result['scheduled_time'].apply(lambda x: get_datetime(date, x))

    on_demand_routes_df = []
    on_demand_geo_json = {}
    for (county, run_id), v in combined_result.groupby(["COUNTYFP", "run_id"]):
        v = v.reset_index(drop=True).sort_values(by="scheduled_time")
        v.loc[0, "lat"] = v.iloc[0].h_lat
        v.loc[0, "lon"] = v.iloc[0].h_lon

        v.loc[v.shape[0]-1, "lat"] = v.iloc[0].h_lat
        v.loc[v.shape[0]-1, "lon"] = v.iloc[0].h_lon

        v['next_lat'] = v['lat'].shift(-1)
        v['next_lon'] = v['lon'].shift(-1)
        v = v[:-1]

        v[['geometry', 'duration', 'distance']] = v.apply(generate_metrics_for_on_demand, axis=1, result_type="expand")
        v.loc[v.index[-1], 'action'] = "depot_end"
        on_demand_routes_df.append(v[['COUNTYFP', 'run_id', 'lat', 'lon', 'next_lat', 'next_lon', 'geometry', 'duration', 'distance', 'action', "scheduled_time"]])
    on_demand_routes_df = pd.concat(on_demand_routes_df)
    on_demand_routes_df["run_id"] = on_demand_routes_df["run_id"].astype('str')
    on_demand_geo_json = dict(type="FeatureCollection", features=[])

    for (county, run_id), v in on_demand_routes_df.groupby(["COUNTYFP", "run_id"]):
        v = v.sort_values(by="scheduled_time", ascending=True)
        feature = dict(type="Feature", geometry=None, properties=dict(run_id=run_id, county=county))
        
        data = []
        timestamp = date.timestamp()
        for j, w in v.iterrows():
            geom = w['geometry']
            duration = w['duration']
            time_per_seg = duration/len(geom.coords)
            for coord in geom.coords:
                timestamp += time_per_seg
                data.append([coord[0], coord[1], 0, int(timestamp)])
            feature["geometry"] = dict(type="LineString", coordinates=data)
        on_demand_geo_json["features"].append(feature)

    ######## FIXED LINE GEOJSON ########
    express_geo_json = {}
    if ON_DEMAND_TO_TRANSIT_HUB:
        df = gpd.read_file(TRAVEL_TIME_DISTANCE_MATRIX_FILE)

        all_county_buses_schedule_df = []
        for county, v in df.groupby("county"):
            bus_count = config["fixed_line_assets"][county.upper()]["count"]
            for bus_id in range(bus_count):
                v['run_id'] = bus_id
                all_county_buses_schedule_df.append(v.copy())
        all_county_buses_schedule_df = pd.concat(all_county_buses_schedule_df)
        
        express_geo_json = dict(type="FeatureCollection", features=[])

        for (county, run_id), v in all_county_buses_schedule_df.groupby(["county", "run_id"]):
            for bus_id in range(bus_count):
                feature = dict(type="Feature", geometry=None, properties=dict(run_id=str(run_id), county=str(county)))
                
                data = []
                timestamp = fixed_line_start_time + dt.timedelta(minutes=(FIXED_LINE_FREQUENCY_MIN * bus_id))
                timestamp = timestamp.timestamp()
                for j, w in v.iterrows():
                    geom = w['geometry']
                    duration = w['duration_s']
                    time_per_seg = duration/len(geom.coords)
                    for coord in geom.coords:
                        timestamp += time_per_seg
                        data.append([coord[0], coord[1], 0, int(timestamp)])
                    feature["geometry"] = dict(type="LineString", coordinates=data)
                express_geo_json["features"].append(feature)

    ######## HUBS and PICKUP POINTS ########
    named_stops = pd.DataFrame(config["named_stops"])
    named_stops = named_stops.T.reset_index()
    named_stops = named_stops.rename(columns={"index":"county"})
    named_stops['icon'] = "place"
    ######## USER REQUESTS DF ########
    served_users = per_user_metrics
    served_users['user_id'] = served_users['user_id'].astype('int')
    
    displayed_requests = []
    for county in counties:
        displayed_requests.append(county_requests[county])
    displayed_requests = pd.concat(displayed_requests)
    displayed_requests["is_served"] = False
    displayed_requests.loc[displayed_requests['user_id'].isin(served_users['user_id'].unique().tolist()), "is_served"] = True

    return mata_geo_json, on_demand_geo_json, express_geo_json, named_stops, displayed_requests

# TODO: If to_transit_hub == False, skip this, but compute the metrics
def convert_on_demand_results_to_transit_arrivals(config, county):
    EXPERIMENT_NAME = config["params"]["output_name"]

    BASE_DATA = f"data/{EXPERIMENT_NAME}/{county}"
    OUTPUT_DIR = f"output/{EXPERIMENT_NAME}/{county}"

    CURRENT_DATE_STR = config["params"]["current_date"]
    CURRENT_DATE_DT = dateparser.parse(CURRENT_DATE_STR)

    COUNTYNUM_TO_NAME = {v: k for k, v in config["counties"].items()}
    results = pd.read_csv(f"{OUTPUT_DIR}/results.csv")
    results['COUNTYFP'] = results['COUNTYFP'].bfill()
    results['COUNTYFP'] = results['COUNTYFP'].ffill()

    results['scheduled_time_dt'] = results['scheduled_time'].apply(lambda x: get_datetime(CURRENT_DATE_DT, x))

    results = results[results['action'] == 'dropoff']
    # Assign what bus they would be able to take based on arrival time.
    results['adjusted_scheduled_time_dt'] = results['scheduled_time_dt'].dt.ceil('15min')
    results['COUNTYFP'] = results['COUNTYFP'].astype('int').astype('str').str.zfill(3)
    results['countyname'] = results['COUNTYFP'].map(COUNTYNUM_TO_NAME)

    employee_arrival_schedules = results.groupby(['countyname', 'adjusted_scheduled_time_dt']).agg({"user_id":["count", list]}).reset_index().rename({"user_id":"count"}, axis=1)
    employee_arrival_schedules.columns = ['_'.join(col).strip() for col in employee_arrival_schedules.columns.values]
    employee_arrival_schedules = employee_arrival_schedules.rename(columns={"countyname_":"countyname", "adjusted_scheduled_time_dt_":"adjusted_scheduled_time_dt", "count_count":"count", "count_list":"list"})
    employee_arrival_schedules['COUNTYFP'] = employee_arrival_schedules['countyname'].map(constants.COUNTY_FPS_MAP)
    employee_arrival_schedules['adjusted_scheduled_time_str'] = employee_arrival_schedules['adjusted_scheduled_time_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # TODO: For later (metrics)
    # requests = pd.read_csv(f"{OUTPUT_DIR}/requests.csv")
    return employee_arrival_schedules

def recreate_fixed_line_bus_events(config, county):
    '''
    Returns buses: list, events: list
    '''
    current_date = config["params"]["current_date"]
    ### Populate pickup and drop off events.
    headway_frequency = config["schedules"]["fixed_line_frequency_min"]
    headway_frequency = int(headway_frequency)
    headway_frequency = dt.timedelta(minutes=headway_frequency)

    bus_capacity = config["fixed_line_assets"][county.upper()]["capacity"]
    bus_count = config["fixed_line_assets"][county.upper()]["count"]
    start_time = config["schedules"]["fixed_line_start_time"]
    end_time = config["schedules"]["fixed_line_end_time"]

    start_time = dateparser.parse(f"{current_date} {start_time}")
    end_time = dateparser.parse(f"{current_date} {end_time}")

    buses = []
    events = []
    current_time = start_time
    for bus_id in range(bus_count):
        event = Event(event_type=constants.EVENT_PICKUP, 
                      time=current_time, 
                      type_specific_information={"bus_id":bus_id, "current_stop": 0})
        events.append(event)
        current_time += headway_frequency

        # TODO: Add while loop, while bus time is not end?
        bus = ExpressBus(capacity=bus_capacity, current_stop=0)
        buses.append(bus)

    events.sort(key=lambda x: (x.time, x.event_type), reverse=False)
    
    return buses, events

def generate_traveltime_and_distance_matrices_for_fixed_line(config):
    '''
    zip the fixed line stops. it should go back and forth.
    Loop until the end of the travel times.??
    linestring, duration, distance = get_linestring_duration_distance_for_OD_pair_dict(source, target)
    '''
    EXPERIMENT_NAME = config["params"]["output_name"]
    counties = config['counties']
    named_stops = config["named_stops"]
    travel_time_distance_matrix_file = config["params"]["travel_time_distance_matrix_file"]
    def pair_and_cycle_stops():
        fixed_stops_travel_distance_matrix = []
        for county in counties:
            county_stops = config["fixed_line_transit_hubs"][county]
            # print(county_stops)
            pairs = zip(county_stops, county_stops[1:])
            pairs = list(pairs)
            reverse_pairs = zip(county_stops[::-1], county_stops[::-1][1:])
            reverse_pairs = list(reverse_pairs)
            pairs.extend(reverse_pairs)

            for ip, pair in enumerate(pairs):
                source = named_stops[pair[0]]
                target = named_stops[pair[1]]
                ls, dur, dist = get_linestring_duration_distance_for_OD_pair_dict(source, target)
                print(pair[0], pair[1])
                print(dur, dist)
                solution = {"idx": ip,
                            "source": pair[0],
                            "target": pair[1],
                            "county": county,
                            "geometry": ls,
                            "duration_s": dur,
                            "distance_m": dist}
                
                fixed_stops_travel_distance_matrix.append(solution)

        fixed_stops_travel_distance_matrix = pd.DataFrame(fixed_stops_travel_distance_matrix)
        fixed_stops_travel_distance_matrix = gpd.GeoDataFrame(fixed_stops_travel_distance_matrix,
                                                              geometry="geometry")
        fixed_stops_travel_distance_matrix.to_file(f"./data/{EXPERIMENT_NAME}/{travel_time_distance_matrix_file}")
    
    pair_and_cycle_stops()
    logger.info("Finished generating matrices.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to JSON config file', default="./data/configs/S1.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    EXPERIMENT_NAME = config["params"]["output_name"]
    Path(f"./data/{EXPERIMENT_NAME}").mkdir(parents=True, exist_ok=True)

    # Use config dictionary here

    if config["params"]["regenerate_matrix"]:
        res = generate_traveltime_and_distance_matrices_for_fixed_line(config)
        if not res:
            raise Exception("Travel time and distance matrices not available.")

    # Setup of input parameters
    # This might be too large to pass around if the data grows.
    county_requests = setup_input_jsons(config)

    logger.info(f"Solving on-demand requests offline: {config['counties'].keys()}.")
    counties, fixed_line_results, on_demand_results, per_user_results = main(config, county_requests)
    
    # ON_DEMAND_TO_TRANSIT_HUB = config["params"]["to_transit_hub"]
    # if ON_DEMAND_TO_TRANSIT_HUB:
    #     employee_arrivals = convert_on_demand_results_to_transit_arrivals(config, "CHESTER")
    # employee_arrivals.to_csv("checkthis.csv")
    # fixed_line_result = run_fixed_line_simulator(employee_arrivals, config, "CHESTER")
    # fixed_line_metrics, on_demand_metrics, per_user_metrics = compute_final_metrics(config=config, county="CHESTER", 
    #                       county_requests=None,
    #                       on_demand_result=None, fixed_line_result=fixed_line_result)

    mata, on_demand, fixed_line, named_stops, displayed_requests = generate_kepler_geojson(config, counties=counties, 
                                                                                           county_requests=county_requests, 
                                                                                           per_user_metrics=per_user_results)
    
    logger.info(f"Saving results to: output/{EXPERIMENT_NAME}")
    fixed_line_results.to_csv(f"./output/{EXPERIMENT_NAME}/fixed_line_results.csv")
    on_demand_results.to_csv(f"./output/{EXPERIMENT_NAME}/on_demand_results.csv")
    per_user_results.to_csv(f"./output/{EXPERIMENT_NAME}/per_user_results.csv")

    # print(mata)
    kepler_json_file = config["visualization"]["kepler_config"]

    with open(f"./data/{kepler_json_file}") as f:
        kepler_config = json.load(f)

    my_map = KeplerGl(data={
                            "on-demand":on_demand, 
                            "fixed-line":fixed_line,
                            "mata": mata, 
                            "requests":displayed_requests, 
                            "hubs": named_stops
                            }, 
                            config=kepler_config,
                            height=800, width=300)

    my_map.save_to_html(file_name=f'./output/{EXPERIMENT_NAME}/kepler_{EXPERIMENT_NAME.split("/")[-1]}.html')