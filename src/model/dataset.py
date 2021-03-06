import os
import json
import datetime
import random
import collections
import typing
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pytz
import timezonefinder
from scipy import spatial
from torch.utils.data import Dataset, DataLoader
from pandas.tseries.holiday import USFederalHolidayCalendar

class Sample(typing.NamedTuple):
    route_id:str    
    num_stops:int   
    input:np.ndarray              # (num_stops, num_stops, num_1d_features + num_2d_features)
    input_0d:np.ndarray           # (num_0d_features, )
    input_1d:np.ndarray           # (num_stops, num_1d_features)
    input_2d:np.ndarray           # (num_stops, num_stops, num_2d_features)
    stop_ids:typing.List[str]     # list of stop_id string in same order as input
    station_id:str                # the stop_id for station
    target:typing.Optional[np.ndarray]                 #  (num_stops, ) || None
    sequence:typing.Optional[typing.Dict[int,str]]     #  {stop_idx -> stop_id} || None
    sequence_map:typing.Optional[typing.Dict[str,int]] #  {stop_id -> stop_idx} || None


class RoutingDataset(Dataset):
    """Routing dataset."""

    def __init__(self, root_dir, stage="build", transform=None) -> None:
        """
        Args:
            root_dir (string): Directory with all training data.
            stage (string): Stages `build` or `apply` 
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert stage in ["build", "apply"], "stage is either `build` or `apply`"

        self.stage = stage
        self.root_dir = root_dir
        self.transform = transform

        if self.stage == "build":
            self.data_dir = os.path.join(root_dir, "model_build_inputs")

            self.route_df = self.get_route_df(path=os.path.join(self.data_dir, "route_data.json"))

            with open(os.path.join(self.data_dir, "travel_times.json")) as json_file:
                self.travel_times_dict = json.load(json_file)
            
            with open(os.path.join(self.data_dir, "package_data.json")) as json_file:
                self.package_dict = json.load(json_file)
            
            with open(os.path.join(self.data_dir, "actual_sequences.json")) as json_file:
                self.actual_sequences = json.load(json_file)
        else:
            self.data_dir = os.path.join(root_dir, "model_apply_inputs")

            self.route_df = self.get_route_df(path=os.path.join(self.data_dir, "new_route_data.json"))

            with open(os.path.join(self.data_dir, "new_travel_times.json")) as json_file:
                self.travel_times_dict = json.load(json_file)
            
            with open(os.path.join(self.data_dir, "new_package_data.json")) as json_file:
                self.package_dict = json.load(json_file)

    def __len__(self) -> int:
        return len(self.route_df)

    def __getitem__(self, idx) -> Sample:
        """
        Get training sample by index.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            sample: (Sample) one training sample
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # prepare route-related information
        route = self.route_df.iloc[idx]
        route_id = self.route_df.index[idx] # <str> 'RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77'

        # preapare stop-related information
        stops = route.stops # dict
        num_stops = len(stops)
        stop_ids =  list(stops.keys()) # str[] list of stop_id string in same order as input
        station_id = [stop_id for stop_id, stop in stops.items() if stop['type'] == 'Station'][0] # str
        stop_embeddings = self.get_stop_embeddings(stops) # list(np.ndarray)

        # prepare package-related information
        route_packages = self.package_dict[route_id] # a dict indexed by stop-id
        stop_package_embeddings = self.get_stop_package_embeddings(route, stops, route_packages) # list(np.ndarray)

        # prepare travel-time information
        travel_times = self.get_travel_times_arr(self.travel_times_dict[route_id]) # np.ndarray (num_stops, num_stops)

        # prepare lat and lng matrix
        lat_matrix, lng_matrix, travel_distance = self.get_lat_lng_distance_matrices(stops)

        # inputs/features
        input_0d = self.get_route_embedding(route) # (num_0d_features, )

        input_1d = np.column_stack((
            stop_embeddings,        # list[np.ndarray]
            stop_package_embeddings # list[np.ndarray]
        )) # (num_stops, num_1d_features)

        input_2d = np.dstack((
            travel_times,    # np.ndarray (num_stops, num_stops)
            lat_matrix,      # np.ndarray (num_stops, num_stops)
            lng_matrix,      # np.ndarray (num_stops, num_stops)
            travel_distance, # np.ndarray (num_stops, num_stops)
        )) # (num_stops, num_stops, num_2d_features)

        # prepare actual sequence
        if self.stage == "build":
            sequence_map = self.actual_sequences[route_id]["actual"] # {str -> int} stop_id string -> index in actual sequence
            sequence = {stop_idx : stop_id for stop_id, stop_idx in sequence_map.items()} # {int -> str} index in actual sequence -> stop_id string
            target = np.array([
                stop_ids.index(sequence[stop_idx+1]) if stop_idx < (len(sequence_map)-1) else stop_ids.index(sequence[0]) 
                for stop_idx in sequence_map.values()
            ]) # np.ndarray (num_stops, ) 

        # training inputs and labels
        input = self.one_to_two_d_feature(input_1d)
        input = np.dstack((
            input,
            input_2d
        )) # (num_stops, num_stops, num_1d_features + num_2d_features)
        
        # return sample
        if self.stage == "build":
            sample = Sample(
                route_id,   # str 
                num_stops,  # int
                input,      # np.ndarray (num_stops, num_stops, num_1d_features + num_2d_features)
                input_0d,   # np.ndarray (num_0d_features, )
                input_1d,   # np.ndarray (num_stops, num_1d_features)
                input_2d,   # np.ndarray (num_stops, num_stops, num_2d_features)
                stop_ids,   # list[str] list of stop_id string in same order as input
                station_id, # str 
                target,     # np.ndarray (num_stops, )
                sequence,   # dict {stop_idx -> stop_id}
                sequence_map, # dict {stop_id -> stop_idx} 
            )
        else:
            sample = Sample(
                route_id,   # str 
                num_stops,  # int
                input,      # np.ndarray (num_stops, num_stops, num_1d_features + num_2d_features)
                input_0d,   # np.ndarray (num_0d_features, )
                input_1d,   # np.ndarray (num_stops, num_1d_features)
                input_2d,   # np.ndarray (num_stops, num_stops, num_2d_features)
                stop_ids,   # list[str] list of stop_id string in same order as input
                station_id, # str
                None,
                None,
                None,
            ) 

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    @property
    def lengths(self) -> typing.List[int]:
        """
        Return:
            lengths (list[int]): the sequence length for all routes in the dataset.
        """
        return [len(route_stops) for route_stops in self.route_df.stops]
    
    @staticmethod
    def get_route_df(path: str, transform:bool=True) -> pd.DataFrame:
        """
        Route dataframe
        Args:
            path (str): Directory of route data json
            transform (bool): Perform initial df tranform or not (default to True)
        Return:
            route_df (pd.Dataframe): Dataframe of route data
        """

        # df column names
        ROUTE_KEYS = {
            "station_code", 
            "date_YYYY_MM_DD", 
            "departure_time_utc", 
            "executor_capacity_cm3", 
            "route_score", 
            "stops", 
            "depature_timestamp_utc",
        }

        # read df from json (index oriented)
        route_df = pd.read_json(path, orient="index") 

        # transformation
        if transform:
            # (1) add depature_timestamp_utc column (dtype=int64)
            # https://stackoverflow.com/questions/40881876/python-pandas-convert-datetime-to-timestamp-effectively-through-dt-accessor
            route_df['depature_timestamp_utc'] = pd.to_datetime(route_df.date_YYYY_MM_DD + ' ' + route_df.departure_time_utc )
            route_df['depature_timestamp_utc'] = route_df['depature_timestamp_utc'].astype('int64')  // 10**9
            # (2) add route_score col if not exist (apply-dataset does not contain route-score)
            if "route_score" not in route_df:
                route_df["route_score"] = "High" 
        
        return route_df
    
    @staticmethod
    def get_route_embedding(route) -> np.ndarray:
        """
        Embedding of package among stops in the route
        Args:
            route (pd.Series): Pandas series store current route's info 
        Return:
            stop_embeddings (np.ndarray): route_embedding ndarray (21, )
        """

        # UTC timestamp 
        timestamp = route.depature_timestamp_utc # int
        date = datetime.datetime.fromtimestamp(timestamp) # datetime.datetime

        # Get local-time
        # (1) https://stackoverflow.com/questions/15742045/getting-time-zone-from-lat-long-coordinates
        tf = timezonefinder.TimezoneFinder()
        stop = list(route["stops"].values())[0] # first stop in the route
        lat, lng = stop["lat"], stop["lng"]
        timezone_str = tf.timezone_at(lat=lat, lng=lng) # str

        # (2) https://stackoverflow.com/questions/4563272/how-to-convert-a-utc-datetime-to-a-local-datetime-using-only-standard-library
        if timezone_str is not None:
            timezone = pytz.timezone(timezone_str) # DstTzInfo type
            date = date.replace(tzinfo=datetime.timezone.utc).astimezone(tz=timezone)

        # Extract useful features
        # (1) raw feature
        year = date.year
        month = date.month
        day_of_month = date.day
        hour = date.hour
        minute = date.minute
        second = date.second
        date_name = date.strftime("%A") # "Monday", "Tuesday"... "Sunday"
        day_part: typing.List[int] = RoutingDataset.daypart(hour)

        # (2) derived feature
        # holiday: https://stackoverflow.com/questions/2394235/detecting-a-us-holiday
        cal = USFederalHolidayCalendar()
        holidays: typing.List[datetime.datetime] = cal.holidays(start=f"{year}-01-01", end=f"{year}-12-31").to_pydatetime()
        is_holiday = True if date in holidays else False
        is_weekend = True if date_name in ["Saturday","Sunday"] else False
        departure_time_sec = second + minute * 60 + hour * 60 * 60

        return np.array((
            *RoutingDataset.hash_str_to_list(route.station_code), # list[int] -> int,int,int,int
            year,                               # int
            month,                              # int
            day_of_month,                       # int (1, 31)
            hour,                               # int (0, 23)
            *day_part,                          # List[int] -> int,int,int,int,int,int
            departure_time_sec,                 # int (0, 86400)
            int(is_weekend),                    # int (0, 1)
            int(is_holiday),                    # int
            route.executor_capacity_cm3,        # int
            int(route.route_score == "High"),   # int
            int(route.route_score == "Medium"), # int
            int(route.route_score == "Low"),    # int
        )) # (num_0d_features, )
    
    @staticmethod
    def get_stop_embeddings(stops: dict) -> typing.List[np.ndarray]:
        """
        Embedding of stops in the route
        Args:
            stops (dict): Dict stores all stops' information {stop_id -> {lat, lng, type, zone_id}}
        Return:
            stop_embeddings (np.ndarray[]): List of stop_embedding ndarray (1+1+27+1, )
        """
        stop_embeddings = []

        # loop through all stops in the route
        for stop in stops.values():
            # encode `type` (boolean: "Station" -> 1 "Dropoff" -> 0)
            stop_type = 1 if stop['type'] == "Station" else 0
            assert stop["type"] == "Station" or stop["type"] == "Dropoff"

            # encode `zone_id` <numpy array> (27, )
            zone_embedding = RoutingDataset.get_zone_id_embedding(stop["zone_id"])

            # append current stop's embedding to stop_embeddings list
            stop_embedding = np.concatenate((
                [stop["lat"]], # float
                [stop["lng"]], # float
                [stop_type],   # int
                zone_embedding # ndarray (27, )
            ))
            stop_embeddings.append(stop_embedding)
        
        return stop_embeddings

    @staticmethod
    def get_stop_package_embeddings(route, stops: dict, route_packages: dict) -> typing.List[np.ndarray]:
        """
        Embedding of package among stops in the route
        Args:
            route (pd.Series): Pandas series store current route's info 
            stops (dict): Dict stores all stops' information {stop_id -> {lat, lng, type, zone_id}}
            route_packages (dict): {stop_id -> {package_id -> {lat, lng, type, zone_id}}}
        Return:
            stop_embeddings (np.ndarray[]): List of stop_embedding ndarray (9, )
        """

        stop_package_embeddings = []

        # get local timezone
        # https://stackoverflow.com/questions/15742045/getting-time-zone-from-lat-long-coordinates
        tf = timezonefinder.TimezoneFinder() 
        stop = list(stops.values())[0] # first stop in the route
        lat, lng = stop["lat"], stop["lng"]
        timezone_str = tf.timezone_at(lat=lat, lng=lng) # str
        timezone = pytz.timezone(timezone_str) or datetime.timezone.utc

        # loop through all stops in the route
        for stop_id in stops.keys():
            stop_packages = route_packages[stop_id]
            
            # initialize stop package embedding
            num_packages = 0
            start_timestamp = route.depature_timestamp_utc
            end_timestamp = route.depature_timestamp_utc
            volume_cm3 = 0
            height_cm = 0
            width_cm = 0
            depth_cm = 0
            planned_service_time_seconds = 0
            scan_status = True # default to delivered
            
            # loop through all packages
            for package in stop_packages.values():

                # package information
                package_height = package["dimensions"]["height_cm"]
                package_width = package["dimensions"]["width_cm"]
                package_depth = package["dimensions"]["depth_cm"]
                package_volume = package_height * package_width * package_depth
                package_planned_service_time_seconds = package["planned_service_time_seconds"]
                if "scan_status" in package:
                    # set scan_status to False if any package fails to deliver
                    scan_status = (package["scan_status"] == "DELIVERED") and scan_status 

                # accumulate package info
                num_packages += 1
                volume_cm3 += package_volume
                height_cm += package_height
                width_cm += package_width
                depth_cm += package_volume
                planned_service_time_seconds += package_planned_service_time_seconds

                if not pd.isnull(package["time_window"]["start_time_utc"]):
                    start_time_str = package["time_window"]["start_time_utc"]
                    start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                    package_start_timestamp = start_time.timestamp()
                    start_timestamp = package_start_timestamp if start_timestamp == 0 else min(start_timestamp, package_start_timestamp)
                
                if not pd.isnull(package["time_window"]["end_time_utc"]):
                    end_time_str = package["time_window"]["end_time_utc"]
                    end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                    package_end_timestamp = end_time.timestamp()
                    end_timestamp = package_end_timestamp if end_timestamp == 0 else min(end_timestamp, package_end_timestamp)
            
            # Convert time-stamp
            # (1) time in utc
            start_time = datetime.datetime.fromtimestamp(start_timestamp)
            end_time = datetime.datetime.fromtimestamp(end_timestamp)
            
            # (2) time in local timezone
            start_time = start_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=timezone)
            end_time = end_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=timezone)
            
            start_time_sec = start_time.second + start_time.minute * 60 + start_time.hour * 3600
            end_time_sec = end_time.second + end_time.minute * 60 + end_time.hour * 3600 

            # append current stop's package embedding to stop_package_embeddings list
            stop_package_embedding = np.array([
                num_packages,                                 # int
                start_time_sec,                               # int
                end_time_sec,                                 # int
                volume_cm3/route.executor_capacity_cm3,       # float
                height_cm/route.executor_capacity_cm3,        # float
                width_cm/route.executor_capacity_cm3,         # float
                depth_cm/route.executor_capacity_cm3,         # float
                planned_service_time_seconds,                 # int
                int(scan_status)                              # int
            ])
            stop_package_embeddings.append(stop_package_embedding)

        return stop_package_embeddings
    
    @staticmethod
    def get_travel_times_arr(travel_times: dict) -> np.ndarray:
        """
        Get travel times matrix from travel times dictionary
        Args:
            travel_times (dict): Dict stores all stops' information {stop_id -> {stop_id -> time}}
        Return:
            travel_times_arr (np.ndarray[]): 2D array of travel time between stops (num_stops, num_stops)
        """
        travel_times_arr = [[travel_time for travel_time in stops.values()] for stops in travel_times.values()]
        return np.array(travel_times_arr)

    @staticmethod
    def get_zone_id_embedding(zone_id: typing.Optional[str]) -> np.ndarray:  
        """
        Embedding of zone-id
        Args:
            zone_id (string): Alpha-numerical identification of stop zone (format: A-NN.NA)
                              e.g., Z-12.3C
        Return:
            embedding (np.ndarray): One-hot encode the first letter + the numerical sub-zone (shape: (26+1, ))
                                    e.g., np.array([0,0,...,1,12])
        """
        if zone_id is None:
            zone_1 = 0
            zone_2 = 0 
        else:
            zone_list = zone_id.split("-")
            zone_1 = zone_list[0] # an alpha letter <str>
            try:
                zone_2 = int(zone_list[1].split(".")[0]) if len(zone_list) == 2 else 0
            except ValueError:
                zone_2 = 0
        
        mapping = {letter: idx for idx, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        zone_1_one_hot_embedding = np.zeros(len(mapping))

        if zone_1 != 0:
            try:
                letter_idx = mapping[zone_1]
                zone_1_one_hot_embedding[letter_idx] = 1
            except KeyError:
                pass

        return np.concatenate((zone_1_one_hot_embedding, [zone_2]))

    @staticmethod
    def one_to_two_d_feature(arr: np.ndarray) -> np.ndarray:
        """
        Convert 1d to 2d input features
        Args:
            arr (np.ndarray): 1D-input feature array (shape: (num_stops, X))
        Return:
            out (np.ndarray): Repeat 1D-input feature for num_stops times (shape: (num_stops, num_stops, X))
        """
        assert len(arr.shape) == 2, "one_to_two_feature expect input ndarray shape of (num_stops, X)"
        return np.repeat(arr[:, np.newaxis, :], arr.shape[0], axis=1)
    
    @staticmethod
    def get_lat_lng_distance_matrices(stops: dict) -> np.ndarray:
        """
        Get lat and lng matrices from stops dict
        Args:
            stops (dict): stops (dict): Dict stores all stops' information {stop_id -> {lat, lng, type, zone_id}}
        Return:
            lat_matrix (np.ndarray): 2D array of pair-wise lat difference (num_stops, num_stops)
            lng_matrix (np.ndarray): 2D array of pair-wise lat difference (num_stops, num_stops)
            travel_distance (np.ndarray): 2D array of pair-wise euclidean-distance (num_stops, num_stops)
        """
        num_stops = len(stops)
        lats = np.zeros(num_stops)
        lngs = np.zeros(num_stops)

        # loop through all stops in the route
        for idx, stop in enumerate(stops.values()):
            lats[idx] = stop["lat"]
            lngs[idx] = stop["lng"]
        
        coordinates = np.column_stack((lats, lngs)) # (num_stops, 2)

        # outputs
        lat_matrix = lats[:, np.newaxis] - lats[np.newaxis, :]
        lng_matrix = lngs[:, np.newaxis] - lngs[np.newaxis, :]
        travel_distance = spatial.distance.cdist(coordinates, coordinates)

        assert lat_matrix.shape == lng_matrix.shape == travel_distance.shape == (num_stops, num_stops)

        return lat_matrix, lng_matrix, travel_distance
    
    @staticmethod
    def hash_str_to_list(str:str, length:int=4) -> typing.List[float]:
        """
        Hash a string input, e.g., station_code "DSE4", to a list of floats.
        Args:
            str (string): A string want to encode
            length (int): The length of the output list.
        Return:
            out (List[float]): A list of hashed characters in string, if the str is shorter than length, pad with 0.
        """

        out = [0] * length

        for i, character in enumerate(str):
            out[i] = hash(character) % (2**16) / (2**16) # has range [0,1)

        return out
    
    @staticmethod
    def daypart(hour:int, one_hot:bool=True) -> typing.Union[str, typing.List[int]]:
        """
        Get day part of a day, e.g., Morning, Dawn
        Args:
            hour (int): A integer from 0-24 want to encode
            one_hot (bool): Return in string or one_hot format (default: True)
        Return:
            out (string || List[float]): Dawn or [1,0,0,0,0,0]; "Morning" or [0,1,0,0,0,0]
        """

        if hour in [2,3,4,5]:
            return [1,0,0,0,0,0] if one_hot else "Dawn" 
        elif hour in [6,7,8,9]:
            return [0,1,0,0,0,0] if one_hot else "Morning"
        elif hour in [10,11,12,13]:
            return [0,0,1,0,0,0] if one_hot else "Noon"
        elif hour in [14,15,16,17]:
            return [0,0,0,1,0,0] if one_hot else "Afternoon"
        elif hour in [18,19,20,21]:
            return [0,0,0,0,1,0] if one_hot else "Evening"
        else: 
            return [0,0,0,0,0,1] if one_hot else "Midnight"


class RandomPermute(object):
    """Permute randomly the stop sequence in a sample."""

    def __call__(self, sample:Sample) -> Sample:

        input_, input_2d, target, stop_ids = sample.input, sample.input_2d, sample.target, sample.stop_ids  

        # Permutated sequence
        permutation = np.random.permutation(len(stop_ids))

        # Permute orignial sample
        input_ = input_[permutation][:, permutation] # np.ndarray (num_stops, num_stops, num_1d_features + num_2d_features)
        input_2d = input_2d[permutation][:, permutation]  # np.ndarray (num_stops, num_stops, num_2d_features)
        target = target[permutation] # np.ndarray (num_stops, )
        target = np.array([np.where(permutation == stop)[0][0] for stop in np.nditer(target)]) # np.ndarray (num_stops, )
        stop_ids = [stop_ids[idx] for idx in np.nditer(permutation)] # list[str]

        return Sample(
                sample.route_id,   # str 
                sample.num_stops,  # int
                input_,                # np.ndarray (num_stops, num_stops, num_1d_features + num_2d_features)
                sample.input_0d,   # np.ndarray (num_0d_features, )
                sample.input_1d,   # np.ndarray (num_stops, num_1d_features)
                input_2d,             # np.ndarray (num_stops, num_stops, num_2d_features)
                stop_ids,             # list[str] list of stop_id string in same order as input
                sample.station_id, # str 
                target,               # np.ndarray (num_stops, )
                sample.sequence,   # dict {stop_idx -> stop_id}
                sample.sequence_map, # dict {stop_id -> stop_idx} 
            )


class BucketSampler(torch.utils.data.Sampler):
    """
    Collate function
    
    (1) separate dataset samples into buckets by length (num_stops of route)
    (2) return random sampled batches with similar length

    """
    
    def __init__(self, lengths, buckets=(0,500,25), shuffle=True, batch_size=32, drop_last=False):
        """
        Args:
            lengths (int[]): a list of route length
            buckets (tuple): (min_bucket_length, max_bucket_length, bucket_step_size)
            shuffle (bool): true -> random shuffle samples orders in buckets
            batch_size (int): size of batch to return when called
            drop_last (bool): true -> drop last batch 
        Return:
            None
        """
        
        super().__init__(lengths)
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        assert isinstance(buckets, tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0
        
        buckets = collections.defaultdict(list)
        for i, length in enumerate(lengths):
            if length > bmin:
                bucket_size = min((length // bstep) * bstep, bmax)
                buckets[bucket_size].append(i)
                
        self.buckets = dict()
        for bucket_size, bucket in buckets.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket, dtype=torch.int, device='cpu')
        
        # call __iter__() to store self.length
        self.__iter__()
            
    def __iter__(self):
        
        if self.shuffle == True:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]
                
        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket, self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket
            
        self.length = len(batches)
        
        if self.shuffle == True:
            random.shuffle(batches)
            
        return iter(batches)
    
    def __len__(self):
        return self.length


def get_collate_fn(stage="build", params=None):
    """
    Get the collate function that is used in dataloader.
    
    Args:
        stage (string): Stages `build` or `apply`
        params (Param)
    Return:
        collate_fn (function): A callable collate_fn
    """

    assert stage in ["build", "apply"], "stage is either `build` or `apply`" 

    def collate_fn(batch):
        """
        Collate function
        
        (1) padd batch input tensors
        (2) create masks for the input tensors

        Args:
            batch (Sample[]): a list of Sample namedtuples 
        Return:
            batch (dict): a dict of torch.Tensors or lists
        """

        # unpack Samples[]
        route_ids, num_stops, inputs, input_0ds, _, _, stop_ids, station_ids, targets, sequences, sequence_maps = zip(*batch)

        # cosntants
        max_num_stops = max(num_stops)
        ignore_index = -100 if params is None else params.ignore_index # for CrossEntropyLoss ignore_idx

        # pad input and targets
        # (1): (num_stops, num_stops, num_1d_features + num_2d_features) -> (max_num_stops, max_num_stops, num_1d_features + num_2d_features)
        inputs = [np.pad(input, ((0, max_num_stops-input.shape[0]), (0, max_num_stops-input.shape[1]), (0, 0)), mode='median') for input in inputs]
        # (2): (num_stops, )
        if stage == "build":
            targets = [np.pad(target, (0, max_num_stops-target.shape[0]), mode='constant', constant_values=ignore_index) for target in targets]

        # create torch tensors: inputs, input_0ds, targets
        inputs = torch.tensor(inputs, dtype=torch.float32)
        input_0ds = torch.tensor(input_0ds, dtype=torch.float32)
        if stage == "build":
            targets = torch.tensor(targets, dtype=torch.long) # float64 (n, max_num_stops)

        # create mask
        # masks = (targets != ignore_index).float()  # (n, max_num_stops) float32
        masks = (torch.arange(max_num_stops)[None,:] < torch.tensor(num_stops)[:,None]).float() # (n, max_num_stops) float32
        masks = torch.einsum('ij,ik->ijk', masks, masks) # (n, max_num_stops) -> (n, max_num_stops, max_num_stops)

        if stage == "build":
            return {
                'route_ids': route_ids,         # str[] 
                'num_stops': num_stops,         # int[]
                'inputs': inputs,               # torch.tensor[] (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
                'input_0ds': input_0ds,         # torch.tensor[] (n, num_0d_features)
                'masks': masks,                 # torch.tensor[] (n, max_num_stops, max_num_stops)
                'targets': targets,             # torch.tensor[] (n, max_num_stops)
                'stop_ids': stop_ids,           # str[][] (n, num_stops)
                'station_ids': station_ids,     # str[]  (n)
                'sequences': sequences,         # dict[] {stop_idx -> stop_id}
                'sequence_maps': sequence_maps, # dict[] {stop_id -> stop_idx}
            }
        else:
            return {
                'route_ids': route_ids,         # str[] 
                'num_stops': num_stops,         # int[]
                'inputs': inputs,               # torch.tensor[] (n, max_num_stops, max_num_stops, num_1d_features + num_2d_features)
                'input_0ds': input_0ds,         # torch.tensor[] (n, num_0d_features)
                'masks': masks,                 # torch.tensor[] (n, max_num_stops, max_num_stops)
                'stop_ids': stop_ids,           # str[][] (n, num_stops)
                'station_ids': station_ids,     # str[]  (n)
            }
    
    return collate_fn


def get_dataset(stages, data_dir=None):
    """
    Get the Dataset object for each stage in stages.
    Args:
        stages: (list[str]) has one or more of 'apply', 'build', depending on which data is required
        data_dir: (string) directory containing the dataset
    Returns:
        data: (dict) contains the Dataset object for each type in stages
    """

    datasets = {}

    if data_dir is None:
        data_dir = Path(__file__).parent / "../../data"

    for stage in ['build', 'apply']:
        transform = RandomPermute() if stage == 'build' else None
        if stage in stages:
            dataset = RoutingDataset(data_dir, stage, transform=transform)
            datasets[stage] = dataset

    return datasets


if __name__ == "__main__":
    print("Creating datasets...")
    datasets = get_dataset(["build", "apply"])
    print("Datasets created successfuly!")

    print("First sample from build dataset:", datasets["build"][0])