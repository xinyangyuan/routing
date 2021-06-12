from os import path
import sys, json, time
import datetime

import numpy as np
import pandas as pd

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# initialize data and output
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

new_package_path = path.join(BASE_DIR, "data/model_apply_inputs/new_package_data.json")
new_routes_path = path.join(BASE_DIR, "data/model_apply_inputs/new_route_data.json")
new_time_path = path.join(BASE_DIR, "data/model_apply_inputs/new_travel_times.json")

print (new_package_path)


df_package = pd.read_json(new_package_path).transpose()
df_route = pd.read_json(new_routes_path).transpose()
df_time = pd.read_json(new_time_path).transpose()
output_dict = {}


# iterate through model_apply routes
for k in range(len(df_route)):
# for k in range(1):
    # obtain required data:
    # route_no, station_no, departure_time, num_stops
    route_no = df_route.index[k]
    keys = df_route.loc[route_no]["stops"].keys()
    mydict = df_route.loc[route_no]["stops"]
    for _ in mydict.keys():
        if mydict[_]["type"] == "Station":
            station_code = _
    departure_time = df_route.loc[route_no][
        ["date_YYYY_MM_DD", "departure_time_utc"]
    ].values
    departure_time = departure_time[0] + " " + departure_time[1]
    departure_time = datetime.datetime.strptime(departure_time, "%Y-%m-%d %H:%M:%S")

    df_dict = df_time.loc[route_no][keys].to_dict()
    df_dist = pd.DataFrame.from_dict(df_dict)
    station_no = df_dist.index.get_loc(station_code)
    num_stops = len(df_dist)

    print("route number:", route_no)
    print("number of stops: ", num_stops)

    # obtain time window
    max_time = 86400
    # todo waiting time!!!!!
    wait_time = 60
    time_windows = []
    package_list = df_package.loc[route_no][keys]

    for i in list(package_list.keys()):
        start_time = departure_time
        end_time = departure_time + datetime.timedelta(seconds=max_time)

        for j in list(package_list[i]):
            start_time0 = package_list[i][j]["time_window"]["start_time_utc"]
            end_time0 = package_list[i][j]["time_window"]["end_time_utc"]
            if start_time0 != None:
                start_time1 = datetime.datetime.strptime(
                    start_time0, "%Y-%m-%d %H:%M:%S"
                )
                start_time = max(start_time, start_time1)
            if end_time0 != None:
                end_time1 = datetime.datetime.strptime(end_time0, "%Y-%m-%d %H:%M:%S")
                end_time = min(end_time, end_time1)

        time_windows.append(
            (
                max(0, int((start_time - departure_time).total_seconds())),
                max(0, int((end_time - departure_time).total_seconds())),
            )
        )

    # define model and solve
    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data["time_matrix"] = df_dist.to_numpy().tolist()
        data["time_windows"] = time_windows
        data["num_vehicles"] = 1
        data["depot"] = station_no
        return data

    def print_solution(data, manager, routing, solution):
        """Prints solution on console."""
        calculated_sequence = [0] * (num_stops)
        time_dimension = routing.GetDimensionOrDie("Time")
        for vehicle_id in range(data["num_vehicles"]):
            i = 0
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                calculated_sequence[index] = i
                index = solution.Value(routing.NextVar(index))
                i += 1
        return calculated_sequence

    def solve():
        """Solve the VRP with time windows."""
        # Instantiate the data problem.
        data = create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["depot"]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback.
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Time Windows constraint.
        time = "Time"
        routing.AddDimension(
            transit_callback_index,
            wait_time,  # allow waiting time
            max_time,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time,
        )
        time_dimension = routing.GetDimensionOrDie(time)
        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        # Add time window constraints for each vehicle start node.
        depot_idx = data["depot"]
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][depot_idx][0], data["time_windows"][depot_idx][1]
            )

        # Instantiate route start and end times to produce feasible times.
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i))
            )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        )
        search_parameters.time_limit.seconds = 5

        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.log_search = False

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            calculated_sequence = print_solution(data, manager, routing, solution)
            return calculated_sequence

    start = time.time()
    calculated_sequence = solve()
    end = time.time()
    print("Elapsed = %s" % (end - start))

    keys_list = df_dist.keys().to_list()
    values_list = calculated_sequence
    zip_itr = zip(keys_list, values_list)
    a_dict = dict(zip_itr)
    a_dict = dict(proposed=a_dict)
    output_dict[route_no] = a_dict

# Write output data
output_path = path.join(BASE_DIR, "data/model_apply_outputs/proposed_sequences.json")
with open(output_path, "w") as out_file:
    json.dump(output_dict, out_file)
print("Done!")
