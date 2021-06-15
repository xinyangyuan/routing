import time
import datetime
import numpy as np
import pandas as pd

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model(df_dist, station_no):
    """Stores the data for the problem."""
    data = {}
    data['time_matrix'] = df_dist
    data['num_vehicles'] = 1
    data['depot'] = station_no
    return data


def print_solution(data, manager, routing, solution, total_stops):
    """Prints solution on console."""
    calculated_sequence = [0]*(total_stops)
    time_dimension = routing.GetDimensionOrDie('Time')
    for vehicle_id in range(data['num_vehicles']):
        i = 0
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            calculated_sequence[index] = i
            index = solution.Value(routing.NextVar(index))
            i += 1       
    return (calculated_sequence)

def ormain(df_dist, station_no, max_time, total_stops):
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
    data = create_data_model(df_dist, station_no)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        3600, # allow waiting time
        max_time,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 3
    search_parameters.log_search = False
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)

    # good combos
    # SIMULATED_ANNEALING, GLOBAL_CHEAPEST_ARC
    # GREEDY_DESCENT, GLOBAL_CHEAPEST_ARC
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        calculated_sequence = print_solution(data, manager, routing, solution, total_stops)
        return calculated_sequence
