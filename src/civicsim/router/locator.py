from shapely.geometry import LineString, Point


def convert_route_to_linestring(G, r):
    r_points = [find_latlng_of_node(G, _r) for _r in r]
    r_linestring = LineString(r_points)
    return r_linestring


def find_latlng_of_node(G, n):
    a = G.nodes[n]
    p = Point(a["x"], a["y"])
    return p


def find_current_location_in_route():
    pass


def find_current_location_in_linestring(ls: LineString, total_time: float, elapsed_time: float) -> Point:
    """
    Estimates the current location on a LineString based on the total travel time and the elapsed time.

    Args:
        ls (LineString): The LineString representing the route.
        total_time (float): The total travel time across the route, in seconds.
        elapsed_time (float): The elapsed time since the start of the journey, in seconds.

    Returns:
        Tuple[float, float]: A tuple containing the (x, y) coordinates of the estimated current location on the LineString.
    """
    if elapsed_time >= total_time:
        return ls.coords[-1]
    elif elapsed_time <= 0:
        return ls.coords[0]
    else:
        fraction = elapsed_time / total_time
        point = ls.interpolate(fraction, normalized=True)
        return Point(point.x, point.y)
