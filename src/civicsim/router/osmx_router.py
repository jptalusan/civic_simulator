from src.civicsim.router.base_router import BaseRouter
import networkx as nx
import osmnx as ox


class OSMX_Router(BaseRouter):
    def __init__(self) -> None:
        super().__init__()
        self.G = None

    def set_network(self, G):
        self.G = G

    def get_route(self, o, d, weight="length"):
        if self.G is None:
            raise "Assign a network target."

        o_lon, o_lat = o.x, o.y
        d_lon, d_lat = d.x, d.y

        h_node = ox.nearest_nodes(self.G, o_lon, o_lat)
        w_node = ox.nearest_nodes(self.G, d_lon, d_lat)

        r = nx.shortest_path(self.G, h_node, w_node, weight=weight)
        return r

    def get_total_dist_and_time(self, o, d, weight="length"):
        try:
            r = self.get_route(o, d, weight)
            gdf_dist = ox.routing.route_to_gdf(self.G, r, weight="length")
            gdf_tt = ox.routing.route_to_gdf(self.G, r, weight="travel_time")
            dist_m = gdf_dist["length"].sum()
            tt_s = gdf_tt["travel_time"].sum()
            return dist_m, tt_s
        except:
            return -1, -1
