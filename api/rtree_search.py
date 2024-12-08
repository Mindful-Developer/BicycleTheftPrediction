import json
from shapely.geometry import Point, Polygon
from pyproj import Transformer
from rtree import index
from h158_to_h140 import converter


def convert_to_140(h158: int) -> (str, str):
    """Convert a H158 code to a N140 code."""
    h140 = converter.get(h158, None)
    if h140:
        return h140


class RTree:
    """Spatial index using R-tree."""

    def __init__(self):
        with open("toronto_map_data_extracted.json") as f:
            data: dict[str, list[dict[str, str | list[list[float]]]]] = json.load(f)

        self.index = index.Index()
        self.polygons = {}

        for region in data['regions']:
            n140, h140, geometry = region['NEIGHBOURHOOD_140'], int(region['HOOD_140']), region['geometry']
            polygon = Polygon(geometry)
            self.polygons[h140] = {"polygon": polygon, "n140": n140}
            self.index.insert(h140, polygon.bounds)

    def search(self, lat: float, lon: float) -> tuple[str, str]:
        """Search for the H140 and N140 codes for a given latitude and longitude."""
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)
        point = Point(x, y)

        candidates = list(self.index.intersection((x, y, x, y)))

        for h140 in candidates:
            str_h140 = str(h140).zfill(3)
            geofence = self.polygons[h140]
            polygon = geofence["polygon"]

            if polygon.contains(point):
                res = convert_to_140(h140)
                if res:
                    return res
                return str_h140, geofence["n140"]

        return "NSA", "NSA"


if __name__ == '__main__':
    rtree = RTree()
    lat = 43.693449
    lon = -79.433288
    h140, n140 = rtree.search(lat, lon)
    print(f"Latitude: {lat}, Longitude: {lon}")
    print(f"Neighbourhood 140: {n140}, Hood 140: {h140}")


