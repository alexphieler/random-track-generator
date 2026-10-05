import gpxpy
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from shapely.geometry import Point, Polygon

PACSIM_TIME_KEEPING_OFFSET = 1.5  # [m]

class Mode(Enum):
    """ 
    Possible modes for how Voronoi regions are selected.
    
    1. Expand:
        Find closest nodes around starting node.
        Results in roundish track shapes.
    
    2. Extend:
        Find nodes closest to line extending from starting node.
        Results in elongated track shapes.
        
    3. Random:
        Select all regions randomly.
        Results in large track shapes.

    4. Hairpin:
        Create a long out-and-back layout with tight U-turns.

    5. Slalom:
        Create an out-and-back layout with alternating bends.

    6. Technical:
        Create a compact circuit with frequent direction changes.
    """
    EXPAND = 1
    EXTEND = 2
    RANDOM = 3
    HAIRPIN = 4
    SLALOM = 5
    TECHNICAL = 6

class SimType(Enum):
    """ Selection between output format for different simulators.

    1. FSSIM:
        Output FSSIM compatible .yaml file.
    2. FSDS:
        Output FSDS compatible .csv file 
    3. PACSIM:
        Output PACSim compatible .yaml file.
    """
    FSSIM = 1
    FSDS = 2
    GPX = 3
    PACSIM = 4

class Preset(Enum):
    """
    Preset track input parameters.

    1. Small:
        Small track.
    2. Medium:
        Mid-sized track.
    3. Large:
        Large track.
    """
    SMALL = 1
    MEDIUM = 2
    LARGE = 3    


def _cross_section_y(cones: np.ndarray, x: float) -> float:
    """Return the boundary intersection nearest the start-line centre."""
    intersections = []
    for start, end in zip(cones, np.roll(cones, -1, axis=0)):
        if (start[0] - x) * (end[0] - x) > 0 or start[0] == end[0]:
            continue
        fraction = (x - start[0]) / (end[0] - start[0])
        intersections.append(start[1] + fraction * (end[1] - start[1]))

    if not intersections:
        raise ValueError(f"Unable to find a track boundary at x={x} m.")
    return float(min(intersections, key=abs))


@dataclass
class Track:
    """ Track dataclass

    Attributes:
         cones_left (np.ndarray): Left cones of track.
         cones_right (np.ndarray): Right cones of track.
    """
    cones_left: np.ndarray
    cones_right: np.ndarray

    def as_tuple(self):
        """ Returns cones as tuple of left and right cones.
        """
        return self.cones_left, self.cones_right

    def save(self, location: str | Path, sim_type: SimType | str, *,
             lat_offset: float = 0.0, lon_offset: float = 0.0, z_offset: float = 0.0,
             filename: str | None = None):
        """ Saves track in specified format for use in different simulators.

        Args:
            location: Location to save track to.
            sim_type: Format to save track in. Must be either "fssim", "fsds",
                "gpx" or "pacsim".
            lat_offset: Latitude offset for GPX output format, in degrees.
            lon_offset: Longitude offset for GPX output format, in degrees.
            z_offset: Altitude offset for GPX output format, in meters.
            filename: Optional output filename. Uses the format's default name
                when omitted.
        """
        sim_type = SimType[sim_type.upper()] if isinstance(sim_type, str) else SimType(sim_type)
        path = Path(location)

        if sim_type == SimType.FSSIM:
            with open(path / (filename or "random_track.yaml"), 'w') as f:
                yaml.dump({
                    'cones_left': self.cones_left.tolist(),
                    'cones_right': self.cones_right.tolist(),
                    'cones_orange': [],
                    'cones_orange_big': [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]],
                    'starting_pose_cg': [0., 0., 0.],
                    'tk_device': [[6., 3.], [6., -3.]],
                }, f)

        elif sim_type == SimType.FSDS:
            out = path / (filename or "random_track.csv")
            with open(out, 'w') as f:
                for cone in self.cones_left:
                    f.write(f"blue,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                for cone in self.cones_right:
                    f.write(f"yellow,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                f.write("big_orange,4.7,2.2,0,0.01,0.01,0\n")
                f.write("big_orange,4.7,-2.2,0,0.01,0.01,0\n")
                f.write("big_orange,7.3,2.2,0,0.01,0.01,0\n")
                f.write("big_orange,7.3,-2.2,0,0.01,0.01,0\n")

        elif sim_type == SimType.GPX:
            gpx = gpxpy.gpx.GPX()
            gpx.tracks.append(gpxpy.gpx.GPXTrack())

            deg_per_m_lat = np.degrees(1 / 6378100)
            deg_per_m_lon = np.degrees(1 / 6378100) / np.cos(np.radians(lat_offset))

            for cone in np.vstack([self.cones_left, self.cones_right]):
                lat = lat_offset + cone[1] * deg_per_m_lat
                lon = lon_offset + cone[0] * deg_per_m_lon
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=z_offset))

            (path / (filename or "random_track.gpx")).write_text(gpx.to_xml())

        elif sim_type == SimType.PACSIM:
            def _cones(cones: np.ndarray, cone_class: str) -> list[dict]:
                return [
                    {
                        "position": [float(cone[0]), float(cone[1]), 0.0],
                        "class": cone_class,
                    }
                    for cone in cones
                ]

            pacsim_track = {
                "track": {
                    "version": 1.0,
                    "lanesFirstWithLastConnected": True,
                    "start": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                    "earthToTrack": {
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [0.0, 0.0, 0.0],
                    },
                    "left": _cones(self.cones_left, "blue"),
                    "right": _cones(self.cones_right, "yellow"),
                    "time_keeping": self.pacsim_time_keeping(),
                    "unknown": [],
                }
            }
            with open(path / (filename or "random_track.yaml"), "w") as f:
                yaml.safe_dump(pacsim_track, f, sort_keys=False)

    def pacsim_time_keeping(self) -> list[dict]:
        """Create a timing gate roughly 9 m ahead of the start position."""
        desired_distance = 9.0
        min_x = max(self.cones_left[:, 0].min(), self.cones_right[:, 0].min())
        max_x = min(self.cones_left[:, 0].max(), self.cones_right[:, 0].max())
        if min_x >= max_x:
            return []

        edge_margin = min(0.01, (max_x - min_x) / 4)
        candidate_x = np.clip(desired_distance, min_x + edge_margin, max_x - edge_margin)
        candidates = sorted(
            np.linspace(min_x + edge_margin, max_x - edge_margin, 101),
            key=lambda x: abs(x - candidate_x),
        )
        for distance_ahead in candidates:
            try:
                left_y = _cross_section_y(self.cones_left, distance_ahead)
                right_y = _cross_section_y(self.cones_right, distance_ahead)
                break
            except ValueError:
                continue
        else:
            return []

        outer_boundary, inner_boundary = sorted(
            (Polygon(self.cones_left), Polygon(self.cones_right)),
            key=lambda boundary: boundary.area,
            reverse=True,
        )
        drivable_area = outer_boundary.difference(inner_boundary)

        def _time_keeping_position(y: float) -> list[float]:
            direction = 1 if y >= 0 else -1
            offset_y = y + direction * PACSIM_TIME_KEEPING_OFFSET
            # Keep the point on its boundary when moving it outward would
            # enter another nearby section of the track.
            if drivable_area.covers(Point(distance_ahead, offset_y)):
                offset_y = y
            return [float(distance_ahead), float(offset_y), 0.0]

        return [
            {
                "position": _time_keeping_position(left_y),
                "class": "timekeeping",
            },
            {
                "position": _time_keeping_position(right_y),
                "class": "timekeeping",
            },
        ]
