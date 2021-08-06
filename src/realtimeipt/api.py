import json
from configparser import ConfigParser
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException

from ground.base import get_context as load_geometries
from bentley_ottmann.planar import contour_self_intersects

import realtimeipt

from realtimeipt.models import Point
from realtimeipt.models import PointList
from realtimeipt.models import HomographyMatrixResult
from realtimeipt.models import HomographyMatrix
from realtimeipt.models import RegionOfInterest
from realtimeipt.models import CameraConfiguration
from realtimeipt.paths import SHARED
from realtimeipt.paths import IMAGES_PATH
from realtimeipt.paths import CAMERAS_CONFIG_PATH
from realtimeipt.paths import NVDS_ANALYTICS_CONFIG_PATH

app = FastAPI()



@app.post("/homography/compute")
def compute_homography_matrix(
    camera_points: PointList, bev_points: PointList
) -> HomographyMatrixResult:
    """Compute homography matrix from a given set of corresponding points.

    Args:
        camera_points (PointList): List of marker's locations on camera's image plane.
        bev_points (PointList): List of the corresponding camera marker's locations on the bird's eye view plane.

    Returns:
        Computed homography matrix, that relates both camera planes.
    """
    lcp = len(camera_points)
    lbp = len(bev_points)
    if lcp != lbp:
        raise HTTPException(
            status_code=400,
            detail=f"Length of coordinates must be the same (got {lcp} and {lbp})"
        )
    if lcp < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Length of coordinates pairs must be bigger than 3 (got ){lcp}"
        )

    pairs = np.array(camera_points), np.array(bev_points)
    homography_matrix_values, outliers_mask = cv2.findHomography(*pairs)

    projected = transform_points(camera_points, homography_matrix_values)
    reprojection_error = np.linalg.norm(pairs[1] - np.array(projected))
    return {
        "homography_matrix": homography_matrix_values.tolist(),
        "mask": outliers_mask.tolist(),
        "projected": projected,
        "reprojection_error": reprojection_error,
    }


@app.post("/homography/transform")
def transform_points(
    points: PointList, homography_matrix: HomographyMatrix
) -> PointList:
    """Convert list of points from camera plane into real world coordinates.

    Args:
        points (PointList): List of points in camera plane.
        homography_matrix (HomographyMatrix): Camera's homography matrix.

    Returns:
        List of real world coordinates.
    """
    transformed_points = []
    for point in points:
        extended_point = np.array([*point, 1])
        projected_scaled_x, projected_scaled_y, scale_factor = np.matmul(
            np.array(homography_matrix), np.array(extended_point)
        )
        projected_x, proyected_y = (
            projected_scaled_x / scale_factor,
            projected_scaled_y / scale_factor,
        )
        transformed_points.append((projected_x, proyected_y))
    return transformed_points


@app.post("/roi/validate")
def validate_roi(roi_vertices: PointList):
    """Checks wether a list of consecutive vertices form a valid simple polygon.

    A valid simple polygon is defined as a non-self intersecting closed polygonal chain.

    Args:
        roi_vertices (list): List of consecutive vertices of the polygon.

    Returns:
        bool: Indicating wether the polygon is valid.
    """
    geometries = load_geometries()
    Point = geometries.point_cls
    Polygon = geometries.contour_cls
    roi_candidate = Polygon([Point(x, y) for x, y in roi_vertices])
    return {"is_valid": not contour_self_intersects(roi_candidate)}


@app.get("/config/cameras/{camera_id}")
def load_camera_config(camera_id: str):
    try:
        with open(CAMERAS_CONFIG_PATH, "r") as config_file:
            config_dict = json.load(config_file)[camera_id]
    except (FileNotFoundError, KeyError, json.decoder.JSONDecodeError) as exc:
        print(f"Error opening cameras configuration at {CAMERAS_CONFIG_PATH}: {exc}")
        config_dict = {}
    return config_dict


@app.put("/config/cameras/{camera_id}")
def dump_camera_config(
    camera_id: str,
    config_dict: dict
):
    # Load existing configuration
    try:
        with open(CAMERAS_CONFIG_PATH, "r") as config_file:
            global_config_dict = json.load(config_file)
    except (FileNotFoundError, KeyError) as exc:
        global_config_dict = {}
    except json.decoder.JSONDecodeError as exc:
        shutil.copy2(
            CAMERAS_CONFIG_PATH, Path(CAMERAS_CONFIG_PATH).with_suffix(".corrupt")
        )
        global_config_dict = {}
    # Update only the current camera
    global_config_dict[camera_id] = config_dict
    with open(CAMERAS_CONFIG_PATH, "w") as config_file:
        json.dump(global_config_dict, config_file)


@app.get("/config/analytics/{camera_id}")
def load_analytics_config(camera_id: str):
    camera_key = f"roi-filtering-stream-{camera_id}"
    config_parser = ConfigParser()
    try:
        config_parser.read(NVDS_ANALYTICS_CONFIG_PATH)
        return dict(config_parser[camera_key])
    except Exception as exc:
        return {}


@app.put("/config/analytics/{camera_id}")
def dump_analytics_config(camera_id: str, camera: CameraConfiguration):
    camera_key = f"roi-filtering-stream-{camera_id}"

    camera_config = camera.dict()
    roi_name = [key for key in camera_config.keys() if key.startswith("roi-")][0]
    camera_config[roi_name] = parse_roi(
        camera_config[roi_name]
    )  # Parse ROI into NvDsAnalytics format

    config_parser = ConfigParser()
    try:
        config_parser.read(NVDS_ANALYTICS_CONFIG_PATH)
    except Exception as exc:
        pass
    config_parser[camera_key] = camera_config
    with open(NVDS_ANALYTICS_CONFIG_PATH, "w") as modified_config_file:
        config_parser.write(modified_config_file)


@app.post("/config/analytics/create")
def create_analytics_config(roi: RegionOfInterest):
    return {
        "enable": "1",
        f"roi-{roi.name}": roi.vertices,
        "inverse-roi": "0",
        "class-id": "-1",
    }


def parse_roi(roi_vertices: list) -> str:
    if isinstance(roi_vertices, str):
        return roi_vertices
    return ";".join(";".join(map(str, x)) for x in roi_vertices)


SERVE = "uvicorn --env-file .env realtimeipt.api:app"

def serve():
    import subprocess as sp
    import shlex
    return sp.run(shlex.split(SERVE))

if __name__ == "__main__":
    serve()
