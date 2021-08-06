from typing import Type
from typing import List

import numpy as np
from pydantic import BaseModel

from pydantic import root_validator


Point = List[int]

PointList = List[Point]


class CameraConfiguration(BaseModel):
    enable: str

    class Config:
        extra = "allow"


class RegionOfInterest(BaseModel):
    name: str
    vertices: PointList


class PointListPair(BaseModel):
    camera_points: PointList
    bev_points: PointList

    @root_validator
    def check_points_length_match(cls, values):
        camera_points, bev_points = values.get("camera_points"), values.get(
            "bev_points"
        )
        if len(camera_points) != len(bev_points):
            raise ValueError("Points list lengths differ")
        return values


FloatArray = List[float]
HomographyMatrix = List[FloatArray]
# TODO add valiation to ensure this can actually be a numpy matrix


class HomographyMatrixResult(BaseModel):
    # TODO mange serialization: np -> list
    # TODO mange de-serialization list -> np
    homography_matrix: HomographyMatrix  # HomographyMatrix
    mask: FloatArray


##############
from fastapi import FastAPI

models_app = FastAPI()


@models_app.get("/")
def hello():
    return {"hello": "world"}


@models_app.post("/")
def show(point: Point):
    print(f"point: {point}, type={type(point)}", flush=True)
    return point
