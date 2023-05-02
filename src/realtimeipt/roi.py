#!/usr/bin/env python

from collections import deque
from contextlib import contextmanager
from enum import Enum
from functools import wraps
import logging
import re
import sys

import cv2
import numpy as np
import requests


HOST_PORT = "localhost:8000"  # FIXME use envvars
API = f"http://{HOST_PORT}"

VERTEX_RADIUS = 20


def raise_on_invalid(*invalids, exc=ValueError):
    def wrapper(func):
        @wraps(func)
        def wrapped(*a, **kw):
            ret = func(*a, **kw)
            if any(ret is invalid for invalid in invalids):
                raise exc(f"{func.__name__}(*{a}, **{kw})")
            return ret

        return wrapped

    return wrapper


class Colors(Enum):
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)


def sqdist(point, xy):
    x, y = xy
    return (point[0] - x) ** 2 + (point[1] - y) ** 2


def state_changer(method):

    from copy import deepcopy

    @wraps(method)
    def wrapped(self, *a, **kw):
        prev = deepcopy(self.state)  # TODO: maybe use copy.deepcopy?
        res = method(self, *a, **kw)
        if prev != self.state:
            self.state_stack.push(prev)
            self.draw()
            if self.on_state_changed:
                self.on_state_changed()
        return res

    return wrapped


class Stack:
    def __init__(self, maxlen):
        self._data = deque(maxlen=maxlen)

    def push(self, value):
        self._data.append(value)

    def pop(self):
        return self._data.pop()

    def peek(self):
        return self._data[-1]


FOCUSED = None

class EventManager:

    state_attrs = (
        "points",
        "projected_points",
        "projecting_points",
        #
        "vertex_selected",
        "projected_vertex_selected",
        "projecting_vertex_selected",
        #
        "roi_selected",
        "projected_roi_selected",
        "projecting_roi_selected",
        
    )

    def __init__(
        self,
        img,
        radius=VERTEX_RADIUS,
        vertices=None,
        roi_name=None,
        window_name=None,
    ):
        self.name = window_name or img
        self.base_img = cv2.imread(img).copy()
        self.img = self.base_img.copy()
        self.radius = radius
        self.r2 = self.radius ** 2
        self.roi_name = roi_name

        self.points = [] if vertices is None else vertices.tolist()
        self.projected_points = []
        self.projecting_points = []

        self.vertex_selected = None
        self.projected_vertex_selected = None
        self.projecting_vertex_selected = None

        self.roi_selected = False
        self.projected_roi_selected = False
        self.projecting_roi_selected = False

        self.pos = (-100, -100)

        self.state_stack = Stack(maxlen=100)

        self.mouse_triggers = {
            cv2.EVENT_LBUTTONDOWN: self._on_left_mouse_down,
            cv2.EVENT_MOUSEMOVE: self._on_mouse_move,
            cv2.EVENT_LBUTTONUP: self._on_left_mouse_up,
            # cv2.EVENT_MOUSEWHEEL: self._on_mousewheel,
            3: self._on_mousewheel_down,
            6: self._on_mousewheel_up,
            10: self._on_mousewheel_scroll,
        }

        self.keyboard_triggers = {
            ord("z"): self.undo,
            ord("c"): self.clear,
        }

        self.on_state_changed = None

        print(f"Listening for events on window `{self.name}`:")
        print(f"  * Drag ROI or a vertex to move it")  # todo automate this
        print(f"  * Click on a vertex to remove it")  # todo automate this
        print(f"  * Click outside the ROI add a vertex")  # todo automate this
        print(f"  * Press `z` to undo")  # todo automate this
        print(f"  * Press `c` to clear")  # todo automate this
        print(f"  * Press `<Enter>` to return")  # todo automate this

    @property
    def state(self):
        return {attr: getattr(self, attr) for attr in self.state_attrs}

    @state.setter
    def state(self, value):
        for attr, val in value.items():
            if attr not in self.state_attrs:
                raise AttributeError(f"`{attr}` not in {self.state_attrs}")
            setattr(self, attr, val)
        self.draw()

    def select_nearby_vertex(self, x, y, points="points"):
        for point in getattr(self, points):
            if sqdist(point, (x, y)) <= self.r2:
                return point

    def select_inside_roi(self, x, y, points="points"):
        pts = np.array(getattr(self, points)).reshape([-1, 1, 2]).astype(np.int64)
        ret = cv2.pointPolygonTest(contour=pts, pt=(x, y), measureDist=True)
        return ret >= 0

    def undo(self):
        current = self.state
        try:
            previous = self.state_stack.pop()
        except IndexError:
            logging.info("Cannot perform undo without history")
        else:
            self.state = previous
            self.draw()

    def clear(self):
        self.points = []
        self.projected_points = []
        self.projecting_points = []

        vertex_selected = None
        projected_vertex_selected = None
        projecting_vertex_selected = None

        roi_selected = None
        projected_roi_selected = None
        projecting_roi_selected = None

    def __grab_focus(self):
        global FOCUSED
        FOCUSED = self.name

    def __on_mouse_down(self, points="points"):
        vertex = self.select_nearby_vertex(*self.pos, points=points)
        prefix = points.rstrip("points")
        if vertex is not None:
            setattr(self, f"{prefix}vertex_selected", (self.pos, vertex))  # ()initial drag pos, coords)
        elif self.select_inside_roi(*self.pos, points=points):
            setattr(self, f"{prefix}roi_selected", self.pos)  # initial drag pos
        else:
            pass


    def _on_left_mouse_down(self, *a, **kw):
        self.__on_mouse_down("points")


    def _on_mousewheel_down(self, *a, **kw):
        self.__on_mouse_down("projecting_points")

    def __on_mouse_move(self, points="points"):
        prefix = points.rstrip("points")
        if getattr(self, f"{prefix}vertex_selected"):
            nu = self.pos
            idx = getattr(self,points).index(getattr(self, f"{prefix}vertex_selected")[1])
            getattr(self, points)[idx] = nu
            setattr(self, f"{prefix}vertex_selected", (getattr(self, f"{prefix}vertex_selected")[0], nu) )
        elif getattr(self, f"{prefix}roi_selected"):
            # print("    held roi moved")
            sel_roi = getattr(self, f"{prefix}roi_selected")
            dx, dy = (
                self.pos[0] - sel_roi[0],
                self.pos[1] - sel_roi[1],
            )
            pts = getattr(self,points)
            for idx in range(len(pts)):
                point = pts[idx]
                getattr(self, points)[idx] = point[0] + dx, point[1] + dy
            setattr(self, f"{prefix}roi_selected", self.pos)

    def _on_mouse_move(
        self,
        event,
        x,
        y,
        flags,
        *a,
        **kw
    ):
        for points in ("points", "projecting_points", "projected_points"):
            self.__on_mouse_move(points)


    def __on_mouse_up(self, points="points"):
        prefix = points.rstrip("points")
        vertex_selected = getattr(self, f"{prefix}vertex_selected")
        if vertex_selected:
            if sqdist(self.pos, vertex_selected[0]) <= self.r2:
                getattr(self, points).remove(vertex_selected[1])
            else:
                pass
            setattr(self, f"{prefix}vertex_selected", False)
        elif getattr(self, f"{prefix}roi_selected"):
            setattr(self, f"{prefix}roi_selected", False)
        else:
            getattr(self, f"{prefix}points").append(self.pos)

    def _on_left_mouse_up(
        self,
        event,
        x,
        y,
        flags,
        *a,
        **kw
    ):
        self.__on_mouse_up("points")



    def _on_mousewheel_up(
        self,
        event,
        x,
        y,
        flags,
        *a,
        **kw
    ):
        self.__on_mouse_up("projecting_points")

    def _on_mousewheel_scroll(
        self,
        *a,
        **kw
    ):
        print(f"_on_mousewheel_scroll: args={a}, kwargs={kw}")

    @state_changer
    def on_mouse(
        self,
        event,
        x,
        y,
        flags,
        *a,
        **kw
    ):
        self.__grab_focus()
        # print(f"Mouse {event} at ({x},{y})")
        self.pos = pos = (int(x), int(y))

        try:
            self.mouse_triggers[event](
                event,
                x,
                y,
                flags,
                *a,
                **kw
            )
        except KeyError as exc:
            raise NotImplementedError(f"Unhandled event {event}") from exc

        # self.draw(x,y)

    def on_keyboard(self, pressed_key):
        try:
            return self.keyboard_triggers[pressed_key]()
        except KeyError as exc:
            logging.info(f"Unhandled keypress: {pressed_key}")

    def _draw_cursor(self):
        if any(coord < 0 for coord in self.pos):
            return
        cv2.circle(
            img=self.img,
            center=self.pos,
            radius=self.radius,
            color=Colors.BLACK.value,
            thickness=1,
        )
        cv2.line(
            self.img,
            (self.pos[0] - 10, self.pos[1]),
            (self.pos[0] + 10, self.pos[1]),
            Colors.BLACK.value,
            1,
        )
        cv2.line(
            self.img,
            (self.pos[0], self.pos[1] - 10),
            (self.pos[0], self.pos[1] + 10),
            Colors.BLACK.value,
            1,
        )


    def __draw_roi(
        self,
        vals,
        points="points"
    ):
        prefix = points.rstrip("points")
        self.img = draw_roi(
            self.img,
            vertices=parse_vertices_other(vals),
            roi_name=self.roi_name if points == "points" else points,
            background_color=(0, 0, 255),
            vertex_color=None,
            vertex_radius=self.radius,
            selected=bool(getattr(self, f"{prefix}roi_selected")),
        )


    def draw(self):
        self.img = self.base_img.copy()

        self._draw_cursor()

        for points in ("points", "projected_points", "projecting_points"):
            vals = getattr(self, points)
            if vals:
                self.__draw_roi(vals, points)

        # if self.projected_points:
        #     self._draw_projected_roi()

        cv2.imshow(self.name, self.img)


@raise_on_invalid(None, exc=FileNotFoundError)
def read_image(image_path):
    return cv2.imread(
        image_path,
        # 0,
        cv2.IMREAD_UNCHANGED
        # cv2.IMREAD_COLOR,
    )


def cv2_windows(*names):
    try:
        for name in names:
            cv2.namedWindow(
                name,
                cv2.WINDOW_NORMAL,
                # cv2.WINDOW_FULLSCREEN | cv2.WINDOW_GUI_EXPANDED,
            )
        # cv2.setWindowProperty(name,cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while True:
            key = cv2.waitKey(1) & 0xFF
            yield key
    finally:
        cv2.destroyAllWindows()


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def edges_to_vertices(edges: str):
    pass


def cv2_draw_centered_text(
    text,
    center,
    image,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_size=1,
    font_weight=2,
    text_color=(255, 0, 0),
    background_color=None,
):

    textsize = cv2.getTextSize(text, font, font_size, font_weight)[0]

    x0 = center[0] - textsize[0] // 2
    y0 = center[1] + textsize[1] // 2

    if background_color:
        alpha = 0.9
        vertices = np.array(
            [
                (x0, y0),
                (x0 + textsize[0], y0),
                (x0 + textsize[0], y0 - textsize[1]),
                (x0, y0 - textsize[1]),
            ],
            np.int32,
        )
        cv2_fill_roi(
            image,
            vertices,
            background_color,
            alpha,
        )

    cv2.putText(
        image,
        text,
        (x0, y0),  # (center[0], center[1]), #
        font,
        font_size,
        (255, 0, 0),
        font_weight,
        cv2.LINE_AA,
    )


def cv2_fill_roi(
    source,
    vertices,
    background_color,
    alpha,
    gamma=0,
):
    overlay = source.copy()
    cv2.fillPoly(overlay, pts=[vertices], color=background_color)
    cv2.addWeighted(overlay, alpha, source, 1 - alpha, gamma, source)


def draw_roi(
    image,
    vertices,
    roi_name,
    edge_color=(0, 0, 0),
    background_color=None,
    selected=False,
    vertex_color=(0, 0, 0),
    vertex_radius=10,
    text_color=(0,0,0),
    text_bg_color=None,
):
    text = re.sub(r"^roi-(.*)$", r"\1", roi_name)
    board = image.copy()

    if background_color:
        cv2_fill_roi(
            board,
            vertices,
            background_color=background_color,
            alpha=0.3 if not selected else 0.5,
            gamma=0.5,
        )

    for idx, vertex in enumerate(vertices):
        cv2.circle(
            img=board,
            center=vertex,
            radius=vertex_radius,
            color=vertex_color
            if isinstance(vertex_color, tuple)
            else list(Colors)[idx % len(Colors)].value,
            thickness=-1,
        )
        cv2.circle(
            img=board,
            center=vertex,
            radius=int(vertex_radius*.8),
            color=(255,255,255),
            thickness=-1,
        )
        cv2_draw_centered_text(
            str(idx),
            vertex,
            board,
            text_color=text_color,
            background_color=text_bg_color,
        )

    pairs = vertices.reshape((-1, 1, 2))

    cv2.polylines(
        board,
        [pairs],
        True,
        # (0,255,255),
        edge_color,
    )
    roi_center = list(map(int, pairs.mean(axis=0).tolist()[0]))

    cv2_draw_centered_text(
        text,
        roi_center,
        board,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_size=1,
        font_weight=1,
        text_color=(255, 0, 0),
        background_color=(0, 255, 0),
    )

    return board


def parse_vertices_other(vertices_in):
    return np.array([list(map(int, coords)) for coords in vertices_in], np.int32)


def parse_vertices(vertices_str):
    if not vertices_str:
        return None
    return parse_vertices_other(batch(vertices_str.split(";"), n=2))


class Controller:
    def __init__(self, camera_manager, bev_manager):
        self.camera_manager = camera_manager
        self.bev_manager = bev_manager
        self.camera_manager.on_state_changed = self.on_manager_state_changed
        self.bev_manager.on_state_changed = self.on_manager_state_changed

        self.journal = []

        self.keyboard_triggers = {
            ord("q"): self.dump,
            13: self.dump,  # enter to save
        }

    def on_keyboard(self, pressed_key):
        try:
            return self.keyboard_triggers[pressed_key]()
        except KeyError as exc:
            return

    def dump(self):
        return self.journal

    def on_manager_state_changed(self):
        entry = self.update_matrix()
        try:
            mtx = entry['homography_matrix']
        except TypeError:
            if entry is None:
                return
            raise
        except KeyError:
            return
        self.update_projected(mtx)

    def update_matrix(self):
        cam_pts, bev_pts = self.camera_manager.points, self.bev_manager.points
        if not any((cam_pts, bev_pts)):
            return
        body = {"camera_points": cam_pts, "bev_points": bev_pts}
        homography_response = requests.post(
            f"{API}/homography/compute", json=body
        )
        if not homography_response.ok:
            try:
                logging.warning(homography_response.json()['detail'])
            except:
                logger.exception("UNHANDLED RESPONSE: {homography_response}")
            finally:
                return

        data = homography_response.json()

        homography_matrix = data["homography_matrix"]
        mask = data["mask"]

        projected = data["projected"]
        reprojection_error = data["reprojection_error"]
        
        entry = {
            "body": body,
            "homography_matrix": homography_matrix,
            "mask": mask,
            "reprojection_error": reprojection_error,
            "projected": projected,
        }

        self.journal.append(entry)
        print(entry)
        return entry

    def update_projected(self, homography_matrix):
        if not homography_matrix:
            return
        for manager_from, manager_to in zip(
            (self.camera_manager, self.bev_manager),
            (self.bev_manager, self.camera_manager),
        ):
            self.__update_projections(manager_from, manager_to, homography_matrix)
        # self.bev_manager.draw()
        
    def __update_projections(
        self,
        manager_from,
        manager_to,
        homography_matrix,
    ):
        if manager_from is not self.camera_manager:
            homography_matrix = np.linalg.inv(homography_matrix).tolist()
        body = {
            "points": manager_from.projecting_points,
            "homography_matrix": homography_matrix,
        }
        transform_response = requests.post(
            f"{API}/homography/transform", json=body
        )
        if not transform_response.ok:
            breakpoint()
        projected_points = transform_response.json()

        manager_to.projected_points = projected_points


def main(
    camera_path="resources/img/cam/salida.png",
    bev_path="resources/img/drone/stitch.JPG",
    camera_region_name="CAMERA HOMOGRAPHY REGION (CHR)",
    camera_window_name = "Camera View",
    bev_region_name="BIRD-EYE HOMOGRAPHY REGION (BHR)",
    bev_window_name = "Birdeye View",
):

    managers = {
        camera_window_name: EventManager(
            camera_path,
            vertices=None,
            roi_name=camera_region_name,
            window_name=camera_window_name,
        ),
        bev_window_name: EventManager(
            bev_path,
            vertices=None,
            roi_name=bev_region_name,
            window_name=bev_window_name,
        ),
    }


    cv2.imshow(camera_window_name, read_image(camera_path))
    cv2.imshow(bev_window_name, read_image(bev_path))
    cbs_set = False


    controller = Controller(
        managers[camera_window_name],
        managers[bev_window_name],
    )

    for pressed_key in cv2_windows(
        camera_window_name,
        bev_window_name
    ):
        if not cbs_set:
            for _, manager in managers.items():
                cv2.setMouseCallback(manager.name, manager.on_mouse)
                manager.draw()

        if pressed_key == 255:
            continue

        if FOCUSED:
            manager = managers[FOCUSED]
            manager.on_keyboard(pressed_key)

        dump = controller.on_keyboard(pressed_key)
        if dump:
            return dump
        # cv2.imshow(window_name, image_with_roi)
        # image_with_roi = draw_roi(
        #     image,
        #     vertices,
        #     roi_name,
        #     background_color=(0,0,255),
        # )


if __name__ == "__main__":
    points = main(*sys.argv[1:])
    print(f"points: {points}")
