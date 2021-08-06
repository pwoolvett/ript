
import os
from pathlib import Path
from typing import Union


SHARED = Path(os.environ["COMMON_PATH_CONTAINER"]).resolve()
SHARED_HOST = Path(os.environ["COMMON_PATH_HOST"])
IMAGES_PATH = SHARED / "img"
IMAGES_PATH_HOST = SHARED_HOST / "img"

CAMERAS_CONFIG_PATH = f"{os.environ['COMMON_PATH_CONTAINER']}/config/cameras.json"
NVDS_ANALYTICS_CONFIG_PATH = os.environ["ANALYTICS_CONFIGURATION_FILE_CONTAINER"]

CAMERAS_JSON = SHARED / "config/cameras.json"


def host_image_to_relative(host_path:Union[str, Path])->str:
    """Convert host image path to relative path.
    
    Args:
        host_path: Local image path.

    Returns:
        Path: Relative image path.
    
    Usage::
    
    >>> os.environ['COMMON_PATH_HOST']="./common"
    >>> os.environ['COMMON_PATH_CONTAINER']="/shared/common"
    >>> host_path = "common/img/cam/estacionamiento.jpg"
    >>> host_image_to_relative(host_path)
    'cam/estacionamiento.jpg'
    """
    return str((
        IMAGES_PATH / Path(host_path).relative_to(IMAGES_PATH_HOST)
    ).resolve().relative_to(IMAGES_PATH))


def relative_image_to_host(relative_path: Union[str,Path])->str:
    """Convert relative image path to absolute host path.
    
    Args:
        relative_path: Local image path.

    Returns:
        Path: Absolute image path on host machine.
    
    Usage::
    
    >>> os.environ['COMMON_PATH_HOST']="./common"
    >>> os.environ['COMMON_PATH_CONTAINER']="/shared/common"
    >>> relative_path = "cam/estacionamiento.jpg"
    >>> relative_image_to_host(relative_path)
    'common/img/cam/estacionamiento.jpg'
    """
    absolute = IMAGES_PATH / relative_path
    return str(absolute.relative_to(SHARED.parent))

def relative_image_to_container(relative_path:Union[str,Path])->str:
    """Convert relative image path to absolute container path.
    
    Args:
        relative_path: Local image path.

    Returns:
        Path: Absolute image path on container.
    
    Usage::
    
    >>> os.environ['COMMON_PATH_HOST']="./common"
    >>> os.environ['COMMON_PATH_CONTAINER']="/shared/common"
    >>> relative_path = "cam/estacionamiento.jpg"
    >>> relative_image_to_container(relative_path)
    '/shared/common/img/cam/estacionamiento.jpg'
    """
    return str(IMAGES_PATH / relative_path)


def container_image_to_relative(container_path: Union[str,Path])->str:
    """Convert container image path to relative path.
    
    Args:
        container_path: Container image path.

    Returns:
        Path: Relative image path.
    
    Usage::

    >>> os.environ['COMMON_PATH_HOST']="./common"
    >>> os.environ['COMMON_PATH_CONTAINER']="/shared/common"
    >>> container_path = "/shared/common/img/cam/estacionamiento.jpg"
    >>> container_image_to_relative(container_path)
    'cam/estacionamiento.jpg'
    """
    return str(Path(container_path).relative_to(IMAGES_PATH))

def host_image_to_container(host_path: Union[str, Path]):
    # TODOC
    return relative_image_to_container(host_image_to_relative(host_path))

def container_image_to_host(container_path: Union[str, Path]):
    # TODOC
    return relative_image_to_host(container_image_to_relative(container_path))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
