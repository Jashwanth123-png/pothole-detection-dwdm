"""
utils/helpers.py
================
Shared helper utilities used across all preprocessing scripts.

Provides:
- Config loading (config.yaml)
- Path resolution relative to the project root
- MD5 hashing for duplicate detection
- Logging setup
- JSON report saving
- Annotation XML parsing

Beginner note:
  Every function here is imported by the scripts in scripts/.
  Think of this file as a "toolbox" of common operations.
"""

import os
import json
import hashlib
import logging
import yaml
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# 1.  PROJECT ROOT
# ──────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    The project root is the parent folder of the 'utils' package,
    i.e., the directory that contains config.yaml.

    Returns
    -------
    Path
        Absolute Path object pointing at the project root.
    """
    # __file__ is this very file (utils/helpers.py)
    # .parent       -> utils/
    # .parent.parent -> project root
    return Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────
# 2.  CONFIG LOADING
# ──────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """
    Load and return the project configuration from config.yaml.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config.yaml. Defaults to <project_root>/config.yaml.

    Returns
    -------
    dict
        Parsed YAML configuration as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If config.yaml cannot be found.
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure config.yaml is present in the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ──────────────────────────────────────────────────────────────
# 3.  PATH RESOLUTION
# ──────────────────────────────────────────────────────────────

def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path that is relative to the project root.

    Parameters
    ----------
    relative_path : str
        A path string such as "data/processed/d40".

    Returns
    -------
    Path
        Absolute Path object.

    Example
    -------
    >>> resolve_path("data/outputs")
    PosixPath('/home/user/project_pathhole/data/outputs')
    """
    return get_project_root() / relative_path


def ensure_dir(path) -> Path:
    """
    Create a directory (and all parents) if it does not already exist.

    Parameters
    ----------
    path : str or Path
        The directory path to create.

    Returns
    -------
    Path
        The same path as an absolute Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────
# 4.  MD5 HASHING  (for duplicate detection)
# ──────────────────────────────────────────────────────────────

def compute_md5(file_path) -> str:
    """
    Compute the MD5 hash of a file's contents.

    Two files with the same MD5 hash are considered duplicates.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to hash.

    Returns
    -------
    str
        Hexadecimal MD5 digest string (32 characters), or empty string
        if the file cannot be read.
    """
    hasher = hashlib.md5()
    try:
        # Read the file in 64 KB chunks to handle large images without
        # loading the entire file into memory at once.
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, OSError) as e:
        logging.warning(f"Could not hash file {file_path}: {e}")
        return ""


# ──────────────────────────────────────────────────────────────
# 5.  IMAGE VALIDATION
# ──────────────────────────────────────────────────────────────

def is_valid_image(file_path) -> tuple:
    """
    Check whether a file is a valid, non-corrupt image.

    Uses Pillow to attempt to open and verify the file.

    Parameters
    ----------
    file_path : str or Path
        Path to the image file.

    Returns
    -------
    tuple[bool, str]
        (True, "OK") if the image is valid.
        (False, reason) if invalid, where reason is a short description.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        with Image.open(file_path) as img:
            img.verify()          # Checks for corruption without full decode
        # verify() closes the file; re-open to confirm it can actually be read
        with Image.open(file_path) as img:
            img.load()            # Forces full pixel data load
        return True, "OK"
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────────────────────
# 6.  LOGGING SETUP
# ──────────────────────────────────────────────────────────────

def setup_logging(log_name: str = "pothole", level=logging.INFO) -> logging.Logger:
    """
    Configure and return a named logger with a consistent format.

    Outputs log messages to the console (stdout).

    Parameters
    ----------
    log_name : str
        Name for the logger (appears in log output).
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Example
    -------
    >>> logger = setup_logging("clean_dataset")
    >>> logger.info("Starting cleaning...")
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ──────────────────────────────────────────────────────────────
# 7.  JSON REPORT SAVING
# ──────────────────────────────────────────────────────────────

def save_report(report: dict, output_path) -> None:
    """
    Serialise a Python dictionary to a pretty-printed JSON file.

    Creates parent directories automatically.

    Parameters
    ----------
    report : dict
        Data to save. Must contain JSON-serialisable values.
    output_path : str or Path
        Destination file path (e.g., "data/outputs/cleaning_report.json").
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)  # default=str handles Path/datetime

    logging.getLogger("pothole").info(f"Report saved → {output_path}")


# ──────────────────────────────────────────────────────────────
# 8.  XML ANNOTATION PARSING  (Pascal VOC format)
# ──────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path) -> dict:
    """
    Parse a Pascal VOC XML annotation file.

    The RDD2022 dataset uses Pascal VOC format:
      <annotation>
        <filename>image.jpg</filename>
        <size><width>…</width><height>…</height></size>
        <object>
          <name>D40</name>
          <bndbox><xmin>…</xmin><ymin>…</ymin>
                  <xmax>…</xmax><ymax>…</ymax></bndbox>
        </object>
      </annotation>

    Parameters
    ----------
    xml_path : str or Path
        Path to the .xml annotation file.

    Returns
    -------
    dict with keys:
        "filename"  : str   — image filename declared inside the XML
        "width"     : int   — image width in pixels
        "height"    : int   — image height in pixels
        "objects"   : list  — list of dicts, each with:
                              {"name": str, "xmin": int, "ymin": int,
                               "xmax": int, "ymax": int}

    Returns an empty dict on parse error.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = ""
        fn_elem = root.find("filename")
        if fn_elem is not None:
            filename = fn_elem.text.strip()

        width, height = 0, 0
        size_elem = root.find("size")
        if size_elem is not None:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            if w_elem is not None:
                width = int(w_elem.text.strip())
            if h_elem is not None:
                height = int(h_elem.text.strip())

        objects = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_elem is None or bndbox is None:
                continue
            try:
                objects.append({
                    "name": name_elem.text.strip(),
                    "xmin": int(float(bndbox.find("xmin").text.strip())),
                    "ymin": int(float(bndbox.find("ymin").text.strip())),
                    "xmax": int(float(bndbox.find("xmax").text.strip())),
                    "ymax": int(float(bndbox.find("ymax").text.strip())),
                })
            except (AttributeError, ValueError):
                continue  # Skip malformed bounding boxes

        return {
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        }

    except Exception as e:
        logging.getLogger("pothole").warning(f"Failed to parse XML {xml_path}: {e}")
        return {}


# ──────────────────────────────────────────────────────────────
# 9.  SUPPORTED IMAGE EXTENSIONS
# ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images(directory) -> list:
    """
    Return a sorted list of image file Paths in a directory (non-recursive).

    Parameters
    ----------
    directory : str or Path
        Directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of image paths. Empty list if directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_images_recursive(directory) -> list:
    """
    Return a sorted list of image file Paths in a directory tree (recursive).

    Parameters
    ----------
    directory : str or Path
        Root directory to scan recursively.

    Returns
    -------
    list[Path]
        Sorted list of image paths. Empty list if directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# Backward-compatible alias used throughout the project.
def setup_logger(log_name: str = "pothole", level=logging.INFO) -> logging.Logger:
    return setup_logging(log_name, level)
