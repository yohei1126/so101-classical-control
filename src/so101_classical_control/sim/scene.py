"""Access to the MuJoCo scene used for simulation."""

from pathlib import Path


def scene_xml_path() -> Path:
    """Return the path to the sponge-to-paper MuJoCo scene."""
    return Path(__file__).resolve().parent / "assets" / "sponge_paper_scene.xml"
