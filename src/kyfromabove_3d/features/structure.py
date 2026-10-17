"""A building's structural class from its roof (flat plate, planar, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Set, Union

from .surface import SurfaceZones

# --------------------------------------------------- structural classification

@dataclass
class BuildingClassification:
    """A building's structural type, as a label for the viewer and the records.

    The label does not steer the mesh: every building is built from its own
    returns. A whole-building type is too coarse for a downtown (one building
    has a flat roof, a curved corner and a party wall at once), and where the
    LiDAR is dense the height field and the wall profiles already say what is
    there. Choosing a model by type belongs where the returns are too thin to
    show the shape -- and then per roof region or face group, checked on a
    before/after sheet.

    Types:
    - ``simple_box``: one level, flat roof, planar walls.
    - ``tiered_box``: several levels, flat roofs, planar walls.
    - ``complex_roof``: planar walls under a pitched, domed or faceted roof.
    - ``sculpted_facade``: curved walls, colonnades, or walls stepping in and out.
    """

    archetype: str
    roof_type: str
    facade_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"archetype": self.archetype, "roof_type": self.roof_type,
                "facade_type": self.facade_type, "metrics": self.metrics}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildingClassification":
        return cls(archetype=d["archetype"], roof_type=d.get("roof_type", "unknown"),
                   facade_type=d.get("facade_type", "unknown"), metrics=d.get("metrics", {}))


#: A roof is flat when the returns sit within this RMS of one plate (m) ...
FLAT_PLATE_RMS_M = 0.35

#: ... most roof returns are on that plate (the rest is parapets and plant) ...
FLAT_PLATEAU_FRAC = 0.50

#: ... and under this share of roof cells slope like a pitched roof.
FLAT_MAX_SLOPED_FRAC = 0.05

#: A street wall stepping in or out by more than this is not planar (m).
PLANAR_MAX_OFFSET_M = 0.45


def classify_building_structure(
    levels: Sequence[Dict[str, Any]],
    zones: Optional[SurfaceZones],
    profiles: Optional[Dict[int, Sequence[Any]]] = None,
    arc_edges: Optional[Union[Set[int], Sequence[int]]] = None,
    *,
    sloped_fraction: float,
    party_faces: Optional[Union[Set[int], Sequence[int]]] = None,
) -> BuildingClassification:
    """Label a building's structure from what the colourise and the roof field measured.

    ``levels`` from :func:`building_levels`; ``zones`` from
    :func:`split_surface_zones`; ``profiles`` maps a face to its bands;
    ``arc_edges`` are the faces on curved walls; ``sloped_fraction`` is the
    share of roof cells that slope like a pitched roof (see
    ``roof_sloped_fraction`` in build_mesh); ``party_faces`` are the faces
    against a neighbour, whose returns are the neighbour's and are ignored.
    """
    profiles = profiles or {}
    party = set(party_faces or [])
    has_arcs = bool(arc_edges)
    street = [b for f, bands in profiles.items() if f not in party for b in bands]
    max_wall_offset = max((abs(float(b.offset_m)) for b in street
                           if b.status in ("observed", "bridged", "extended") and b.offset_m is not None),
                          default=0.0)
    has_open_bands = any(b.status == "open" for b in street)

    if zones is not None:
        plate_rms, roof_frac = zones.plate_rms_m, zones.roof_plateau_frac
        is_flat_roof = (plate_rms is not None and plate_rms <= FLAT_PLATE_RMS_M
                        and roof_frac >= FLAT_PLATEAU_FRAC and sloped_fraction < FLAT_MAX_SLOPED_FRAC)
    else:
        plate_rms = roof_frac = None
        is_flat_roof = False
    is_planar_facade = not has_arcs and not has_open_bands and max_wall_offset < PLANAR_MAX_OFFSET_M

    num_levels = max(len(levels), 1)
    if is_flat_roof and is_planar_facade:
        archetype = "simple_box" if num_levels == 1 else "tiered_box"
    elif is_planar_facade and num_levels == 1:
        archetype = "complex_roof"
    else:
        archetype = "sculpted_facade"

    return BuildingClassification(
        archetype=archetype,
        roof_type="flat" if is_flat_roof else "complex",
        facade_type="planar" if is_planar_facade else "sculpted",
        metrics={
            "num_levels": num_levels,
            "plate_rms_m": round(plate_rms, 4) if plate_rms is not None else None,
            "roof_plateau_frac": round(roof_frac, 4) if roof_frac is not None else None,
            "sloped_fraction": round(float(sloped_fraction), 4),
            "max_wall_offset_m": round(max_wall_offset, 3),
            "has_arcs": has_arcs,
            "has_open_bands": has_open_bands,
            "party_faces": len(party),
        },
    )
