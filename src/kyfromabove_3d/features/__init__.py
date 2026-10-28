"""Check features and match anchors derived from LiDAR.

Phase 1 originally hardcoded seven "check features" as literal coordinates with
no stated provenance; plotted on the imagery they land in the parking lot. This
module derives them from the LiDAR instead, so a check feature refers to a
surface that is actually there.

Two products:

``PlanarAnchor``
    A 3D point plus a locally-fitted surface plane. The plane is what makes
    cross-view patch rectification possible, which is what turns reprojection
    QA into a measurement instead of an assertion.

``CheckFeature``
    A named, human-recognisable point (roof corners, roof centroid) for visual
    inspection and reporting.
"""

from .planes import (  # noqa: F401
    _outward_normal, facade_tangent_basis, fit_local_planes, surface_coherence, tangent_basis,
)
from .surface import (  # noqa: F401
    assess_surface_complexity, classify_surface_points, FACADE_CAPTURE_PER_M2, facade_extent,
    FacadeExtent, NOISE_COHERENCE, ROUGH_RESIDUAL_M, split_surface_zones, STRUCTURE_COHERENCE,
    SurfaceComplexity, SurfacePointClasses, SurfaceZones,
)
from .walls import (  # noqa: F401
    _ccw_ring, _fit_circle, arc_faces, ArcWall, facade_faces, FacadeFace, infer_prism_faces,
    load_wall_planes, rectify_footprint_corners, save_wall_planes, VerifiedWallPlane, verify_face_plane,
    wall_planes, WallPlane,
)
from .ground import (  # noqa: F401
    ground_field_for_building, GroundField, OverhangShadow, predict_overhang_shadow, ray_to_wall_planes,
)
from .anchors import (  # noqa: F401
    _curved_roof_features, _local_surface_z, _plane_fit_rms, build_planar_anchors, building_surface,
    CheckFeature, extract_roof_points, ground_elevation_under, ground_surface, height_above_ground,
    PlanarAnchor, roof_check_features, roof_footprint,
)
from .profile import (  # noqa: F401
    _band_segments, band_extent, carry_extents_down, clip_profiles_by_neighbours, crown_bands,
    extrude_profile, face_profile, FaceBand, fill_under_roof_edge, find_overhangs,
    mirror_colour_from_twin, mirror_hidden_bands, opposite_face, profile_keeps, roof_behind_bands, WALL_STATUSES,
)
from .levels import (  # noqa: F401
    _along_edge, _mask_to_ring, _straight_upper_ring, building_levels, LEVEL_FACE_STRIDE,
)
from .structure import (  # noqa: F401
    BuildingClassification, classify_building_structure, FLAT_MAX_SLOPED_FRAC, FLAT_PLATE_RMS_M,
    FLAT_PLATEAU_FRAC, PLANAR_MAX_OFFSET_M,
)
