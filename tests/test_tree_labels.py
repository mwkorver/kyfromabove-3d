"""The tree rule (lidar.label_trees): the state's LiDAR labels no trees, so
the returns that lie on no surface are labeled them, leaving roofs, ridges
and walls."""
import numpy as np

from kyfromabove_3d.lidar import TREE_CLASS, label_trees


def _house_under_a_tree(seed=0):
    """A 12 x 10 m gabled house (eaves 4 m, ridge 7 m) at about 4 returns a
    square metre, its two long walls sampled thinly, the ground around it, and
    a bare crown scattered over it from 5 to 20 m."""
    rng = np.random.default_rng(seed)
    n = 480
    x, y = rng.uniform(0, 12, n), rng.uniform(0, 10, n)
    roof = np.column_stack([x, y, 7.0 - 3.0 * np.abs(y - 5.0) / 5.0 + rng.normal(0, 0.02, n)])
    walls = np.vstack([np.column_stack([rng.uniform(0, 12, 60), np.full(60, yy), rng.uniform(0.5, 4.0, 60)])
                       for yy in (0.0, 10.0)])
    g = rng.uniform(-10, 22, (1500, 2))
    ground = np.column_stack([g, rng.normal(0, 0.02, len(g))])
    crown = np.column_stack([rng.normal(6, 4, 700), rng.normal(5, 4, 700), rng.uniform(5, 20, 700)])
    xyz = np.vstack([roof, walls, ground, crown])
    cls = np.r_[np.ones(len(roof) + len(walls)), np.full(len(ground), 2), np.ones(len(crown))].astype(np.uint8)
    kind = np.r_[np.zeros(len(roof)), np.ones(len(walls)), np.full(len(ground), 2), np.full(len(crown), 3)]
    # a pulse through bare branches splits into several echoes (83% in St.
    # Matthews' crowns); one on a roof or a wall ends in one
    echoes = np.where(kind == 3, rng.choice([1, 2, 3, 4], len(kind), p=[0.17, 0.33, 0.3, 0.2]), 1)
    return xyz, cls, kind, roof, echoes


def test_a_crown_over_a_house_is_labeled_trees_and_its_roof_stays():
    xyz, cls, kind, roof, echoes = _house_under_a_tree()
    out = label_trees(xyz, cls, echoes)
    tree = out == TREE_CLASS
    assert tree[kind == 3].mean() > 0.8                 # the crown (its lowest returns, among the roof's, can stay)
    assert tree[kind == 0].mean() < 0.2                 # the roof, most of it
    assert (out[kind == 2] == 2).all()                  # ground is left alone
    # the roof's top, measured without the trees, is near its 7 m ridge (a
    # few crown returns just over it, among the roof's single echoes, can
    # stay), not the crown's 20 m
    inner = (xyz[:, 0] > 1.5) & (xyz[:, 0] < 10.5) & (xyz[:, 1] > 1.5) & (xyz[:, 1] < 8.5)
    on = inner & (cls == 1) & ~tree & (xyz[:, 2] > 2.0)
    assert 6.5 < np.percentile(xyz[on, 2], 99) < 9.0
    assert np.percentile(xyz[inner & (cls == 1) & (xyz[:, 2] > 2.0), 2], 99) > 15     # as it was


def test_walls_stay():
    xyz, cls, kind, _, echoes = _house_under_a_tree(1)
    out = label_trees(xyz, cls, echoes)
    assert (out[kind == 1] == TREE_CLASS).mean() < 0.2


def test_labels_the_survey_gave_are_kept():
    xyz, cls, kind, _, echoes = _house_under_a_tree(2)
    cls[kind == 0] = 6                                  # a survey that labels its buildings
    out = label_trees(xyz, cls, echoes)
    assert (out[kind == 0] == 6).all()


def test_a_small_smooth_clump_standing_alone_is_a_tree():
    """A dozen branch returns that happen to lie on a plane, 10 m over the
    roof and apart from it, set a house's top (6453e105: 18.7 m for 8.7 m)."""
    xyz, cls, kind, _, echoes = _house_under_a_tree(3)
    rng = np.random.default_rng(3)
    clump = np.column_stack([rng.uniform(5, 6.5, 15), rng.uniform(4, 5.5, 15), np.full(15, 17.0)])
    xyz = np.vstack([xyz, clump])
    cls = np.r_[cls, np.ones(15, dtype=np.uint8)]
    out = label_trees(xyz, cls, np.r_[echoes, np.full(15, 2)])
    assert (out[-15:] == TREE_CLASS).all()
    assert (out[:len(kind)][kind == 0] == TREE_CLASS).mean() < 0.2   # the roof, one large patch, stays


def test_a_spire_and_a_mast_stay():
    """Rough is not enough: a stone spire is rough but its pulses end in one
    echo (the Cathedral of the Assumption's top went 84 m to 50 m on
    roughness alone), and a lattice mast splits pulses as a crown does but
    stands higher than any tree (the Kentucky Home Life Building's, 70 to
    94 m up)."""
    xyz, cls, kind, _, echoes = _house_under_a_tree(4)
    rng = np.random.default_rng(4)
    spire = np.column_stack([rng.normal(40, 0.4, 120), rng.normal(5, 0.4, 120), rng.uniform(7, 25, 120)])
    mast = np.column_stack([rng.normal(6, 0.6, 150), rng.normal(5, 0.6, 150), rng.uniform(70, 94, 150)])
    xyz = np.vstack([xyz, spire, mast])
    cls = np.r_[cls, np.ones(270, dtype=np.uint8)]
    n = len(kind)
    out = label_trees(xyz, cls, np.r_[echoes, np.ones(120, int), rng.choice([2, 3], 150)])
    assert (out[n:n + 120] == TREE_CLASS).mean() < 0.1          # the spire
    assert (out[n + 120:] == TREE_CLASS).mean() < 0.1           # the mast
    assert (out[:n][kind == 3] == TREE_CLASS).mean() > 0.8      # the crown, still


#: the edge step as it was before TREE_RULE 4, looking at the 8
#: nearest surface returns only, and no level step
_RULE_3 = dict(edge_k=8, level_min=10**9)


def _roof_with_a_parapet(seed=0):
    """A 30 x 20 m flat roof at 24 m at the Phase 3 survey's 3.3 returns a
    square meter, a parapet cap 1.5 m wide round it at about 25.3 m, each
    pulse on the cap splitting and leaving a second echo on the roof behind
    it (Homewood Suites, Newport), and the ground around it."""
    rng = np.random.default_rng(seed)
    p = rng.uniform([0, 0], [30, 20], (int(30 * 20 * 3.3), 2))
    edge = np.minimum.reduce([p[:, 0], 30 - p[:, 0], p[:, 1], 20 - p[:, 1]]) < 1.5
    roof = np.column_stack([p, np.where(edge, 25.3 + rng.normal(0, 0.08, len(p)), 24.0 + rng.normal(0, 0.02, len(p)))])
    behind = roof[edge].copy()
    behind[:, :2] += rng.normal(0, 0.3, (len(behind), 2))
    behind[:, 2] = 24.0 + rng.normal(0, 0.02, len(behind))
    g = rng.uniform(-15, 45, (6000, 2))
    g = g[~((g[:, 0] > 0) & (g[:, 0] < 30) & (g[:, 1] > 0) & (g[:, 1] < 20))]
    ground = np.column_stack([g, rng.normal(0, 0.02, len(g))])
    xyz = np.vstack([roof, behind, ground])
    kind = np.r_[edge.astype(int), np.full(len(behind), 2), np.full(len(ground), 3)]
    cls = np.where(kind == 3, 2, 1).astype(np.uint8)
    echoes = np.where((kind == 1) | (kind == 2), 2, 1)
    return xyz, cls, kind, echoes


def test_a_parapet_over_a_flat_roof_stays():
    """A cap 1.3 m over its roof is higher than the edge step allows, and its
    split pulses read as branches: Homewood Suites' walls came out short of
    it (holes 19 -> 106 m2). It is level, as no crown is."""
    xyz, cls, kind, echoes = _roof_with_a_parapet()
    tree = label_trees(xyz, cls, echoes) == TREE_CLASS
    assert tree[kind == 1].mean() < 0.1                 # the cap, most of it
    assert tree[kind != 3].mean() < 0.02                # the whole roof
    assert (label_trees(xyz, cls, echoes, **_RULE_3)[kind == 1] == TREE_CLASS).mean() > 0.2   # as it was (a quarter)


def _upper_roof_beside_a_lower_one(seed=2):
    """An upper roof at 20 m (x 0 to 9) whose edge (x 9 to 10) is a rough
    cornice 0.2 to 0.9 m over it, its pulses splitting, beside a lower roof at
    12 m (x 10 to 30) flown more densely."""
    rng = np.random.default_rng(seed)
    up = rng.uniform([0, 0], [9, 20], (int(9 * 20 * 3.3), 2))
    up = np.column_stack([up, 20.0 + rng.normal(0, 0.02, len(up))])
    cornice = rng.uniform([9, 0], [10, 20], (132, 2))
    cornice = np.column_stack([cornice, rng.uniform(20.2, 20.9, len(cornice))])
    lo = rng.uniform([10.2, 0], [30, 20], (int(19.8 * 20 * 6), 2))
    lo = np.column_stack([lo, 12.0 + rng.normal(0, 0.02, len(lo))])
    g = rng.uniform(-15, 45, (6000, 2))
    g = g[~((g[:, 0] > 0) & (g[:, 0] < 30) & (g[:, 1] > 0) & (g[:, 1] < 20))]
    ground = np.column_stack([g, rng.normal(0, 0.02, len(g))])
    xyz = np.vstack([up, cornice, lo, ground])
    kind = np.r_[np.zeros(len(up)), np.ones(len(cornice)), np.full(len(lo), 2), np.full(len(ground), 3)]
    cls = np.where(kind == 3, 2, 1).astype(np.uint8)
    return xyz, cls, kind, np.where(kind == 1, 2, 1)


def test_an_edge_over_a_lower_roof_is_judged_against_its_own_roof():
    """The edge step once took the highest of the 8 nearest smooth returns,
    and beside a setback those were all on the lower roof, 8 m down."""
    xyz, cls, kind, echoes = _upper_roof_beside_a_lower_one()
    tree = label_trees(xyz, cls, echoes) == TREE_CLASS
    assert tree[kind == 1].mean() < 0.1                 # the cornice
    assert tree[kind != 3].mean() < 0.02
    assert (label_trees(xyz, cls, echoes, **_RULE_3)[kind == 1] == TREE_CLASS).mean() > 0.3   # as it was
