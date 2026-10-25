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
