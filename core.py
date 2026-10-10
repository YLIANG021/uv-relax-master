# numpy-only UV math.  No bpy types: every function runs on plain arrays, so the exact
# same code is testable outside Blender.

import numpy as np


def face_metric_prep(p):
    """Precompute the model-only part of face_metric: Ginv @ TN.

    This depends only on the 3D geometry, so it can be hoisted out of the relax loop.
    Pass the result to face_metric_uv() for each UV candidate.
    Returns (T,2,2) array."""
    t1 = p[:, 1] - p[:, 0]
    t2 = p[:, 2] - p[:, 0]
    la = np.sqrt(np.maximum((t1 * t1).sum(1), 1e-30))
    c = (t1 * t2).sum(1) / la
    lb2 = np.sqrt(np.maximum(((t2 - c[:, None] * (t1 / la[:, None])) ** 2).sum(1), 1e-30))
    a = (t1 * t1).sum(1)
    b = (t1 * t2).sum(1)
    d = (t2 * t2).sum(1)
    detG = np.maximum(a * d - b * b, 1e-30)
    Ginv = np.stack([np.stack([d, -b], 1), np.stack([-b, a], 1)], 1) / detG[:, None, None]
    TN = np.stack([np.stack([la, np.zeros_like(la)], 1), np.stack([c, lb2], 1)], 1)
    return Ginv @ TN


def face_metric_uv(prepped, tuv):
    """Per-face slack from precomputed geometry and UV corners tuv (T,3,2).

    Returns (shape, det):  det = UV area / 3D area, i.e. texel density up to one
    constant shared by the whole mesh;  shape = distortion ratio, 1.0 means conformal."""
    s1 = tuv[:, 1] - tuv[:, 0]
    s2 = tuv[:, 2] - tuv[:, 0]
    S = np.stack([np.stack([s1[:, 0], s2[:, 0]], 1), np.stack([s1[:, 1], s2[:, 1]], 1)], 1)
    K = S @ prepped
    k00, k01, k10, k11 = K[:, 0, 0], K[:, 0, 1], K[:, 1, 0], K[:, 1, 1]
    tr = k00 * k00 + k01 * k01 + k10 * k10 + k11 * k11
    det = k00 * k11 - k01 * k10
    disc = np.sqrt(np.maximum(tr * tr - 4 * det * det, 0.0))
    g1 = np.sqrt(np.maximum((tr + disc) * 0.5, 1e-30))
    g2 = np.sqrt(np.maximum((tr - disc) * 0.5, 1e-30))
    return g1 / np.maximum(g2, 1e-12), np.abs(det)


def face_metric(p, tuv):
    """Per-face slack from model corners p (T,3,3) and UV corners tuv (T,3,2).

    Returns (shape, det):  det = UV area / 3D area, i.e. texel density up to one
    constant shared by the whole mesh;  shape = distortion ratio, 1.0 means conformal."""
    return face_metric_uv(face_metric_prep(p), tuv)


def uv3d(co, tri):
    """Model corners and 3D face areas of triangulation (tri = (T,3) vertex ids)."""
    p = co[tri]
    a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return p, a3


def uv_area(p, a3, tuv):
    """Summed UV area of faces given model corners, their 3D areas and UV corners."""
    return float((face_metric(p, tuv)[1] * a3).sum())


def uvvert_ids(tri, tuv):
    """Identity of a UV point = same 3D vertex AND same UV coordinate.  Coincident
    corners along one seam merge; corners at different seams of a vertex stay apart."""
    key = np.stack([tri, np.rint(tuv[..., 0] * 1e5).astype(np.int64),
                    np.rint(tuv[..., 1] * 1e5).astype(np.int64)], -1).reshape(-1, 3)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(tri.shape)


def _root(par, i):
    while True:
        p = par[i]
        done = p == i
        if done.all():
            return i
        i = np.where(done, i, p)


def islands(inv, U):
    """UV-space connected components, labelled 0..n-1, one label per UV point."""
    par = np.arange(U)
    for _ in range(60):
        old = par.copy()
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ra = _root(par, inv[:, a])
            rb = _root(par, inv[:, b])
            m = ra != rb
            if m.any():
                hi = np.maximum(ra[m], rb[m])
                lo = np.minimum(ra[m], rb[m])
                o = np.unique(hi, return_index=True)[1]
                par[hi[o]] = lo[o]
        if np.array_equal(par, old):
            break
    _, lab = np.unique(_root(par, np.arange(U)), return_inverse=True)
    return lab


def glued_points(tri, inv, isl):
    """Which UV points something else is matched to, so moving that one alone would show on the model.

    Two kinds of glue.  A 3D vertex that owns more than one UV point is cut by a seam: the two sides
    of the cut sit at different places in the map and each samples its own patch of the texture, so
    moving one of them and not the other is exactly what makes a seam visibly tear.  And a UV point
    that faces of more than one island sit on is the mirror case -- `uvvert_ids` merged two islands
    there, and they are one unknown now, so they cannot be pulled apart.

    What is left is the edge of the model itself: nothing on the mesh continues past it and nothing in
    the map is matched to it, so relaxing may move it.  That is what lets a border that was unwrapped
    badly be repaired instead of being kept as the shape everything else has to fit inside.
    `isl` are per-face island labels.  Returns (U,) bool, one entry per UV point."""
    U = int(inv.max()) + 1
    glued = np.zeros(U, bool)
    v = tri.ravel()
    w = inv.ravel()
    if len(v) == 0:
        return glued
    sides = np.unique(np.stack([v, w], 1), axis=0)
    cnt = np.bincount(sides[:, 0], minlength=int(v.max()) + 1)
    glued[sides[cnt[sides[:, 0]] > 1, 1]] = True
    both = np.unique(np.stack([w, np.repeat(np.asarray(isl), 3)], 1), axis=0)
    cnt = np.bincount(both[:, 0], minlength=U)
    glued[both[cnt[both[:, 0]] > 1, 0]] = True
    return glued


def signed_uv_area(t):
    """Twice each face's UV area, with its sign: negative means the face is inside out.
    Takes the face's three corners (T,3,2)."""
    return (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) \
        - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])


# how far a model triangle's angles are allowed to go before its stiffness is capped: a face
# squashed flat would otherwise ask for an infinite weight, and a face thrown out entirely asks for
# nothing at all, which leaves it dead material the rest of the island flows around.  RizomUV
# clamps the angle it reads a cotangent weight from into [MinAngle, pi - MinAngle] and its
# `#MinAngle` default is the same order as this.
MIN_ANGLE = 1e-5
MIN_ANGLE_SIN = float(np.sin(MIN_ANGLE))


def stretch_terms(p, tuv, e_side=None):
    """Per triangle, the stretch that relaxing is meant to remove.

    The triangle's own map from the model to UV is compared with the nearest rotation plus
    uniform scale of *the orientation it already has*, so what is left over is shear and
    uneven scaling only, and an island that was mirrored on purpose stays mirrored.  Pass
    `e_side` (+1 or -1) to ask every triangle for that orientation instead, which is what an
    island with a few faces folded over wants: see relax_solve.  Returns (A, w): the two
    coefficient blocks A (T,2,3,2) and the weight w (T,).  The island's stretch energy is then
    sum_t w_t * (A_t . its six corners)**2 -- quadratic in the UV coordinates, so walking it
    down is a matter of repeating one simple step.

    The weight is 1/(8 * model area), so a face squashed towards a line wants to be stiffer than
    any of its neighbours without limit.  It is capped instead of dropped: the area the weight is
    taken from is floored at the smallest area the face's own two longest edges allow once every
    angle is held inside [MIN_ANGLE, pi - MIN_ANGLE], which bounds how hard one sliver can pull
    while still letting it take part in the relax.  Only a face with no length in it at all says
    nothing about stretch, and gets weight zero.  And note that this energy is as content with a
    smaller island as with an even one: see relax_solve."""
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 2] - p[:, 1]
    l1 = np.sqrt(np.maximum((e1 * e1).sum(1), 1e-30))
    c = (e1 * e2).sum(1) / l1
    b2 = np.sqrt(np.maximum(((e2 - (c / l1)[:, None] * e1) ** 2).sum(1), 0.0))
    a3 = 0.5 * l1 * b2
    edges = np.sort(np.stack([np.sqrt((e1 * e1).sum(1)), np.sqrt((e2 * e2).sum(1)),
                              np.sqrt((e3 * e3).sum(1))], 1), 1)
    a3_cap = np.maximum(a3, 0.5 * edges[:, 1] * edges[:, 2] * MIN_ANGLE_SIN)
    if e_side is None:
        e = np.sign(signed_uv_area(tuv))
        e[e == 0.0] = 1.0
    else:
        e = np.full(len(l1), float(e_side))
    z = np.zeros_like(b2)
    A1 = np.stack([np.stack([-b2, e * (l1 - c)], -1), np.stack([b2, e * c], -1),
                   np.stack([z, -e * l1], -1)], 1)
    A2 = np.stack([np.stack([c - l1, -e * b2], -1), np.stack([-c, e * b2], -1),
                   np.stack([l1, z], -1)], 1)
    w = np.zeros_like(a3)
    np.divide(1.0, 8.0 * a3_cap, out=w, where=a3_cap > 1e-30)
    return np.stack([A1, A2], 1), w


# how far out the density term is asked to care about a single face: its error is the logarithm of
# the face's density over the island's target, saturated at this factor in both directions
DENSITY_CAP = 8.0
BOUNDARY_LOG_CAP = 2.5
BOUNDARY_WEIGHT_BASE = 0.12
BOUNDARY_WEIGHT_DENSITY = 0.42


def relax_solve(p, tl, uv0, pin, maxit=200, knee=2.0, density_weight=0.30,
                fold_rounds=6, density_refine=8, density_tol=0.3,
                density_face_weight=None, boundary_weight=None):
    """Bring one island to the evenest shape its boundary allows -- by solving for it.

    The conformal part of the energy this relaxes is a quadratic form in the UV coordinates:
    `sum_t w_t * (A_t . its six corners)**2` (see `stretch_terms`), whose coefficients depend only
    on the model and on which orientation each face is asked for, so its Hessian is the constant
    `2 sum_t w_t A_t^T A_t` and finding its bottom is a linear system.  That sum is divided by the
    island's own UV area, which is what makes it a number of the same kind as the density term below
    -- without it the trade-off between the two would be set by however large the author happened to
    pack the island, and one `density_weight` would mean a different balance on every island.  A
    triangle's UV area is quadratic in its corners rather than linear, so the density part is not, and
    it enters as its first-order expansion about the state the solve starts from -- a Gauss-Newton
    step, whose model matches the true energy's value and gradient there, re-expanded by the next solve
    about the state that one reached.  With the density weight at zero the objective is exactly
    quadratic and the first solve is already the global minimum of the problem, so `density_refine`
    costs nothing.

    The system is solved by preconditioned conjugate gradients, matrix free: Hessian times a vector
    is the same two `einsum` products the gradient used to be made of, and the diagonal of that same
    Hessian -- what the old Jacobi step divided by -- is the preconditioner.  Conjugate gradients
    rather than a dense Cholesky because the dense form costs the square of the island's point count
    in memory and time (30 MB for an island of 1900 points, and no room to grow), while this costs
    O(n) memory and its iteration count tracks the island's linear resolution.

    That replaces a walk which spent eight hundred passes, a tenth of a folded face each, crawling
    down a bowl it could have jumped to the bottom of.  It is why no tuning of the trust radius, the
    step count or the restarts ever bought more than about the same wall time per face: the cost
    was in the format of the iteration, not in its parameters.

    Points the caller names in `pin` -- those the selection left out, and those it holds because
    something else is matched to them -- are not a force here but Dirichlet conditions whose
    displacement is held at zero; which outline points belong in that set is the caller's business,
    and this module does not look at borders at all.  Held rather than softly pulled is what makes
    seam alignment a promise instead of a trend: a point no selected face pulls hardly moves, a
    pinned one does not move.  A wholly loose island keeps the four global similarity modes out of
    the Krylov directions, rather than holding any physical UV point; the packing it started with
    is put back after every accepted density round, and once more at the end for the rounds that
    carry no density term to hold it.

    Faces the wrong way round are not priced either.  A flip is not a place on the energy surface,
    it is a change of *which problem the island should be solved for*, so it is handled
    combinatorially, the way LSCM's own inversion step does: solve asking every face for the majority
    orientation, look at what the global solution still leaves inside out, pin the corners of those
    faces where the solution put them, and solve again.  Every round moves the whole island at once
    and each one is the exact answer to a convex problem, so a fold opens by as much as the rest of
    the island's least squares will pay for, rather than by as far as one Jacobi step can carry.
    Once a round comes back clean the pins stay on while the density settles, and the last solve
    releases them, so the opened fold is not left in the shape chosen for the round that opened it.

    What pinning costs is stated rather than hidden: a fold whose own boundary is folded too cannot
    be opened this way, because there is nothing left outside the pins to push from.  That is the
    non-disk case, and the answer to it is a seam, not a longer solve.

    p: (T,3,3) model corners, tl: (T,3) ids into uv0, uv0: (V,2), pin: (V,) bool.
    `maxit` is the conjugate-gradient cap of one solve, `knee` the point where the unevenness score
    stops squaring (see `shape`), `density_weight` the strength of the island-local texel-density
    term against the island's own shape term (see `density_error`), `fold_rounds` how many times the wrong-way-round pins may grow,
    `density_refine` how many
    Gauss-Newton re-expansions a clean island may ask for at most (a round whose step has gone too
    small to matter ends them sooner), and `density_tol` how much of a density round's gradient its
    solve has to remove -- the round only needs the direction, since the length is chosen afterwards
    on the true energy, and asking the solve for more than that costs iterations the step length
    then throws away.
    Returns (new_uv, stats) with stats["pass"] the CG iterations used, stats["solve"] how many solves
    that covers, stats["converged"] whether every one of them reached its residual tolerance instead
    of running out of iterations, stats["why"] ("relaxed", "already even", "nothing to move",
    "left alone"), stats["gain"] the drop of the solved energy, stats["fold"] how many faces were
    inside out to begin with, stats["fold_left"] how many still are in what is handed back, and
    stats["ask"] which of the two questions won: "open" asked for one orientation and worked to turn
    the folds back, "even" left them mirrored and evened them out instead."""
    V = len(uv0)
    q0 = signed_uv_area(uv0[tl])
    sg = np.sign(q0)
    maj = 1.0 if (sg > 0).sum() >= (sg < 0).sum() else -1.0
    wrong = (sg != maj) & (sg != 0.0)
    fold = bool(wrong.any())

    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    valid = a3 > 1e-30
    total_a3 = float(a3.sum())
    weight = a3 / total_a3 if total_a3 > 1e-30 else np.zeros_like(a3)
    density_weight = max(float(density_weight), 0.0)
    # Boundary protection is now a separate soft geometric metric.  Keep the old argument in the
    # signature so a reloaded operator from the previous version remains callable.
    # the density target is the island's own starting UV area per 3D area, so the term evens the
    # packing inside the island without asking the island to change size
    density_target = (0.5 * float(np.abs(q0[valid]).sum()) / total_a3) if total_a3 > 1e-30 else 0.0
    use_d = density_weight > 0.0 and density_target > 1e-30
    area2_0 = float(np.abs(q0).sum())
    # the density residual is a ratio of areas and has no units, while each conformal term is
    # 1/(8*model area) times coefficients that carry model length squared: the model scale cancels
    # (which is why scaling his mesh changes nothing) and what is left is (UV length)^2, so the sum
    # grows as the SQUARE of however big the author packed the island.  Dividing it by that area is
    # what makes one Density Balance mean the same shape / evenness trade-off on a tightly packed
    # island and a loosely packed one: his mesh's islands span UV areas 0.04 to 0.36, so the single
    # slider value was really nine different ratios (.test/sweep_norm_balance.py).  An island
    # collapsed to a point has no area to divide by and no stretch to minimise either, so there the
    # weights stay as they are.
    conf_norm = 1.0 / area2_0 if area2_0 > 1e-30 else 1.0
    spread = float(np.linalg.norm(uv0.max(0) - uv0.min(0)) + 1e-30)

    prepped = face_metric_prep(p)

    # Recover the outline from topology and measure it in 3D.  The target UV length uses the
    # island's starting average density, so a bad starting outline can move freely without making
    # the whole island change scale just to satisfy the boundary term.
    all_edges = np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]], tl[:, [2, 0]]], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    all_model_edges = np.concatenate([p[:, [0, 1]], p[:, [1, 2]], p[:, [2, 0]]], axis=0)
    edge_order = np.lexsort((all_edges[:, 1], all_edges[:, 0]))
    sorted_edges = all_edges[edge_order]
    sorted_model_edges = all_model_edges[edge_order]
    unique_edges, first_edge, edge_count = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True)
    boundary_edges = unique_edges[edge_count == 1]
    boundary_model_edges = sorted_model_edges[first_edge[edge_count == 1]]
    boundary_3d_length = np.linalg.norm(
        boundary_model_edges[:, 1] - boundary_model_edges[:, 0], axis=1)
    boundary_valid = boundary_3d_length > 1e-30
    boundary_edges = boundary_edges[boundary_valid]
    boundary_3d_length = boundary_3d_length[boundary_valid]
    boundary_target = np.sqrt(max(density_target, 1e-30)) * boundary_3d_length
    if boundary_weight is None:
        # Normalized below, so this does not depend on the number of outline edges.  Density
        # pressure raises the protection automatically; there is no user-facing B slider to tune.
        boundary_weight = BOUNDARY_WEIGHT_BASE + BOUNDARY_WEIGHT_DENSITY * min(density_weight, 1.0)
    boundary_weight = max(float(boundary_weight), 0.0)
    boundary_weights = (np.full(len(boundary_edges), boundary_weight / len(boundary_edges))
                        if len(boundary_edges) else np.zeros(0, dtype=float))

    def shape(u):
        """How uneven the island is, the way the overlay sees it: every face's stretch above 1.0
        counted by square up to the knee, then by plain amount, so one ruined face cannot outvote
        the six hundred ordinary ones around it.  Blind to which way round a face lies, which is
        why every comparison below counts the folds first and reaches shape only to break that
        tie."""
        z = np.maximum(face_metric_uv(prepped, u[tl])[0] - 1.0, 0.0)
        z = z * z if knee is None else np.where(z <= knee, z * z, 2.0 * knee * z - knee * knee)
        return float(z.mean())

    flat = tl.ravel()

    def scatter(gs):
        """Sum the per-face corner contributions (T,3,2) onto the island's points."""
        out = np.empty((V, 2))
        for k in (0, 1):
            out[:, k] = np.bincount(flat, weights=gs[:, :, k].ravel(), minlength=V)
        return out

    def scatter_edges(gs):
        """Sum boundary-edge endpoint contributions (E,2,2) onto island points."""
        out = np.zeros((V, 2))
        if len(boundary_edges):
            np.add.at(out, boundary_edges[:, 0], gs[:, 0])
            np.add.at(out, boundary_edges[:, 1], gs[:, 1])
        return out

    def scatter_blocks(gs):
        """Sum per-face 2x2 corner blocks (T,3,2,2) onto island points."""
        out = np.zeros((V, 2, 2))
        np.add.at(out, tl.ravel(), gs.reshape(-1, 2, 2))
        return out

    def scatter_edge_blocks(gs):
        """Sum per-boundary-edge 2x2 endpoint blocks (E,2,2,2) onto island points."""
        out = np.zeros((V, 2, 2))
        if len(boundary_edges):
            ids = boundary_edges.ravel()
            np.add.at(out, ids, gs.reshape(-1, 2, 2))
        return out

    def ask(e_side):
        """An orientation trial's fixed coefficients: every face asked for `e_side` (+1, -1) or for
        the way round it already is (None)."""
        A, w = stretch_terms(p, uv0[tl], e_side=e_side)
        w = w * conf_norm
        if e_side is None:
            signs = np.where(sg == 0.0, 1.0, sg)
        else:
            signs = np.full(len(q0), float(e_side))
        return A, w, signs

    def area_grad(u):
        """d(UV double area)/d(each corner), (T,3,2) -- linear in the corners, because the area
        itself is quadratic in them."""
        t = u[tl]
        t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
        return np.stack([
            np.stack([t1[:, 1] - t2[:, 1], t2[:, 0] - t1[:, 0]], 1),
            np.stack([t2[:, 1] - t0[:, 1], t0[:, 0] - t2[:, 0]], 1),
            np.stack([t0[:, 1] - t1[:, 1], t1[:, 0] - t0[:, 0]], 1),
        ], 1)

    def density_error(signs, u):
        """How far each face sits from the island's own texel density, as the logarithm of the ratio
        of the two, saturated at DENSITY_CAP either way.

        The logarithm is what keeps the term from needling an island.  `ratio - 1` charges a face
        five times too large 25x what it charges one five times too small, so the rounds buy their
        fall by flattening the loudest outlier: on a 1872-face island whose outline wedge started at
        38.4 deg the convex stage left faces at x5.36 and x0.27, and the density stage collapsed the
        wedge to 7.2 deg to even the big one.  `.test/probe_needle_energy.py` scores that needle
        0.00759 against 0.0494 for the state that kept the wedge, so the linear error did mean it --
        but the same probe says the needle was a basin, not a verdict -- the state the logarithm
        settles on (wedge 18.6 deg) is cheaper under BOTH errors.  Faces near the target are hardly
        touched either, since `log(ratio)` and `ratio - 1` share a slope at ratio 1; what changes is
        that crushing a shape no longer pays for an outlier, and DENSITY_CAP stops one ruined face
        outvoting the six hundred tight ones around it.
        """
        rat = np.ones_like(a3)
        np.divide(signs * signed_uv_area(u[tl]), 2.0 * density_target * a3, out=rat, where=valid)
        err = np.log(np.clip(rat, 1.0 / DENSITY_CAP, DENSITY_CAP))
        err[~valid] = 0.0
        return err

    def density_lin(signs, base):
        """First-order expansion of that error at `base`, as the affine function `L . u - r` whose
        value agrees with the error there.  The area is quadratic in the corners, so what the
        expansion leaves out grows with the square of how far the step goes -- which is why each
        round takes the length the true energy will vouch for, not the one the model asks for."""
        err = density_error(signs, base)
        L = np.zeros((len(a3), 3, 2))
        np.divide(signs[:, None, None] * area_grad(base),
                  (2.0 * density_target * a3)[:, None, None], out=L, where=valid[:, None, None])
        # d(log ratio) = d(ratio) / ratio.  The divisor is the saturated ratio rather than the true
        # one, so a face beyond the cap keeps a small slope instead of leaving the model flat there
        L = L / np.exp(err)[:, None, None]
        return L, np.einsum("tcv,tcv->t", L, base[tl]) - err

    def boundary_error(u):
        """Robust log edge-length residual against the 3D metric target."""
        if not len(boundary_edges):
            return np.zeros(0, dtype=float)
        d = u[boundary_edges[:, 1]] - u[boundary_edges[:, 0]]
        length = np.linalg.norm(d, axis=1)
        ratio = np.ones_like(length)
        np.divide(length, boundary_target, out=ratio, where=boundary_target > 1e-30)
        return np.log(np.clip(ratio, np.exp(-BOUNDARY_LOG_CAP), np.exp(BOUNDARY_LOG_CAP)))

    def boundary_lin(base):
        """Linearize log(|uv edge| / target) at the current UV state."""
        if not len(boundary_edges):
            return np.zeros((0, 2, 2)), np.zeros(0, dtype=float)
        d = base[boundary_edges[:, 1]] - base[boundary_edges[:, 0]]
        length2 = np.einsum("ei,ei->e", d, d)
        length = np.sqrt(np.maximum(length2, 1e-30))
        err = boundary_error(base)
        L = np.zeros((len(boundary_edges), 2, 2), dtype=float)
        # d(log |b-a|) = (b-a) . (db-da) / |b-a|^2
        L[:, 0] = -d / length2[:, None]
        L[:, 1] = d / length2[:, None]
        return L, np.einsum("eij,eij->e", L, base[boundary_edges]) - err

    def system(A, w, signs, base, dens, boundary_factor=1.0):
        """The quadratic that stands in for the objective at `base`: Hessian-times-vector, the
        Hessian's diagonal, the constant part of its gradient, and a function that scores any state
        by this same quadratic.  `dens` False leaves the density term out, which makes it the
        conformal energy exactly -- the problem the fold rounds want, because that one is convex and
        its bottom is the whole island's answer.  With the density term in, the scoring function is
        what a step promises, the true `energy` being what the step is judged on."""
        lin = None
        wd = None
        blin = None
        wb = None
        boundary_diag = np.zeros((V, 2))

        def hvp(v):
            d = np.einsum("tkcv,tcv->tk", A, v[tl])
            gs = np.einsum("tkcv,tk->tcv", A, d) * (2.0 * w)[:, None, None]
            if lin is None:
                out = scatter(gs)
            else:
                L, _ = lin
                out = scatter(gs + (2.0 * wd * np.einsum("tcv,tcv->t", L, v[tl]))[:, None, None] * L)
            if blin is not None:
                Lb, _ = blin
                db = np.einsum("eij,eij->e", Lb, v[boundary_edges])
                out = out + scatter_edges((2.0 * wb * db)[:, None, None] * Lb)
            return out

        face_blocks = 2.0 * w[:, None, None, None] * np.einsum(
            "tkci,tkcj->tcij", A, A)
        dg = np.diagonal(face_blocks, axis1=2, axis2=3)
        c0 = np.zeros((V, 2))
        if dens:
            L, r = density_lin(signs, base)
            wd = density_weight * weight
            lin = (L, r)
            c0 = scatter((-2.0 * wd * r)[:, None, None] * L)
            density_blocks = (2.0 * wd)[:, None, None, None] * np.einsum(
                "tci,tcj->tcij", L, L)
            face_blocks = face_blocks + density_blocks
            dg = dg + np.diagonal(density_blocks, axis1=2, axis2=3)

        if len(boundary_edges):
            Lb, rb = boundary_lin(base)
            wb = boundary_weights * boundary_factor
            blin = (Lb, rb)
            c0 = c0 + scatter_edges((-2.0 * wb * rb)[:, None, None] * Lb)
            boundary_blocks = (2.0 * wb)[:, None, None, None] * np.einsum(
                "eai,eaj->eaij", Lb, Lb)
            dg_b = np.diagonal(boundary_blocks, axis1=2, axis2=3)
            boundary_diag = scatter_edges(dg_b)
        else:
            boundary_blocks = np.zeros((0, 2, 2, 2))

        def value(u):
            d = np.einsum("tkcv,tcv->tk", A, u[tl])
            e = float((w[:, None] * d * d).sum())
            if lin is not None:
                L, off = lin
                e += float((wd * (np.einsum("tcv,tcv->t", L, u[tl]) - off) ** 2).sum())
            if blin is not None:
                Lb, off_b = blin
                e += float((wb * (np.einsum("eij,eij->e", Lb, u[boundary_edges]) - off_b) ** 2).sum())
            return e

        block_diag = scatter_blocks(face_blocks) + scatter_edge_blocks(boundary_blocks)
        return hvp, scatter(dg) + boundary_diag, c0, value, block_diag

    def energy(A, w, signs, u, dens):
        """The objective itself rather than its local stand-in, so successive Gauss-Newton steps can
        be compared on the thing they are trying to reduce."""
        d = np.einsum("tkcv,tcv->tk", A, u[tl])
        e = float((w[:, None] * d * d).sum())
        if dens:
            err = density_error(signs, u)
            e += density_weight * float((weight * err * err).sum())
        if len(boundary_edges):
            e += float((boundary_weights * boundary_error(u) ** 2).sum())
        return e

    def inverted(u):
        """Faces of the island lying the way round the majority is not, in the state u."""
        s = np.sign(signed_uv_area(u[tl]))
        return (s != maj) & (s != 0.0)

    def inside_out(u):
        return int(inverted(u).sum())

    loose = not bool(pin.any())
    # A loose island has a four-dimensional similarity nullspace.  These are global gauge freedoms,
    # not boundary conditions: fixing two real UV vertices turns them into visible corners.  The
    # solve below projects its directions away from translation, rotation and scale instead, so every
    # island vertex remains free to move.
    gauge = None
    scale_gauge = None
    if loose:
        cen = uv0.mean(0)
        q = uv0 - cen
        gauge = np.stack([
            np.broadcast_to(np.array([1.0, 0.0]), (V, 2)),
            np.broadcast_to(np.array([0.0, 1.0]), (V, 2)),
            np.stack([-q[:, 1], q[:, 0]], axis=1),
            q,
        ], axis=2)
    else:
        # A partial solve has real Dirichlet points, so translation and rotation are already
        # determined.  Its remaining failure mode is a uniform contraction of the free region
        # towards the held part; remove only that one scale direction, leaving local shape changes
        # unconstrained.
        scale_gauge = (uv0 - uv0.mean(0))[:, :, None]
    # which points may move is worked out once, before any solve, and shared by both questions below:
    # a point no live triangle sits on has no stiffness and nothing to be even about
    _, stiff, _, _, _ = system(*ask(maj), uv0, False)
    held = pin | (stiff.min(1) <= 0.0)

    stats = {"free": int((~held).sum()), "pass": 0, "solve": 0, "gain": 0.0, "why": "relaxed",
             "fold": int(wrong.sum()), "ask": None, "converged": True, "fold_left": 0}
    if held.all():
        stats["why"] = "nothing to move"
        return uv0.copy(), stats
    flip0 = inside_out(uv0)

    def restore(v):
        """Put the island's packing back at the area it started with, about its own centre.  The
        energy is as small for a shrunken island as for an even one, and with nothing held there is
        nothing else to settle its scale.  A uniform positive scaling turns no face over."""
        now = float(np.abs(signed_uv_area(v[tl])).sum())
        if now <= 1e-18:
            return v
        cen = v.mean(0)
        return (v - cen) * np.sqrt(area2_0 / now) + cen

    def solve(A, w, signs, clamp, base, dens, tol=1e-6, model=None):
        """One global move over every point not held by selection or a fold clamp.

        On a loose island, each Krylov vector is projected off the four global similarity modes.
        This fixes the numerical gauge without fixing any physical boundary vertex.
        """
        if model is None:
            model = system(A, w, signs, base, dens)
        hvp, diag, c0, _, block_diag = model
        live = (~held & ~clamp)[:, None]
        basis = None
        if gauge is not None and not clamp.any():
            basis = gauge
        elif scale_gauge is not None:
            basis = scale_gauge

        # Build the small projection basis once per linear solve.  The old path recomputed a
        # least-squares fit over every UV point for every CG vector; the basis is fixed while this
        # solve runs, so a QR factorization gives the same orthogonal projection with two cheap
        # matrix-vector products per iteration.
        projector = None
        if basis is not None:
            active = live[:, 0]
            M = basis[active].reshape(-1, basis.shape[2])
            if len(M):
                Q, _ = np.linalg.qr(M, mode="reduced")
                projector = Q

        def project(v):
            v = v * live
            if projector is None:
                return v
            active = live[:, 0]
            flat = v[active].reshape(-1)
            if len(flat) == 0:
                return v
            out = v.copy()
            out[active] = (flat - projector @ (projector.T @ flat)).reshape(-1, 2)
            return out

        g = project(hvp(base) + c0)
        g0 = float(np.linalg.norm(g))
        inv_diag = live / np.maximum(diag, 1e-30)
        block_inv = np.zeros_like(block_diag)
        active = live[:, 0]
        if active.any():
            blocks = block_diag[active]
            scale = np.maximum(np.max(np.abs(blocks), axis=(1, 2)), 1e-30)
            det = blocks[:, 0, 0] * blocks[:, 1, 1] - blocks[:, 0, 1] * blocks[:, 1, 0]
            good = det > 1e-12 * scale * scale
            if good.any():
                selected = blocks[good]
                inv_det = 1.0 / det[good]
                inverse = np.empty_like(selected)
                inverse[:, 0, 0] = selected[:, 1, 1] * inv_det
                inverse[:, 1, 1] = selected[:, 0, 0] * inv_det
                inverse[:, 0, 1] = -selected[:, 0, 1] * inv_det
                inverse[:, 1, 0] = -selected[:, 1, 0] * inv_det
                active_ids = np.flatnonzero(active)
                block_inv[active_ids[good]] = inverse

        def precondition(v):
            """Apply a per-UV-point 2x2 Jacobi block, with scalar fallback for flat points."""
            out = np.einsum("vij,vj->vi", block_inv, v)
            fallback = active & (np.max(np.abs(block_inv), axis=(1, 2)) <= 0.0)
            if fallback.any():
                out[fallback] = v[fallback] * inv_diag[fallback]
            return out * live

        r = project(-g)
        # what the model says is left to win; when that is nothing the island is already standing
        # where this solve would put it, and there is no iteration worth spending
        if 0.5 * float((r * precondition(r)).sum()) <= 1e-12 * max(
                energy(A, w, signs, base, dens), 1e-30):
            return base.copy(), 0, False, np.zeros((V, 2))
        x = np.zeros((V, 2))
        z = project(precondition(r))
        dirn = z.copy()
        rz = float((r * z).sum())
        cap = max(1, int(maxit))
        iters = 0
        out = True
        for iteration in range(1, cap + 1):
            hp = project(hvp(dirn))
            ph = float((dirn * hp).sum())
            if ph <= 1e-30 or rz <= 1e-30:
                out = False
                break
            step = rz / ph
            x = project(x + step * dirn)
            r = project(r - step * hp)
            iters = iteration
            # the recycled residual drifts away from the true one once the Krylov space has grown
            # long, and a drift nobody catches says "converged" about a state that is still moving
            rn = float(np.linalg.norm(r))
            check = rn
            if iteration % 20 == 0 or rn <= tol * g0:
                rt = project(-(hvp(base + x) + c0))
                check = float(np.linalg.norm(rt))
                if abs(check - rn) > 0.5 * check:
                    r = rt
                    z = project(precondition(r))
                    dirn = z.copy()
                    rz = float((r * z).sum())
                    continue
            if check <= tol * g0:
                out = False
                break
            z = project(precondition(r))
            rz_new = float((r * z).sum())
            dirn = project(z + (rz_new / rz) * dirn)
            rz = rz_new
        return base + x, iters, out, x

    def run(e_side, rounds, pin_folds):
        """Ask the island the `e_side` question, and keep the state worth handing back: fewest faces
        inside out, and the lowest objective among those.

        Two stages, because the two terms want different problems.  The first solves the convex
        conformal quadratic and grows the set of pinned wrong-way-round faces between solves; the
        density term is left out of it, since its linearization is only good near the state it was
        taken at and a fold has to travel through the flat state to open, which a model taken at the
        folded state does not ask for.  Measured on a grid island of 2178 faces with 196 of them
        inside out and a block of them stamped down to a twenty-fifth of their size: the convex
        rounds cleared all 196 in one solve and left no pair of faces over each other.  The second
        stage brings the density back as Gauss-Newton rounds about the opened island."""
        A, w, signs = ask(e_side)
        best = [None]
        base = uv0
        clamp = np.zeros(V, bool)
        count = [0, 0, False]

        def note(iters, hit):
            count[0] += iters
            count[1] += 1
            count[2] = count[2] or hit

        def keep(cand):
            # the objective the solves are actually minimising, folds first: `shape` is blind to
            # density, and keying on it threw every density state the island reached away
            key = (inside_out(cand), energy(A, w, signs, cand, True))
            if best[0] is None or key < best[0][0]:
                best[0] = (key, cand, count[0], count[1], count[2])
            return key

        def step(clamp_at, base_at, dens):
            cand, iters, hit, _ = solve(A, w, signs, clamp_at, base_at, dens)
            note(iters, hit)
            keep(cand)
            return cand

        def polish(base_at, clamp_at):
            """Even the packing out, one Gauss-Newton round at a time about where the last round
            left the island.

            The density model is only good near where it was taken: a face's UV area is quadratic in
            its corners, so its affine model goes on promising an area the face can only reach by
            turning itself over.  So each round solves for one direction -- no damping term, only its
            length -- and tries it at t = 1, 1/2, 1/4..., keeping the first t that turns no face over
            and whose fall in the TRUE objective is at least a tenth of what the model's own slope
            promises for that length.  The slope is read off the model at three points of the ray
            rather than taken as the fall at t = 1, because the round's solve is deliberately inexact
            and the parabola's promise is only the model's own at the exact minimiser; on the islands
            measured so far the two tests pick the same length, so that is a correctness argument
            rather than a speed one.  Descent is then guaranteed by construction rather than by hoping
            the linearization reached, and a round costs one conjugate-gradient solve."""
            base = base_at
            for _ in range(density_refine):
                model = system(A, w, signs, base, True)
                value = model[3]
                cand, iters, hit, x = solve(
                    A, w, signs, clamp_at, base, True, tol=density_tol, model=model)
                note(iters, hit)
                if iters == 0:
                    break
                e_last = energy(A, w, signs, base, True)
                # m(t) = e_last + a t + b t^2 along the ray, read off the model at t = 1/2 and t = 1
                slope = 4.0 * (value(base + 0.5 * x) - e_last) - (value(cand) - e_last)
                fold_last = inside_out(base)
                moved = None
                for t in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    u = base + t * x
                    if loose:
                        u = restore(u)
                    if (inside_out(u) <= fold_last
                            and energy(A, w, signs, u, True) <= e_last + 0.1 * t * slope):
                        moved = u - base
                        base = u
                        keep(base)
                        break
                if moved is None or float(np.abs(moved).max()) <= 1e-7 * spread:
                    break
            return base

        if pin_folds:
            while rounds > 0:
                rounds -= 1
                cand = step(clamp, base, False)
                bad = inverted(cand)
                base = cand
                if not bad.any():
                    break
                nxt = clamp.copy()
                nxt[tl[bad].ravel()] = True
                # a fold that is already pinned, or an island with nothing left outside the pins,
                # has no further pinning to do: that is the non-disk case and it is reported, not
                # fought
                if (nxt == clamp).all() or int((~held & ~nxt).sum()) < 3:
                    break
                clamp = nxt
        if not pin_folds:
            # the mirrored question gets no fold rounds: one convex solve is its whole answer
            base = step(clamp, base, False)
        if use_d:
            base = polish(base, clamp)
        if pin_folds and clamp.any():
            # the pins have done their job pushing the good material into the room the fold needs;
            # let go of them and see whether the island settles even with the fold opened
            base = step(np.zeros(V, bool), base, False)
            if use_d:
                base = polish(base, np.zeros(V, bool))
        return best[0][1], best[0][2], best[0][3], best[0][4], \
            energy(A, w, signs, uv0, True) - energy(A, w, signs, best[0][1], True)

    at, used, nsolve, out, gain = run(maj if fold else None, fold_rounds, True)

    def good(f, drop):
        """Worth handing back at all: fewer faces folded over than the island started with, or as
        many and a lower objective.  `drop` is the objective's fall under the question that trial
        asked, so each trial is judged against its own start -- the two questions are different
        problems and their energies do not compare.  Shape alone would rather have a fold flattened
        on top of its neighbours than opened, and it is blind to density."""
        return f < flip0 or (f == flip0 and drop > 1e-15)

    trials = [("open", at, used, nsolve, out, gain)]
    if fold and (inside_out(at) > 0 or not good(inside_out(at), gain)):
        # the mirrored-flat question is one solve, not a second campaign: it says what the island
        # looks like if the folds are left lying where they are and merely evened out, which on an
        # overlap that pinning cannot open is the better answer
        trials.append(("even",) + run(None, 1, False))
    win = None
    for name, t_at, t_used, t_solves, t_out, t_gain in trials:
        if not good(inside_out(t_at), t_gain):
            continue
        rank = (inside_out(t_at), shape(t_at))
        if win is None or rank < win[0]:
            win = (rank, name, t_at, t_used, t_solves, t_out, t_gain)
    if win is None:
        stats["ask"] = "open"
        stats["why"] = "left alone" if fold else "already even"
        return uv0.copy(), stats
    _, stats["ask"], at, stats["pass"], stats["solve"], out, stats["gain"] = win
    if loose:
        # a no-op for anything the density rounds touched -- they keep the area themselves -- but
        # the density-free answers need their scale put back
        at = restore(at)
    stats["converged"] = not out
    stats["fold_left"] = inside_out(at)
    return at, stats


def overlapping_pairs(tl, u, lab=None, moved=None, window=200, block=65536):
    """How many pairs of these faces lie on top of each other in UV.

    The relaxed energy cannot see this: it asks each face to be a good similarity of itself and is
    satisfied by two faces sharing one patch of the map, which is why an island can come back with
    every face the right way round and still have material stamped over material.  Overlap is a
    property of pairs, so it is measured on pairs and reported, rather than pulled into the
    objective, where it would need a barrier of its own and a gradient for it.

    Two triangles overlap when no edge normal of either separates them -- the separating-axis test,
    which never has to build the intersection polygon.  A pair counts only where the two overlap by
    more than a twentieth of the smaller face's own edge on *every* one of those six axes, so the
    way a conformal triangulation meets along a shared edge does not count against the island, and
    neither does a block of faces that was itself stamped down to a hundredth of its size.

    The broad phase sweeps the boxes in x: each face is compared with the faces whose left edge
    starts inside its own span, which is every pair that could possibly overlap and no pair that
    could not, then the y boxes and the island labels filter the list, and the rest is tested in
    blocks.  That keeps the work linear for an island of roughly equally sized faces; a face whose
    x span holds more than `window` others is beyond what this counts, and that case is an island
    stamped over itself wholesale, where the answer is a seam anyway.

    tl: (T,3) point ids, u: (V,2) their UV coordinates, lab: (T,) island label per face, or None to
    compare every face in the batch with every other.  `moved`: (T,) bool, which faces this relax was
    allowed to move -- given, only pairs with a moved face in them are counted, and they come back
    split two ways, because an island stamped over itself and an island that grew over a neighbour
    are different problems: the first is opened by a seam, the second by the packer.
    Returns (pairs within one island, pairs across islands)."""
    t = u[tl]
    area2 = signed_uv_area(t)
    biggest = float(np.abs(area2).max()) if len(area2) else 0.0
    f = np.flatnonzero(np.abs(area2) > 1e-12 * max(biggest, 1e-30))
    if f.size < 2:
        return 0, 0
    t = t[f]
    n = len(t)
    lab = np.zeros(n, np.int64) if lab is None else np.asarray(lab)[f]
    mv = None if moved is None else np.asarray(moved)[f]
    box = np.stack([t.min(1), t.max(1)], 1)
    edge_vectors = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1],
                             t[:, 0] - t[:, 2]], 1)
    edge_normals = np.stack([-edge_vectors[..., 1], edge_vectors[..., 0]], -1)
    # each face's own size, from its area: the edge of an equilateral triangle of double area S is
    # sqrt(2 S / sqrt 3)
    scale = np.sqrt(2.0 * np.abs(area2[f]) / 1.7320508) + 1e-30
    order = np.argsort(box[:, 0, 0])
    left = box[order, 0, 0]
    ends = np.searchsorted(left, box[order, 1, 0], side="right")
    cnt = np.clip(ends - np.arange(n) - 1, 0, window)
    total = int(cnt.sum())
    if total == 0:
        return 0, 0
    starts = np.cumsum(cnt) - cnt
    ii = np.repeat(np.arange(n), cnt)
    jj = ii + 1 + (np.arange(total) - np.repeat(starts, cnt))
    a, b = order[ii], order[jj]
    # with `moved` given, a pair is only charged against this run where one of its two faces sits on
    # an island the run wrote back -- and then a pair of two *different* islands counts too, which is
    # what an outline set free can grow into.  Two islands that were already on top of each other
    # before the button was pressed are not the button's doing.
    sel = (box[a, 1, 1] > box[b, 0, 1]) & (box[b, 1, 1] > box[a, 0, 1])
    if mv is not None:
        sel &= mv[a] | mv[b]
    a, b = a[sel], b[sel]
    same = lab[a] == lab[b]

    def narrow(ii):
        """Separating-axis test over the listed pairs, in blocks."""
        hits = 0
        for s in range(0, len(ii), block):
            pa, pb = a[ii[s:s + block]], b[ii[s:s + block]]
            ta, tb = t[pa], t[pb]
            ax = np.concatenate([edge_normals[pa], edge_normals[pb]], 1)
            ga = np.einsum("pnc,pkc->pnk", ta, ax)
            gb = np.einsum("pnc,pkc->pnk", tb, ax)
            gap = np.minimum(ga.max(1), gb.max(1)) - np.maximum(ga.min(1), gb.min(1))
            # the slack is the smaller face's own size, not the batch's: a block stamped down over
            # itself is made of faces a hundred times smaller than the ones around it, and a
            # tolerance measured on those would let the stamp pass for clean
            slack = 0.05 * np.minimum(scale[pa], scale[pb])[:, None]
            hits += int((gap > slack).all(1).sum())
        return hits

    return narrow(np.flatnonzero(same)), 0 if mv is None else narrow(np.flatnonzero(~same))
