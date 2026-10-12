import time

import bpy
import bmesh
import numpy as np

from . import core, bridge


def _inside_out(signs):
    """How many faces are the way round the others are not.  The yardstick is whichever way
    round most of the selection is, so an island that was mirrored on purpose still counts
    as correct instead of adding itself to the number."""
    s = np.sign(signs)
    return int((s != (1.0 if s.sum() >= 0.0 else -1.0)).sum())


def _density_stats(density, area3d):
    """3D-area-weighted density CV and percentile-to-median ratios."""
    valid = np.isfinite(density) & np.isfinite(area3d) & (area3d > 1e-30)
    values = density[valid]
    weights = area3d[valid]
    if not len(values) or weights.sum() <= 0.0:
        return 0.0, 0.0, 0.0
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    cv = np.sqrt(max(variance, 0.0)) / max(mean, 1e-30)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])

    def quantile(q):
        index = np.searchsorted(cumulative, q * cumulative[-1], side="left")
        return float(sorted_values[min(index, len(sorted_values) - 1)])

    median = quantile(0.5)
    p90 = quantile(0.9)
    p99 = quantile(0.99)
    median = max(median, 1e-30)
    return float(cv), p90 / median, p99 / median


def _console_log(message):
    """Keep timing output visible in Blender's system console without changing operator UI."""
    print("[UV Relax] " + message, flush=True)


class UVRELAX_OT_relax(bpy.types.Operator):
    """Relax the selected part of a UV island.  Everything you did not select keeps its UV
    coordinates exactly, so seams stay matched.  A partial island selection keeps its untouched
    UV points fixed; selecting a whole island relaxes its outer contour too."""

    bl_idname = "uv.relax_even"
    bl_label = "Relax Selection"
    bl_options = {"REGISTER", "UNDO"}

    steps: bpy.props.IntProperty(
        name="Steps", default=200, min=1, max=5000,
        description="How many refinements one island may make inside a single solve.  The island's "
                    "shape is worked out rather than walked towards, so this only decides how exact "
                    "that answer is; raise it on a big island whose report says it ran out")

    density_weight: bpy.props.FloatProperty(
        name="Density Balance", default=0.30, min=0.0, max=1.0, soft_max=0.5,
        description="How much of the island's shape relaxation is allowed to spend in order to even "
                    "out texel density, weighed against the island's own area so one setting behaves "
                    "the same on a tightly packed island and a loosely packed one.  0 keeps pure shape "
                    "fidelity and leaves the packing as the unwrap left it.  Around this default the "
                    "packing evens out and the faces keep their own shape; past about 0.5 the evenness "
                    "is bought from the faces themselves, which start to look skewed, and under about "
                    "0.1 the packing does not even out -- some faces keep sitting far off the island's "
                    "average density")

    @classmethod
    def poll(cls, context):
        # Mesh.type reports None while the object is in edit mode, so look for UV layers
        ob = context.active_object
        return (context.mode == "EDIT_MESH" and ob is not None
                and getattr(ob.data, "uv_layers", None) is not None
                and ob.data.uv_layers.active is not None)

    def execute(self, context):
        t0 = time.perf_counter()
        me = context.active_object.data
        bm = bmesh.from_edit_mesh(me)
        got = bridge.read(bm)
        if got is None:
            _console_log("CANCEL reason=no_uv_map elapsed_ms=%.1f" %
                         ((time.perf_counter() - t0) * 1000.0))
            self.report({"ERROR"}, "This object has no UV map")
            return {"CANCELLED"}
        co, tri, lt, uvs, loops, uvl, poly_of_tri = got
        tuv = uvs[lt]
        read_ms = (time.perf_counter() - t0) * 1000.0

        inv = core.uvvert_ids(tri, tuv)
        U = int(inv.max()) + 1
        point_isl = core.islands(inv, U)
        isl = point_isl[inv[:, 0]]

        sync = bool(context.tool_settings.use_uv_select_sync)
        sync_uv_valid = bool(sync and getattr(bm, "uv_select_sync_valid", False))
        selection_source = ("UV_CORNERS" if sync_uv_valid else "MESH") if sync else "UV"
        selection = bridge.selection_counts(bm, sync)
        corner_selected = bridge.selected_corners(bm, lt, sync, context.tool_settings)
        selected_uv = np.zeros(U, dtype=bool)
        np.logical_or.at(selected_uv, inv.ravel(), corner_selected.ravel())
        selection_present = bool(selected_uv.any())
        if selection_present:
            labs = np.unique(point_isl[selected_uv])
            scope = np.any(selected_uv[inv], axis=1)
        else:
            # With nothing picked, relax every island on the mesh.
            labs = np.unique(point_isl)
            scope = np.isin(isl, labs)

        _console_log(
            "START object=%s mode=%s selected_verts=%d selected_edges=%d selected_faces=%d "
            "scope_faces=%d islands=%d selection_source=%s steps=%d density=%.3f" %
            (context.active_object.name, "SYNC" if sync else "UV",
             selection["verts"], selection["edges"], selection["faces"],
             int(scope.sum()), len(labs),
             selection_source,
             int(self.steps), float(self.density_weight)))
        setup_ms = (time.perf_counter() - t0) * 1000.0 - read_ms

        pos = np.zeros((U, 2))
        pos[inv.ravel()] = tuv.reshape(-1, 2)
        movable_uv = np.zeros(U, dtype=bool)
        # which UV points something else is matched to, so that holding them is what keeps a seam
        # together -- see core.glued_points
        glued = core.glued_points(tri, inv, isl)

        relaxed = 0
        points = 0
        even = 0
        settled = 0
        stuck = 0
        declined = 0
        turned = 0
        mirrored = 0
        mirrored_faces = 0
        ran_out = 0
        took = 0
        solves = 0
        solver_profile = {"model_ms": 0.0, "preconditioner_ms": 0.0,
                          "cg_ms": 0.0, "line_search_ms": 0.0}
        reduced_islands = 0
        reduced_faces = 0
        reduced_points = 0
        # which islands the loop actually wrote back: the overlap count is charged against these
        # faces only, so an island that was already stamped over a neighbour before the button was
        # pressed is not reported as this button's work
        moved_labs = []
        solve_t0 = time.perf_counter()
        for lab in labs:
            f = np.flatnonzero(isl == lab)
            ids = np.unique(inv[f])
            remap = -np.ones(U, np.int64)
            remap[ids] = np.arange(len(ids))
            tl = remap[inv[f]]

            selected_in_island = selected_uv[ids]
            partial_island = bool(selected_in_island.any() and not selected_in_island.all())
            local_mode = partial_island

            edge = np.sort(np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]],
                                           tl[:, [2, 0]]]), 1)
            uk, cnt = np.unique(edge, axis=0, return_counts=True)
            rim = np.zeros(len(ids), bool)
            rim[uk[cnt == 1].ravel()] = True
            if local_mode:
                # Only UV points explicitly selected by the user may move.  All other points are
                # exact Dirichlet conditions, including vertices on faces touching the selection.
                free = selected_in_island.copy()
            else:
                # The whole island relaxes, and the only part of its outline held where it stands is
                # the part something else is matched to -- a seam side, or a point two islands were
                # merged onto -- because moving one side of those is what tears a seam on the model.
                # The rest of the outline is the edge of the mesh itself: nothing continues past it,
                # nothing is glued to it, and it joins the unknowns.  Held anyway it would be the one
                # thing this could never repair, since every interior point then has to fit inside
                # whatever shape the unwrap happened to give the border.
                #
                # That the outline was held at all was measured, not guessed -- with density switched
                # off entirely the smallest outline chord angle fell 49.2 -> 28.7 deg on a 10080-face
                # island, and at matched density evenness 39.2 -> 33.4 deg on a 1872-face one -- so an
                # outline that is free is spent on the packing, and what keeps the island from being
                # squeezed into its neighbours is Density Balance and the report's own overlap count
                # rather than a wall that cannot be moved either way.  An island with no interior
                # point -- one face, a two-face strip -- has no outline to hold and would freeze
                # completely, so those still relax whole.
                edge = np.sort(np.concatenate([tl[:, [0, 1]], tl[:, [1, 2]],
                                               tl[:, [2, 0]]]), 1)
                uk, cnt = np.unique(edge, axis=0, return_counts=True)
                rim = np.zeros(len(ids), bool)
                rim[uk[cnt == 1].ravel()] = True
                # UV seams are intentional discontinuities.  Holding every seam
                # vertex makes a sharp outer contour impossible to repair, so
                # the whole-island path keeps the contour in the unknowns.
                free = np.ones(len(ids), bool)
            if not free.any():
                stuck += 1
                continue

            pf, a3 = core.uv3d(co, tri[f])
            solve_p = pf
            solve_tl = tl
            solve_uv = pos[ids]
            solve_pin = ~free
            solve_kwargs = {}
            reduced = False
            if local_mode:
                # Faces made entirely of held points contribute only a constant to the local
                # objective.  Drop them, while retaining every face that touches a free point and
                # the original island-wide reference values used by density and boundary terms.
                active_face = np.any(free[tl], axis=1)
                if active_face.any() and int(active_face.sum()) < len(f):
                    local_ids = np.unique(tl[active_face])
                    local_remap = -np.ones(len(ids), np.int64)
                    local_remap[local_ids] = np.arange(len(local_ids))
                    solve_tl = local_remap[tl[active_face]]
                    solve_p = pf[active_face]
                    solve_uv = pos[ids][local_ids]
                    solve_pin = ~free[local_ids]

                    q_full = core.signed_uv_area(pos[ids][tl])
                    valid_full = a3 > 1e-30
                    total_a3_full = float(a3.sum())
                    density_target_full = (0.5 * float(np.abs(q_full[valid_full]).sum()) /
                                           total_a3_full if total_a3_full > 1e-30 else 0.0)
                    full_boundary_edges, full_boundary_lengths = core.boundary_geometry(pf, tl)
                    if len(full_boundary_edges):
                        keep_boundary = np.all(local_remap[full_boundary_edges] >= 0, axis=1)
                        boundary_edges = local_remap[full_boundary_edges[keep_boundary]]
                        boundary_lengths = full_boundary_lengths[keep_boundary]
                    else:
                        boundary_edges = np.zeros((0, 2), np.int64)
                        boundary_lengths = np.zeros(0, dtype=float)
                    signs_full = np.sign(q_full)
                    majority_full = (1.0 if (signs_full > 0).sum() >= (signs_full < 0).sum()
                                     else -1.0)
                    solve_kwargs = {
                        "total_a3_override": total_a3_full,
                        "density_target_override": density_target_full,
                        "area2_override": float(np.abs(q_full).sum()),
                        "spread_override": float(np.linalg.norm(
                            pos[ids].max(0) - pos[ids].min(0)) + 1e-30),
                        "scale_center_override": pos[ids].mean(0),
                        "majority_orientation": majority_full,
                        "boundary_edges_override": boundary_edges,
                        "boundary_3d_length_override": boundary_lengths,
                        "boundary_count_override": len(full_boundary_edges),
                    }
                    reduced = True
                    reduced_islands += 1
                    reduced_faces += int(active_face.sum())
                    reduced_points += len(local_ids)
            newuv, st = core.relax_solve(
                solve_p, solve_tl, solve_uv, solve_pin, maxit=self.steps,
                density_weight=self.density_weight,
                boundary_weight=None, **solve_kwargs)
            for key in solver_profile:
                solver_profile[key] += st.get(key, 0.0)
            if st["pass"] == 0:
                # an island whose every point is held by a face left out is a different
                # answer from one that has settled, and saying the wrong one sends him
                # looking for a bug that is not there
                if st["why"] == "nothing to move":
                    stuck += 1
                elif st["why"] == "left alone":
                    # the island was solved and the only answers on offer were ones with more faces
                    # turned over and no better shape -- which is a different answer from "there was
                    # nothing to do"
                    declined += 1
                    turned += int(st["fold"])
                else:
                    even += 1
                    settled += int(st["free"])
                continue

            if not st["converged"] and st["fold_left"]:
                # a solve ran out of refinements and the island still holds faces turned over, so the
                # number he can raise is the answer to what he is looking at -- an island that came
                # out clean has nothing left for Steps to buy
                ran_out += 1
            if st["ask"] == "even" and st["fold_left"]:
                # the island was evened where it lies on top of itself rather than unfolded:
                # worth him knowing which of the two he is looking at
                mirrored += 1
                mirrored_faces += int(st["fold_left"])
            if reduced:
                pos[ids[local_ids]] = newuv
                movable_uv[ids[local_ids[free[local_ids]]]] = True
            else:
                pos[ids] = newuv
                movable_uv[ids[free]] = True
            moved_labs.append(lab)
            points += int(free.sum())
            relaxed += 1
            took = max(took, st["pass"])
            solves = max(solves, st["solve"])

        if relaxed == 0:
            why = []
            if even:
                why.append("%d point(s) on %d island(s) already keep their shape as well as "
                           "they can" % (settled, even))
            if stuck:
                why.append("%d island(s) in which every point touches a face you left out"
                           % stuck)
            if declined:
                why.append("%d island(s) with %d face(s) turned the way round the rest is not, "
                           "where relaxing would only flatten those further, so they were left "
                           "as they are" % (declined, turned))
            if not why:
                why = ["nothing was left to relax"]
            # a settled island is the selection doing its job; anything that stopped him short
            # of what he asked for has to be a popup, because a report at INFO level shows
            # nowhere on screen and the button then simply looks dead
            self.report({"WARNING" if (stuck or declined) else "INFO"},
                        "Nothing relaxed: " + ", ".join(why))
            _console_log(
                "CANCEL object=%s reason=nothing_relaxed elapsed_ms=%.1f read_ms=%.1f "
            "setup_ms=%.1f solve_ms=%.1f islands=%d reduced_islands=%d "
            "reduced_faces=%d reduced_points=%d solves=%d cg_passes=%d" %
                (context.active_object.name, (time.perf_counter() - t0) * 1000.0,
                 read_ms, setup_ms, (time.perf_counter() - solve_t0) * 1000.0,
                 len(labs), reduced_islands, reduced_faces, reduced_points, solves, took))
            return {"CANCELLED"}

        solve_ms = (time.perf_counter() - solve_t0) * 1000.0
        solver_profile["misc_ms"] = max(
            0.0, solve_ms - sum(solver_profile.values()))
        overlap_fix_t0 = time.perf_counter()
        moved_faces = np.isin(isl, moved_labs)
        overlap_initial = core.overlapping_pairs(
            inv, pos, isl, moved=moved_faces, return_pairs=True)
        pos, overlap_fix = core.untangle_overlaps(
            inv, pos, movable_uv, isl, moved_faces=moved_faces,
            initial=overlap_initial)
        overlap_fix_ms = (time.perf_counter() - overlap_fix_t0) * 1000.0
        write_t0 = time.perf_counter()
        new_loop_uv = np.empty_like(uvs)
        new_loop_uv[lt.ravel()] = pos[inv.ravel()]
        bridge.write(loops, uvl, new_loop_uv)
        bmesh.update_edit_mesh(me)
        write_ms = (time.perf_counter() - write_t0) * 1000.0

        post_t0 = time.perf_counter()
        p, a3 = core.uv3d(co, tri)
        metric_t0 = time.perf_counter()
        metric_prep = core.face_metric_prep(p)
        shape0, k0 = core.face_metric_uv(metric_prep, tuv)
        shape1, k1 = core.face_metric_uv(metric_prep, new_loop_uv[lt])
        density0 = _density_stats(k0[scope], a3[scope])
        density1 = _density_stats(k1[scope], a3[scope])
        was = float((k0[scope] * a3[scope]).sum())
        now = float((k1[scope] * a3[scope]).sum())
        turn0 = _inside_out(core.signed_uv_area(tuv)[scope])
        turn1 = _inside_out(core.signed_uv_area(new_loop_uv[lt])[scope])
        metric_ms = (time.perf_counter() - metric_t0) * 1000.0
        # the energy asks each face to be a good similarity of itself and is content with two of
        # them sharing one patch of the map, so evenness and zero folds do not add up to an
        # unoverlapped island; what is left over is counted and said, not priced into the solve.
        # Every face of the mesh is looked at, not just the ones in scope, because an outline that
        # may move can grow into a neighbour that was never touched -- and a pair is only charged
        # against this button where one of its two faces sits on an island this button wrote back.
        within, across = overlap_fix["within"], overlap_fix["across"]
        overlap_ms = overlap_fix_ms
        tail = ""
        if declined:
            tail += " | %d island(s) left alone: %d face(s) in them are turned the way round " \
                    "the rest is not, and relaxing only flattens those further" % (declined,
                                                                                    turned)
        if stuck:
            tail += " | %d island(s) where every point touches a face you left out" % stuck
        if mirrored:
            tail += " | %d island(s) were evened out where they lie on top of themselves rather " \
                    "than unfolded: %d face(s) are still the wrong way round, and only cutting a " \
                    "seam takes that off" % (mirrored, mirrored_faces)
        if ran_out:
            tail += " | %d island(s) worked to the Steps limit and still hold faces turned over " \
                    "-- raise Steps" % ran_out
        if within:
            tail += " | %d face pair(s) still lie on top of each other inside one island, which " \
                    "only cutting a seam takes off" % within
        if across:
            tail += " | %d face pair(s) now lie on top of a neighbouring island because a border " \
                    "that moved took the room -- repack the UVs" % across
        post_ms = (time.perf_counter() - post_t0) * 1000.0
        ms = (time.perf_counter() - t0) * 1000.0
        _console_log(
            "DONE object=%s elapsed_ms=%.1f read_ms=%.1f setup_ms=%.1f solve_ms=%.1f "
            "write_ms=%.1f post_ms=%.1f metric_ms=%.1f overlap_ms=%.1f "
            "solver_model_ms=%.1f precond_ms=%.1f cg_ms=%.1f density_line_ms=%.1f "
            "solver_misc_ms=%.1f "
            "selected_verts=%d selected_edges=%d "
            "selected_faces=%d islands=%d relaxed_islands=%d reduced_islands=%d "
            "reduced_faces=%d reduced_points=%d points=%d solves=%d "
            "cg_passes=%d overlap_fix=%d->%d/%d "
            "selection_source=%s steps=%d density=%.3f" %
            (context.active_object.name, ms, read_ms, setup_ms, solve_ms, write_ms, post_ms,
             metric_ms, overlap_ms,
             solver_profile["model_ms"], solver_profile["preconditioner_ms"],
             solver_profile["cg_ms"], solver_profile["line_search_ms"],
             solver_profile["misc_ms"],
             selection["verts"], selection["edges"], selection["faces"], len(labs), relaxed,
             reduced_islands, reduced_faces, reduced_points, points, solves, took,
             overlap_fix["before"], overlap_fix["after"], overlap_fix["rounds"],
             selection_source,
             int(self.steps), float(self.density_weight)))
        self.report({"WARNING" if (declined or stuck or ran_out or within or across) else "INFO"},
                    "%d UV points on %d island(s) relaxed in %.0f ms | stretch median %.3f "
                    "-> %.3f, worst 1%% %.2f -> %.2f | density CV %.3f -> %.3f, "
                    "p90/median %.2f -> %.2f, p99/median %.2f -> %.2f | "
                    "relaxed UV area %+.1f%% | %d face(s) inside out, was %d | "
                    "%d solve(s), %d refinement(s)%s"
                    % (points, relaxed, ms,
                       float(np.median(shape0[scope])), float(np.median(shape1[scope])),
                       float(np.percentile(shape0[scope], 99)),
                       float(np.percentile(shape1[scope], 99)),
                       density0[0], density1[0], density0[1], density1[1],
                       density0[2], density1[2],
                       100.0 * (now / was - 1.0), turn1, turn0, solves, took, tail))
        return {"FINISHED"}
