"""Temporal operations for tquery: before/after, within, inside/outside.

All functions operate on boolean masks (pd.Series) aligned to the
input DataFrame's index. They never mutate the DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eval_before_after(
    df: pd.DataFrame,
    left_mask: pd.Series,
    right_mask: pd.Series,
    op: str,
    pid_col: str,
    date_col: str,
    every_left: bool = False,
    every_right: bool = False,
    any_left: bool = False,
    any_right: bool = False,
    left_offset_days: int = 0,
    right_offset_days: int = 0,
) -> pd.Series:
    """Person-level temporal comparison: does left occur before/after right?

    Returns a row-level boolean mask where all rows for matching persons
    are marked True (for the left-hand side events).

    Default semantics ("first vs first" — NOT existential):
        before: min(left) < min(right)  — the first left precedes the first right
        after:  min(left) > min(right)  — the first left follows the first right
        simultaneously: any left date == any right date

    Existential semantics (any_left / any_right) — "some X vs some Y":
        before: any_right (with non-every left) → min(left) < max(right)
                (`X before any Y` / `any X before any Y`); `any X before Y`
                is a documented no-op (∃x < first(Y) ⇔ default).
        after:  any_left (with non-every right) → max(left) > min(right)
                (`any X after Y` / `any X after any Y`); `X after any Y`
                is the mirrored no-op.
        simultaneously: `any` is a no-op (already existential).

    Universal semantics (every_left / every_right):
        For 'after K51':
            every_right=True  → max(K51) < max(K50): every K51 is followed by some K50
            every_left=True   → min(K50) > min(K51): every K50 is preceded by some K51
            both              → min(K50) > max(K51): every K50 follows every K51

        'before' is symmetric (swap roles).

        'simultaneously' universal: every event on the quantified side has a
        same-date partner on the other side.

    Universal sides additionally require non-empty events for that side
    (no vacuous truth: a person with no K51 does NOT satisfy `every K51`).
    """
    if not left_mask.any() or not right_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]
    left_shift = pd.Timedelta(days=left_offset_days)
    right_shift = pd.Timedelta(days=right_offset_days)

    if op == "simultaneously":
        # Build per-person date sets for both sides (offsets applied)
        left_dates = (dates[left_mask] + left_shift).groupby(pid[left_mask]).apply(set)
        right_dates = (dates[right_mask] + right_shift).groupby(pid[right_mask]).apply(set)
        common = left_dates.index.intersection(right_dates.index)
        if common.empty:
            return pd.Series(False, index=df.index)
        matching_pids: set = set()
        for p in common:
            l = left_dates[p]
            r = right_dates[p]
            if every_left and not l.issubset(r):
                continue
            if every_right and not r.issubset(l):
                continue
            if not every_left and not every_right and not (l & r):
                continue
            matching_pids.add(p)
        return pid.isin(matching_pids) & left_mask

    # before / after: per-person aggregates
    left_groups = dates[left_mask].groupby(pid[left_mask])
    right_groups = dates[right_mask].groupby(pid[right_mask])

    # Need both min and max for universal comparisons
    left_min = left_groups.min()
    left_max = left_groups.max()
    right_min = right_groups.min()
    right_max = right_groups.max()

    common_pids = left_min.index.intersection(right_min.index)
    if common_pids.empty:
        return pd.Series(False, index=df.index)

    left_min = left_min.reindex(common_pids) + left_shift
    left_max = left_max.reindex(common_pids) + left_shift
    right_min = right_min.reindex(common_pids) + right_shift
    right_max = right_max.reindex(common_pids) + right_shift

    # NOTE: the historical default for `K50 before K51` is "first K50 < first K51",
    # which mathematically corresponds to "∃ K50 (the earliest one) ∀ K51, k50 < k51"
    # — i.e. universal quantification over the right side. By symmetry, the historical
    # default for `K50 after K51` corresponds to universal LHS. We preserve those
    # semantics so adding `every` on the "implied" side is a no-op (documented).
    if op == "before":
        if every_left and every_right:
            matching = left_max < right_min   # every left strictly before every right
        elif every_left:
            matching = left_max < right_max   # every K50 has at least one K51 after it
        elif every_right:
            # `K50 before every K51` — same as the default (first < first)
            matching = left_min < right_min
        elif any_right:
            # existential: some K50 before some K51 (`any` on the left is
            # a no-op here — ∃x < first(Y) ⇔ first(X) < first(Y))
            matching = left_min < right_max
        else:
            matching = left_min < right_min
    else:  # after
        if every_left and every_right:
            matching = left_min > right_max
        elif every_right:
            matching = left_max > right_max   # every K51 has at least one K50 after it
        elif every_left:
            # `every K50 after K51` — same as the default (first > first)
            matching = left_min > right_min
        elif any_left:
            # existential: some K50 after some K51 (`any` on the right is
            # the mirrored no-op)
            matching = left_max > right_min
        else:
            matching = left_min > right_min

    matching_pids = set(matching.index[matching])
    return pid.isin(matching_pids) & left_mask


def window_bands(
    direction: str | None, min_days: int, days: int
) -> list[tuple[int, int]]:
    """Translate a window spec into closed day-offset bands.

    A target (ref) row at day `t` qualifies for a query (child) row at
    day `q` iff `t ∈ [q + a, q + b]` for some band `(a, b)`:

    - `after`  — child after ref: `child − ref ∈ [min_days, days]`
      ⇒ ref ∈ [child − days, child − min_days].
    - `before` — child before ref: `ref − child ∈ [min_days, days]`.
    - `around`, signed (`min_days < 0`): signed diff `child − ref`
      in `[min_days, days]` (covers wholly-negative windows too).
    - `around`, unsigned: `|child − ref| ∈ [min_days, days]` — one band
      when `min_days == 0`, two disjoint bands otherwise.
    - `None` (no direction, with ref) behaves like unsigned `around`.
    """
    if direction == "after":
        return [(-days, -min_days)]
    if direction == "before":
        return [(min_days, days)]
    if min_days < 0:
        return [(-days, -min_days)]
    if min_days == 0:
        return [(-days, days)]
    return [(-days, -min_days), (min_days, days)]


def mirror_bands(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Swap query/target roles: if target ∈ [q + a, q + b], then from the
    target's viewpoint query ∈ [t − b, t − a]."""
    return [(-b, -a) for a, b in bands]


def band_window_match(
    pid_codes: np.ndarray,
    day: np.ndarray,
    query_mask: np.ndarray,
    target_mask: np.ndarray,
    bands: list[tuple[int, int]],
    query_shift: int = 0,
    target_shift: int = 0,
    exclude_self: bool = True,
) -> np.ndarray:
    """Existence test: for each query row, is there a target row of the
    same person whose day falls in one of the closed bands?

    Backend-neutral numpy core shared by the pandas and Polars
    evaluators. Counts *all* qualifying targets per band (not just the
    nearest), so lower-bounded windows cannot produce false negatives.

    Args:
        pid_codes: int person codes per row (factorized pids).
        day: int64 day number per row (dates truncated to day precision).
        query_mask / target_mask: bool arrays over all rows.
        bands: closed day-offset bands from `window_bands`; a target at
            day `t` qualifies for a query row at day `q` iff
            `t ∈ [q + a, q + b]`. Bands must be non-overlapping.
        query_shift / target_shift: signed day offsets added to the
            query / target dates before comparison (shifted anchors).
        exclude_self: if True, a row never satisfies the window as its
            own target — `X inside 0 to 5 days after X` then means
            "another X row 0-5 days after an X", so a lone X does not
            match itself but a second X on the same date does.

    Returns:
        Bool array over all rows: True at query rows with >= 1
        qualifying target (excluding, if requested, the row itself).
    """
    n = len(pid_codes)
    out = np.zeros(n, dtype=bool)
    q_idx = np.flatnonzero(query_mask)
    t_idx = np.flatnonzero(target_mask)
    if q_idx.size == 0 or t_idx.size == 0:
        return out

    q_code = pid_codes[q_idx].astype(np.int64)
    q_day = day[q_idx] + query_shift
    t_code = pid_codes[t_idx].astype(np.int64)
    t_day = day[t_idx] + target_shift

    order = np.lexsort((t_day, t_code))
    t_code = t_code[order]
    t_day = t_day[order]

    # Composite (person, day) keys so one sorted array serves all persons.
    base = int(t_day.min())
    t_norm = t_day - base
    span = int(t_norm.max()) + 2
    t_keys = t_code * span + t_norm

    counts = np.zeros(q_idx.size, dtype=np.int64)
    for a, b in bands:
        lo = q_day + a - base
        hi = q_day + b - base
        impossible = (hi < 0) | (lo > span - 1)
        lo_c = np.clip(lo, 0, span - 1)
        hi_c = np.clip(hi, 0, span - 1)
        left = np.searchsorted(t_keys, q_code * span + lo_c, side="left")
        right = np.searchsorted(t_keys, q_code * span + hi_c, side="right")
        band_counts = right - left
        band_counts[impossible] = 0
        counts += band_counts

    required = np.ones(q_idx.size, dtype=np.int64)
    if exclude_self:
        # The query row itself sits at signed offset (target_shift −
        # query_shift) in its own band arithmetic; when that offset is
        # inside a band AND the row is also a target, it was counted —
        # demand one additional (i.e. genuinely other) target.
        self_delta = target_shift - query_shift
        if any(a <= self_delta <= b for a, b in bands):
            required += target_mask[q_idx].astype(np.int64)

    out[q_idx[counts >= required]] = True
    return out


def scalar_window_agg(v: np.ndarray, func: str, relative: bool = False) -> float:
    """Spec-conformant scalar aggregate over one window's raw values.

    `v` may contain NaN (skipped). Empty-window defaults follow the
    aggregate table in spec/semantics.md: sum → 0, count → 0, everything
    else → NaN (which fails any threshold comparison).
    """
    v = v[~np.isnan(v)]
    n = v.size
    if func == "sum":
        return float(v.sum()) if n else 0.0
    if func in ("count", "n"):
        return float(n)
    if n == 0:
        return float("nan")
    if func in ("mean", "avg"):
        return float(v.mean())
    if func == "min":
        return float(v.min())
    if func == "max":
        return float(v.max())
    if func == "median":
        return float(np.median(v))
    if func == "sd":
        return float(v.std(ddof=1)) if n > 1 else float("nan")
    if func == "var":
        return float(v.var(ddof=1)) if n > 1 else float("nan")
    if func == "range":
        mn = float(v.min())
        spread = float(v.max()) - mn
        if not relative:
            return spread
        return spread / mn if mn > 0 else float("nan")
    if func == "rise":
        if n == 1:
            return 0.0
        cm = np.minimum.accumulate(v)
        if not relative:
            return float((v - cm).max())
        safe = cm > 0
        if not safe.any():
            return 0.0
        return float(np.where(safe, (v - cm) / np.where(safe, cm, 1.0), 0.0).max())
    if func == "fall":
        if n == 1:
            return 0.0
        cm = np.maximum.accumulate(v)
        if not relative:
            return float((cm - v).max())
        safe = cm > 0
        if not safe.any():
            return 0.0
        return float(np.where(safe, (cm - v) / np.where(safe, cm, 1.0), 0.0).max())
    raise ValueError(f"Unknown aggregate function: {func!r}")


def per_ref_window_agg_pass(
    pid_codes: np.ndarray,
    day: np.ndarray,
    values: np.ndarray,
    ref_mask: np.ndarray,
    bands: list[tuple[int, int]],
    ref_shift: int,
    func: str,
    op,
    threshold: float,
    relative: bool = False,
) -> np.ndarray:
    """Universal-ref anchored aggregate: `AGG(col) OP x inside ... every REF`.

    For EACH reference row, aggregate `values` over the rows of the same
    person falling in that ref's own day-window, and test the threshold.
    A person matches iff they have at least one ref (no vacuous truth)
    and EVERY ref's window-aggregate passes. Empty windows follow
    `scalar_window_agg` defaults (sum/count → 0 pass normally; NaN
    aggregates fail). The anchor row itself is part of its own window
    when the band contains offset 0 (no self-exclusion — matches the
    existential anchored-aggregate convention).

    Args:
        pid_codes / day: as in `band_window_match`; rows MUST be sorted
            by (pid, date) so (pid_code, day) keys are nondecreasing.
        values: float array of the aggregated column (NaN = missing).
        bands: child-relative bands from `window_bands` — a row at day
            `d` is in ref `t`'s window iff `t + shift ∈ [d + a, d + b]`,
            i.e. `d ∈ [t + shift − b, t + shift − a]`.
        op: comparison callable (operator.gt etc.); NaN compares False.

    Returns:
        Bool array over all rows: True at every row of matching persons.
    """
    n = len(pid_codes)
    out = np.zeros(n, dtype=bool)
    r_idx = np.flatnonzero(ref_mask)
    if r_idx.size == 0:
        return out

    codes = pid_codes.astype(np.int64)
    base = int(day.min())
    norm = day - base
    span = int(norm.max()) + 2
    row_keys = codes * span + norm

    r_code = codes[r_idx]
    r_day = day[r_idx] + ref_shift

    # Row-window bounds per ref and band: [t − b, t − a] in day space.
    # Bands sorted by descending b so windows concatenate in chronological
    # order (matters for order-sensitive aggregates: rise / fall).
    slices: list[tuple[np.ndarray, np.ndarray]] = []
    for a, b in sorted(bands, key=lambda ab: -ab[1]):
        lo = r_day - b - base
        hi = r_day - a - base
        impossible = (hi < 0) | (lo > span - 1)
        lo_c = np.clip(lo, 0, span - 1)
        hi_c = np.clip(hi, 0, span - 1)
        left = np.searchsorted(row_keys, r_code * span + lo_c, side="left")
        right = np.searchsorted(row_keys, r_code * span + hi_c, side="right")
        right = np.where(impossible, left, right)
        slices.append((left, right))

    passed = np.empty(r_idx.size, dtype=bool)
    for i in range(r_idx.size):
        parts = [values[lo[i]:hi[i]] for lo, hi in slices]
        window_vals = parts[0] if len(parts) == 1 else np.concatenate(parts)
        agg = scalar_window_agg(window_vals, func, relative)
        passed[i] = bool(op(agg, threshold)) if agg == agg else False

    failed_codes = np.unique(r_code[~passed])
    ok_codes = np.setdiff1d(np.unique(r_code), failed_codes)
    return np.isin(codes, ok_codes)


def eval_within_days(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series | None,
    days: int,
    direction: str | None,
    pid_col: str,
    date_col: str,
    min_days: int = 0,
    every_left: bool = False,
    every_right: bool = False,
    ref_offset_days: int = 0,
    exclude_self: bool = True,
) -> pd.Series:
    """Row-level: mark child rows within a day range of reference events.

    A child row matches iff at least one reference row of the same person
    lies in the closed day-band(s) implied by `direction`, `min_days` and
    `days` (see `window_bands`). ALL reference rows are considered, not
    just the nearest one, so lower-bounded windows (`inside 30 to 90 days
    after Y`) match whenever ANY ref falls in the band.

    Args:
        days: Maximum days distance (upper bound).
        min_days: Minimum days distance (lower bound, for 'between M and N days').
        direction: 'before', 'after', 'around', or None.
        every_left: If True, every child event must have a qualifying ref event.
        every_right: If True, every ref event must have a qualifying child event.
        exclude_self: If True (default), a row never serves as its own
            reference. Only relevant when a row matches both child and
            ref patterns (`X inside 0 to 5 days after X`); a *different*
            same-date row still qualifies. Anchored aggregates pass
            False so an anchor row stays inside its own window.

    If ref_mask is None, this is a standalone time window relative to
    the first event per person (no self-exclusion — the anchor is a
    per-person date, not a row).

    Universal modes return ALL child rows for matching persons (not just
    the matched ones), matching the convention of `eval_before_after`.
    """
    if not child_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    dates = df[date_col]

    if ref_mask is None:
        # No reference: within day range of first event per person.
        # (ref_offset_days ignored here — no ref to shift.)
        first_date = dates.groupby(pid).transform("first")
        diff = (dates - first_date).dt.days.abs()
        return child_mask & (diff >= min_days) & (diff <= days)

    if not ref_mask.any():
        return pd.Series(False, index=df.index)

    pid_codes, _ = pd.factorize(pid, use_na_sentinel=False)
    day = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    child_np = child_mask.to_numpy()
    ref_np = ref_mask.to_numpy()

    bands = window_bands(direction, min_days, days)
    lhs_match = band_window_match(
        pid_codes, day, child_np, ref_np, bands,
        target_shift=ref_offset_days, exclude_self=exclude_self,
    )

    if not (every_left or every_right):
        return pd.Series(lhs_match, index=df.index)

    # ----- Universal mode: determine which persons satisfy the predicate -----
    # Persons must have both child and ref events present (non-empty rule).
    matching_codes = np.intersect1d(
        np.unique(pid_codes[child_np]), np.unique(pid_codes[ref_np])
    )

    if every_left:
        # Every child event must have a qualifying ref.
        failed = np.unique(pid_codes[child_np & ~lhs_match])
        matching_codes = np.setdiff1d(matching_codes, failed)

    if every_right:
        # Every ref event must have a qualifying child: same band test
        # with roles swapped and bands mirrored.
        rhs_match = band_window_match(
            pid_codes, day, ref_np, child_np, mirror_bands(bands),
            query_shift=ref_offset_days, exclude_self=exclude_self,
        )
        failed = np.unique(pid_codes[ref_np & ~rhs_match])
        matching_codes = np.setdiff1d(matching_codes, failed)

    # Return ALL child rows for matching persons
    person_ok = np.isin(pid_codes, matching_codes)
    return child_mask & pd.Series(person_ok, index=df.index)


def eval_inside_outside(
    df: pd.DataFrame,
    child_mask: pd.Series,
    ref_mask: pd.Series,
    inside: bool,
    min_events: int,
    max_events: int,
    direction: str,
    pid_col: str,
    exclude_self: bool = True,
) -> pd.Series:
    """Row-level: mark child rows inside/outside an event-count window.

    The window is the closed integer range ``[min_events, max_events]``
    in row offsets from each ref row. For example,
    ``inside 1 to 5 events after Y`` ⇒ min_events=1, max_events=5,
    direction="after"; `direction` of "before" mirrors the offsets
    (offset −k is |k| rows earlier); `around` uses signed offsets
    directly (so `inside -3 to 5 events around Y` is positions −3..+5).

    `exclude_self` (default): a row never matches a window anchored at
    itself. Event positions are unique per row, so offset 0 IS the
    anchor row — without exclusion, any window containing offset 0
    (`around` ranges spanning zero, explicit `0 to N`) trivially
    matches self-referential patterns like `X inside -3 to 5 events
    around X`. Anchored event-window aggregates pass False so the
    anchor row stays inside its own window.

    `inside=False` flips to the row-level complement (of the
    self-excluded window).
    """
    if not child_mask.any() or not ref_mask.any():
        return pd.Series(False, index=df.index)

    pid = df[pid_col]
    result = pd.Series(False, index=df.index)

    # Compute event number per person
    event_num = pid.groupby(pid).cumcount()

    # For each person, find reference event positions
    ref_positions = event_num[ref_mask]
    ref_pids = pid[ref_mask]

    # For each person, check if child events fall in the window
    for p in ref_pids.unique():
        p_mask = pid == p
        p_event_num = event_num[p_mask]
        p_child = child_mask[p_mask]
        p_ref_positions = ref_positions[ref_pids == p].values

        in_window = pd.Series(False, index=p_event_num.index)
        for ref_pos in p_ref_positions:
            if direction == "after":
                # offsets +min_events..+max_events relative to ref
                lo = ref_pos + min_events
                hi = ref_pos + max_events
            elif direction == "before":
                # offsets −max_events..−min_events (i.e., positions
                # ref-max..ref-min going backwards)
                lo = ref_pos - max_events
                hi = ref_pos - min_events
            else:  # around: signed offsets
                lo = ref_pos + min_events
                hi = ref_pos + max_events
            window = (p_event_num >= lo) & (p_event_num <= hi)
            if exclude_self:
                # Positions are unique per person: != ref_pos removes
                # exactly the anchor row itself.
                window = window & (p_event_num != ref_pos)
            in_window = in_window | window

        if inside:
            matched = (p_child & in_window).values
        else:
            matched = (p_child & ~in_window).values
        result.loc[p_mask.values] = matched

    return result
