# R port of tquery/_temporal.py — temporal evaluators.
# Operates on logical masks aligned to env$nrow.


# --- before / after / simultaneously -----------------------------------

eval_before_after <- function(env, left_mask, right_mask, op,
                              every_left = FALSE, every_right = FALSE,
                              any_left = FALSE, any_right = FALSE,
                              left_offset_days = 0L, right_offset_days = 0L) {
  if (!any(left_mask) || !any(right_mask)) return(rep(FALSE, env$nrow))

  pid   <- env$pid
  dates <- env$date

  if (op == "simultaneously") {
    left_dates  <- split(dates[left_mask]  + left_offset_days,  pid[left_mask])
    right_dates <- split(dates[right_mask] + right_offset_days, pid[right_mask])
    common <- intersect(names(left_dates), names(right_dates))
    matching_pids <- character()
    for (p in common) {
      l <- left_dates[[p]]; r <- right_dates[[p]]
      if (every_left && !all(l %in% r)) next
      if (every_right && !all(r %in% l)) next
      if (!every_left && !every_right && !any(l %in% r)) next
      matching_pids <- c(matching_pids, p)
    }
    # Coerce back to pid type (split keys are character)
    pid_str <- as.character(pid)
    return(left_mask & (pid_str %in% matching_pids))
  }

  # before / after — per-person aggregates via data.table groupby
  ldt <- data.table(pid = pid[left_mask],  date = dates[left_mask]  + left_offset_days)
  rdt <- data.table(pid = pid[right_mask], date = dates[right_mask] + right_offset_days)
  lagg <- ldt[, .(l_min = min(date), l_max = max(date)), by = pid]
  ragg <- rdt[, .(r_min = min(date), r_max = max(date)), by = pid]
  agg <- merge(lagg, ragg, by = "pid")
  if (nrow(agg) == 0L) return(rep(FALSE, env$nrow))

  if (op == "before") {
    agg[, ok := if (every_left && every_right) l_max < r_min
               else if (every_left)            l_max < r_max
               else if (every_right)           l_min < r_min
               else if (any_right)             l_min < r_max  # existential
               else                             l_min < r_min]
  } else {  # after
    agg[, ok := if (every_left && every_right) l_min > r_max
               else if (every_right)           l_max > r_max
               else if (every_left)            l_min > r_min
               else if (any_left)              l_max > r_min  # existential
               else                             l_min > r_min]
  }
  matching_pids <- agg$pid[agg$ok]
  left_mask & (pid %in% matching_pids)
}

# --- within days -------------------------------------------------------
# Band-existence evaluation (mirrors tquery/_temporal.py):
#   * The window spec is translated into closed day-offset bands; a child
#     row matches iff ANY ref row of the person falls in a band. Using
#     non-equi joins (not roll/nearest joins) means lower-bounded windows
#     (`inside 30 to 90 days after Y`) cannot miss a qualifying ref just
#     because a nearer, non-qualifying ref exists.
#   * exclude_self (default TRUE): a row never serves as its own
#     reference — `X inside 0 to 5 days after X` means "another X row",
#     so a lone X does not match itself but a second same-date X does.
#     Anchored aggregates pass FALSE (anchor stays in its own window).
#   * ref_mask == NULL → window relative to each person's first event.
#   * outside flag → row-level complement restricted to evaluable persons.
#   * universal modes (every_left / every_right) → "ALL child rows for
#     persons where every child row (or every ref row) has a counterpart".

# Closed day-offset bands: a ref at day t qualifies for a child at day d
# iff t ∈ [d + a, d + b] for some band c(a, b).
.window_bands <- function(direction, min_days, days) {
  if (!is.null(direction) && direction == "after") {
    return(list(c(-days, -min_days)))
  }
  if (!is.null(direction) && direction == "before") {
    return(list(c(min_days, days)))
  }
  # around (signed / unsigned) or no direction
  if (min_days < 0L) return(list(c(-days, -min_days)))
  if (min_days == 0L) return(list(c(-days, days)))
  list(c(-days, -min_days), c(min_days, days))
}

# Swap query/target roles: if target ∈ [q + a, q + b] then query ∈ [t − b, t − a].
.mirror_bands <- function(bands) lapply(bands, function(b) c(-b[2L], -b[1L]))

# Existence test: TRUE at query rows having ≥ 1 target row (of the same
# person) in one of the bands — excluding, if requested, the row itself.
# `values` is the axis: per-row dates by default, or event positions for
# event-count windows (any numeric works — the joins are generic).
.band_match <- function(env, query_mask, target_mask, bands,
                        query_shift = 0L, target_shift = 0L,
                        exclude_self = TRUE, values = NULL) {
  if (is.null(values)) values <- env$date
  out <- rep(FALSE, env$nrow)
  q_idx <- which(query_mask)
  t_idx <- which(target_mask)
  if (length(q_idx) == 0L || length(t_idx) == 0L) return(out)

  qdt <- data.table(
    pid    = env$pid[q_idx],
    q_date = values[q_idx] + query_shift,
    q_rid  = q_idx
  )
  tdt <- data.table(
    pid    = env$pid[t_idx],
    t_date = values[t_idx] + target_shift,
    t_rid  = t_idx
  )

  matched <- rep(FALSE, length(q_idx))
  for (band in bands) {
    qdt[, `:=`(lo = q_date + band[1L], hi = q_date + band[2L])]
    hits <- tdt[qdt,
      .(n = sum(!is.na(x.t_rid) & (!exclude_self | x.t_rid != i.q_rid))),
      by = .EACHI,
      on = .(pid, t_date >= lo, t_date <= hi)
    ]
    matched <- matched | (hits$n > 0L)
  }
  out[q_idx] <- matched
  out
}

eval_within_days <- function(env, child_mask, ref_mask, days,
                             min_days = 0L, direction = NULL,
                             every_left = FALSE, every_right = FALSE,
                             ref_offset_days = 0L, outside = FALSE,
                             exclude_self = TRUE) {
  if (!any(child_mask)) return(rep(FALSE, env$nrow))

  # ref_mask = NULL: distance from each person's first event (a
  # per-person date, not a row — no self-exclusion).
  if (is.null(ref_mask)) {
    dt_local <- data.table(pid = env$pid, date = env$date, row = seq_len(env$nrow))
    dt_local[, first := min(date), by = pid]
    diff <- as.integer(abs(dt_local$date - dt_local$first))
    return(child_mask & (diff >= min_days) & (diff <= days))
  }
  if (!any(ref_mask)) return(rep(FALSE, env$nrow))

  result <- .eval_within_days_core(
    env, child_mask, ref_mask, days, min_days, direction,
    every_left, every_right, ref_offset_days, exclude_self
  )
  if (outside) return(.row_complement_evaluable(env, child_mask, ref_mask, result))
  result
}

.eval_within_days_core <- function(env, child_mask, ref_mask, days, min_days, direction,
                                   every_left, every_right, ref_offset_days,
                                   exclude_self = TRUE) {
  bands <- .window_bands(direction, min_days, days)
  lhs <- .band_match(env, child_mask, ref_mask, bands,
                     query_shift = 0L, target_shift = ref_offset_days,
                     exclude_self = exclude_self)

  if (!every_left && !every_right) return(lhs)

  # Universal modes: persons need child AND ref events (non-empty rule).
  pid_chr <- as.character(env$pid)
  matching_pids <- intersect(unique(pid_chr[child_mask]), unique(pid_chr[ref_mask]))

  if (every_left) {
    # Every child row must have a qualifying ref.
    bad <- unique(pid_chr[child_mask & !lhs])
    matching_pids <- setdiff(matching_pids, bad)
  }

  if (every_right) {
    # Every ref row must have a qualifying child: same band test with
    # roles swapped and bands mirrored.
    rhs <- .band_match(env, ref_mask, child_mask, .mirror_bands(bands),
                       query_shift = ref_offset_days, target_shift = 0L,
                       exclude_self = exclude_self)
    bad <- unique(pid_chr[ref_mask & !rhs])
    matching_pids <- setdiff(matching_pids, bad)
  }

  child_mask & (pid_chr %in% matching_pids)
}

# --- inside / outside (event-position window) --------------------------

# exclude_self (default TRUE): a row never matches a window anchored at
# itself — event positions are unique per row, so offset 0 IS the anchor
# row. Anchored event-window aggregates pass FALSE (the anchor row stays
# inside its own window).
eval_inside_outside <- function(env, child_mask, ref_mask, inside,
                                min_events, max_events, direction,
                                exclude_self = TRUE) {
  if (!any(child_mask) || !any(ref_mask)) return(rep(FALSE, env$nrow))

  # Event positions per person (0-based, input order — assumes sorted),
  # used as the axis of the generic band match: a child at position p is
  # in the window of a ref at q iff p ∈ [q+min, q+max] (after/around) or
  # p ∈ [q−max, q−min] (before) — i.e. the ref lies in a band relative
  # to the child. Self-exclusion falls out via row ids as for dates.
  dt_local <- data.table(pid = env$pid)
  dt_local[, pos := seq_len(.N) - 1L, by = pid]
  positions <- dt_local$pos

  bands <- if (direction == "before") list(c(min_events, max_events))
           else list(c(-max_events, -min_events))

  match <- .band_match(env, child_mask, ref_mask, bands,
                       exclude_self = exclude_self, values = positions)
  if (inside) return(match)
  has_ref <- env$pid %in% unique(env$pid[ref_mask])
  child_mask & !match & has_ref
}

# --- helpers -----------------------------------------------------------

# Row-level complement of a positive mask, restricted to evaluable persons.
# A person is evaluable iff they have ≥1 child row AND (when ref_mask given) ≥1 ref row.
.row_complement_evaluable <- function(env, child_mask, ref_mask, positive_mask) {
  pid <- env$pid
  child_pids <- unique(as.character(pid[child_mask]))
  evaluable <- if (is.null(ref_mask)) child_pids
               else intersect(child_pids, as.character(unique(pid[ref_mask])))
  child_mask & (as.character(pid) %in% evaluable) & !positive_mask
}
