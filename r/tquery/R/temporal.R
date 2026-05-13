# R port of tquery/_temporal.py — temporal evaluators.
# Operates on logical masks aligned to env$nrow.


# --- before / after / simultaneously -----------------------------------

eval_before_after <- function(env, left_mask, right_mask, op,
                              every_left = FALSE, every_right = FALSE,
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
               else                             l_min < r_min]
  } else {  # after
    agg[, ok := if (every_left && every_right) l_min > r_max
               else if (every_right)           l_max > r_max
               else                             l_min > r_min]
  }
  matching_pids <- agg$pid[agg$ok]
  left_mask & (pid %in% matching_pids)
}

# --- within days -------------------------------------------------------
# Lift of r-spike/within_days.R with three additions:
#   * ref_mask == NULL → window relative to each person's first event.
#   * outside flag → row-level complement restricted to evaluable persons.
#   * universal modes (every_left / every_right) → "ALL child rows for
#     persons where every child row (or every ref row) has a counterpart".

eval_within_days <- function(env, child_mask, ref_mask, days,
                             min_days = 0L, direction = NULL,
                             every_left = FALSE, every_right = FALSE,
                             ref_offset_days = 0L, outside = FALSE) {
  if (!any(child_mask)) return(rep(FALSE, env$nrow))

  pid   <- env$pid
  dates <- env$date

  # ref_mask = NULL: distance from each person's first event
  if (is.null(ref_mask)) {
    dt_local <- data.table(pid = pid, date = dates, row = seq_len(env$nrow))
    dt_local[, first := min(date), by = pid]
    diff <- as.integer(abs(dt_local$date - dt_local$first))
    return(child_mask & (diff >= min_days) & (diff <= days))
  }
  if (!any(ref_mask)) return(rep(FALSE, env$nrow))

  result <- .eval_within_days_core(
    env, child_mask, ref_mask, days, min_days, direction,
    every_left, every_right, ref_offset_days
  )
  if (outside) return(.row_complement_evaluable(env, child_mask, ref_mask, result))
  result
}

.eval_within_days_core <- function(env, child_mask, ref_mask, days, min_days, direction,
                                   every_left, every_right, ref_offset_days) {
  pid   <- env$pid
  dates <- env$date

  child_dt <- data.table(
    pid      = pid[child_mask],
    date     = dates[child_mask],
    orig_idx = which(child_mask)
  )
  ref_dt <- data.table(
    pid      = pid[ref_mask],
    ref_date = dates[ref_mask] + ref_offset_days
  )
  setkey(child_dt, pid, date)
  setkey(ref_dt, pid, ref_date)

  signed <- !is.null(direction) && direction == "around" && min_days < 0L

  roll_join <- function(roll_val, ends) {
    ref_dt[child_dt,
      .(orig_idx = i.orig_idx, child_date = i.date, ref_date = x.ref_date),
      on = .(pid, ref_date = date), roll = roll_val, rollends = ends
    ]
  }

  # LHS-anchored: for each child, find a ref in the window.
  if (is.null(direction) || direction == "around") {
    bw_roll <- days
    fw_roll <- if (signed) -abs(min_days) else -days
    bw <- roll_join(bw_roll, c(FALSE, TRUE))
    fw <- roll_join(fw_roll, c(TRUE, FALSE))
    matched <- rbind(bw, fw)
  } else if (direction == "after") {
    matched <- roll_join(days, c(FALSE, TRUE))
  } else if (direction == "before") {
    matched <- roll_join(-days, c(TRUE, FALSE))
  } else {
    stop("Unknown direction: ", direction)
  }
  matched <- matched[!is.na(ref_date)]
  matched[, delta := as.integer(child_date - ref_date)]
  if (signed) {
    matched <- matched[delta >= min_days & delta <= days]
  } else {
    matched <- matched[abs(delta) >= min_days & abs(delta) <= days]
  }

  if (!every_left && !every_right) {
    out <- rep(FALSE, env$nrow)
    out[unique(matched$orig_idx)] <- TRUE
    return(out)
  }

  # Universal modes: rebuild per-person totals.
  # every_left: every child row in person must have a qualifying ref.
  # every_right: every ref row in person must have a qualifying child.
  child_pids_unique <- unique(pid[child_mask])
  ref_pids_unique   <- unique(pid[ref_mask])
  candidate <- intersect(as.character(child_pids_unique), as.character(ref_pids_unique))
  matching_pids <- candidate

  if (every_left) {
    # number of matched child rows per pid vs. total child rows
    child_total <- tapply(seq_along(pid[child_mask]), as.character(pid[child_mask]), length)
    if (nrow(matched) > 0L) {
      matched_pid <- pid[matched$orig_idx]
      hit <- tapply(matched$orig_idx, as.character(matched_pid), function(idx) length(unique(idx)))
    } else hit <- stats::setNames(integer(), character())
    full <- names(child_total)[
      vapply(names(child_total),
             function(p) !is.na(hit[p]) && hit[p] == child_total[[p]],
             logical(1))
    ]
    matching_pids <- intersect(matching_pids, full)
  }

  if (every_right) {
    # Run the join from the RHS side, looking the opposite direction.
    rhs <- .eval_every_right(env, child_mask, ref_mask, days, min_days, direction, ref_offset_days)
    matching_pids <- intersect(matching_pids, rhs)
  }

  child_mask & (as.character(pid) %in% matching_pids)
}

# every_right: every ref row has a qualifying child within the window.
# Mirrors the rhs_merged block in Python _temporal.py.
.eval_every_right <- function(env, child_mask, ref_mask, days, min_days, direction, ref_offset_days) {
  pid   <- env$pid
  dates <- env$date

  ref_lookup <- data.table(
    pid      = pid[ref_mask],
    ref_date = dates[ref_mask] + ref_offset_days,
    orig_idx = which(ref_mask)
  )
  child_lookup <- data.table(
    pid        = pid[child_mask],
    child_date = dates[child_mask]
  )
  setkey(ref_lookup, pid, ref_date)
  setkey(child_lookup, pid, child_date)

  # Opposite direction from LHS: if child is after ref (direction "after"),
  # then from ref we look FORWARD for a child.
  if (is.null(direction) || direction == "around") {
    bw <- child_lookup[ref_lookup, .(orig_idx = i.orig_idx, ref_date = i.ref_date, child_date = x.child_date),
                       on = .(pid, child_date = ref_date), roll = days, rollends = c(FALSE, TRUE)]
    fw <- child_lookup[ref_lookup, .(orig_idx = i.orig_idx, ref_date = i.ref_date, child_date = x.child_date),
                       on = .(pid, child_date = ref_date), roll = -days, rollends = c(TRUE, FALSE)]
    matched <- rbind(bw, fw)
  } else if (direction == "after") {
    matched <- child_lookup[ref_lookup, .(orig_idx = i.orig_idx, ref_date = i.ref_date, child_date = x.child_date),
                            on = .(pid, child_date = ref_date), roll = -days, rollends = c(TRUE, FALSE)]
  } else if (direction == "before") {
    matched <- child_lookup[ref_lookup, .(orig_idx = i.orig_idx, ref_date = i.ref_date, child_date = x.child_date),
                            on = .(pid, child_date = ref_date), roll = days, rollends = c(FALSE, TRUE)]
  }
  matched <- matched[!is.na(child_date)]
  matched[, delta := as.integer(abs(child_date - ref_date))]
  matched <- matched[delta >= min_days & delta <= days]

  total <- tapply(seq_along(pid[ref_mask]), as.character(pid[ref_mask]), length)
  hit   <- if (nrow(matched) > 0L) tapply(matched$orig_idx, as.character(pid[matched$orig_idx]),
                                          function(idx) length(unique(idx)))
           else stats::setNames(integer(), character())
  names(total)[vapply(names(total),
                      function(p) !is.na(hit[p]) && hit[p] == total[[p]],
                      logical(1))]
}

# --- inside / outside (event-position window) --------------------------

eval_inside_outside <- function(env, child_mask, ref_mask, inside,
                                min_events, max_events, direction) {
  if (!any(child_mask) || !any(ref_mask)) return(rep(FALSE, env$nrow))

  pid <- env$pid
  result <- rep(FALSE, env$nrow)

  # Event number per person (0-based, by appearance order — assumes sorted)
  dt_local <- data.table(idx = seq_len(env$nrow), pid = pid)
  dt_local[, event_num := seq_len(.N) - 1L, by = pid]
  event_num <- dt_local$event_num

  ref_positions <- event_num[ref_mask]
  ref_pids      <- pid[ref_mask]

  for (p in unique(ref_pids)) {
    p_rows         <- which(pid == p)
    p_event_num    <- event_num[p_rows]
    p_child        <- child_mask[p_rows]
    p_ref_positions <- ref_positions[ref_pids == p]

    in_window <- rep(FALSE, length(p_rows))
    for (rp in p_ref_positions) {
      if (direction == "after") {
        lo <- rp + min_events; hi <- rp + max_events
      } else if (direction == "before") {
        lo <- rp - max_events; hi <- rp - min_events
      } else {  # around: signed offsets
        lo <- rp + min_events; hi <- rp + max_events
      }
      in_window <- in_window | (p_event_num >= lo & p_event_num <= hi)
    }
    result[p_rows] <- if (inside) (p_child & in_window) else (p_child & !in_window)
  }
  result
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
