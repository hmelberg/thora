# R port of tquery/_evaluator.py — AST dispatch + prefix helpers.


# Evaluation environment. Holds the data + config + (later) cache.
make_eval_env <- function(dt, pid_col, date_col, cols = NULL,
                          sep = NULL, variables = list()) {
  stopifnot(pid_col %in% names(dt), date_col %in% names(dt))
  if (is.null(cols)) {
    cols <- setdiff(names(dt), c(pid_col, date_col))
    cols <- cols[vapply(dt[, ..cols], is.character, logical(1))]
  } else if (is.character(cols) && length(cols) == 1L) {
    cols <- c(cols)
  }
  env <- new.env(parent = emptyenv())
  env$dt        <- dt
  env$pid       <- dt[[pid_col]]
  env$date      <- dt[[date_col]]
  env$pid_col   <- pid_col
  env$date_col  <- date_col
  env$cols      <- cols
  env$sep       <- sep
  env$variables <- variables
  env$nrow      <- nrow(dt)
  env$all_codes <- collect_unique_codes(dt, cols, sep)
  env
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

tq_eval <- function(node, env) UseMethod("tq_eval")

tq_eval.default <- function(node, env) {
  stop(sprintf("Unhandled AST node: %s", paste(class(node), collapse = ",")))
}

# --- atoms -------------------------------------------------------------

tq_eval.CodeAtom <- function(node, env) {
  cols <- if (is.null(node$columns)) env$cols
          else resolve_columns(node$columns, names(env$dt))
  codes <- expand_all_codes(node$codes, env$all_codes, env$variables)
  get_matching_rows(env$dt, codes, cols, env$sep)
}

tq_eval.EventAtom <- function(node, env) rep(TRUE, env$nrow)

tq_eval.ComparisonAtom <- function(node, env) {
  if (!node$column %in% names(env$dt)) {
    stop(sprintf("Column '%s' not in data", node$column))
  }
  v <- env$dt[[node$column]]
  switch(node$op,
    ">"  = !is.na(v) & v >  node$value,
    "<"  = !is.na(v) & v <  node$value,
    ">=" = !is.na(v) & v >= node$value,
    "<=" = !is.na(v) & v <= node$value,
    "==" = !is.na(v) & v == node$value,
    "!=" = !is.na(v) & v != node$value
  )
}

# --- logical & negation ------------------------------------------------

tq_eval.NotExpr <- function(node, env) {
  child <- tq_eval(node$child, env)
  pid <- env$pid
  # Persons matching the child (anywhere)
  match_persons <- unique(as.character(pid[child]))
  all_persons   <- unique(as.character(pid))
  not_pids      <- setdiff(all_persons, match_persons)
  as.character(pid) %in% not_pids
}

tq_eval.BinaryLogical <- function(node, env) {
  l <- tq_eval(node$left,  env)
  r <- tq_eval(node$right, env)
  pid <- as.character(env$pid)
  l_pids <- unique(pid[l])
  r_pids <- unique(pid[r])
  matching <- if (node$op == "and") intersect(l_pids, r_pids) else union(l_pids, r_pids)
  (pid %in% matching) & (l | r)
}

# --- prefix / quantifiers ----------------------------------------------

tq_eval.PrefixExpr <- function(node, env) {
  child <- tq_eval(node$child, env)
  pid   <- as.character(env$pid)
  switch(node$kind,
    min      = ,
    max      = ,
    exactly  = .eval_count_prefix(node$kind, node$n, child, pid),
    ordinal  = .eval_ordinal(node$n, child, pid),
    first    = .eval_first_n(node$n, child, pid),
    last     = .eval_last_n(node$n, child, pid),
    stop(sprintf("Unknown prefix kind: %s", node$kind))
  )
}

tq_eval.RangePrefixExpr <- function(node, env) {
  child <- tq_eval(node$child, env)
  pid   <- as.character(env$pid)
  counts <- .by_transform_sum(child, pid)
  child & counts >= node$min_n & counts <= node$max_n
}

.by_transform_sum <- function(mask, pid) {
  # Per-pid total of TRUEs in `mask`, broadcast back to row level.
  # data.table groupby is dramatically faster than tapply()+match() on
  # character pids (10×+ on 50k persons).
  dt <- data.table(m = as.integer(mask), pid = pid)
  dt[, total := sum(m), by = pid]
  dt$total
}

.by_cumsum_in_group <- function(mask, pid) {
  # Cumulative count of TRUEs in `mask` within each pid group, preserving
  # input order. Returns integer vector aligned to `mask`.
  dt <- data.table(idx = seq_along(mask), pid = pid, m = as.integer(mask))
  dt[, cs := cumsum(m), by = pid]
  dt$cs
}

.eval_count_prefix <- function(kind, n, mask, pid) {
  counts <- .by_transform_sum(mask, pid)
  ok <- switch(kind, min = counts >= n, max = counts <= n, exactly = counts == n)
  mask & ok
}

.eval_ordinal <- function(n, mask, pid) {
  cs <- .by_cumsum_in_group(mask, pid)
  if (n > 0L) return(mask & cs == n)
  total <- .by_transform_sum(mask, pid)
  reverse_pos <- total - cs + 1L
  mask & reverse_pos == abs(n)
}

.eval_first_n <- function(n, mask, pid) {
  cs <- .by_cumsum_in_group(mask, pid)
  mask & cs <= n
}

.eval_last_n <- function(n, mask, pid) {
  cs <- .by_cumsum_in_group(mask, pid)
  total <- .by_transform_sum(mask, pid)
  mask & cs > (total - n)
}

# --- temporal ----------------------------------------------------------

# Strip optional Quantifier wrapper. Returns list(inner, every).
.unwrap_quantifier <- function(node) {
  if (inherits(node, "Quantifier")) {
    return(list(inner = node$child, every = node$kind == "every"))
  }
  list(inner = node, every = FALSE)
}

# Strip ShiftExpr chain. Returns list(inner, offset).
.unwrap_shift <- function(node) {
  offset <- 0L
  while (inherits(node, "ShiftExpr")) {
    offset <- offset + node$offset_days
    node <- node$child
  }
  list(inner = node, offset = offset)
}

tq_eval.TemporalExpr <- function(node, env) {
  L  <- .unwrap_quantifier(node$left)
  if (inherits(L$inner, "ShiftExpr")) {
    stop("Shifted anchors are only valid on the reference side of before/after/simultaneously")
  }
  R  <- .unwrap_quantifier(node$right)
  rh <- .unwrap_shift(R$inner)
  R$inner  <- rh$inner
  R_offset <- rh$offset

  left_mask  <- tq_eval(L$inner, env)
  right_mask <- tq_eval(R$inner, env)
  eval_before_after(env, left_mask, right_mask, node$op,
                    every_left  = L$every, every_right = R$every,
                    left_offset_days = 0L, right_offset_days = R_offset)
}

tq_eval.WithinExpr <- function(node, env) {
  # v0.2: dispatch to aggregate handling when the child is an AggregateExpr.
  # Same surface, different semantics (sliding vs anchored) — see spec.
  if (inherits(node$child, "AggregateExpr")) {
    return(.eval_within_aggregate(node, env))
  }
  L  <- .unwrap_quantifier(node$child)
  child_mask <- tq_eval(L$inner, env)
  ref_mask <- NULL; R_offset <- 0L; R_every <- FALSE
  if (!is.null(node$ref)) {
    R  <- .unwrap_quantifier(node$ref)
    rh <- .unwrap_shift(R$inner)
    ref_mask <- tq_eval(rh$inner, env)
    R_offset <- rh$offset
    R_every  <- R$every
  }
  eval_within_days(env, child_mask, ref_mask, node$days,
                   min_days = node$min_days, direction = node$direction,
                   every_left = L$every, every_right = R_every,
                   ref_offset_days = R_offset, outside = isTRUE(node$outside))
}

# ---- Aggregate evaluation (v0.2) -----------------------------------------

# v0.2.3: route to *_pct variants when AggregateExpr$relative is TRUE.
.agg_fn_key <- function(node) {
  if (isTRUE(node$relative) && node$func %in% c("rise", "fall")) {
    paste0(node$func, "_pct")
  } else {
    node$func
  }
}

.AGG_FNS <- list(
  sum    = function(x) sum(x, na.rm = TRUE),
  mean   = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x) },
  avg    = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x) },
  min    = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x) },
  max    = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x) },
  median = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else stats::median(x) },
  sd     = function(x) { x <- x[!is.na(x)]; if (length(x) <  2L) NA_real_ else stats::sd(x) },
  var    = function(x) { x <- x[!is.na(x)]; if (length(x) <  2L) NA_real_ else stats::var(x) },
  count  = function(x) sum(!is.na(x)),
  n      = function(x) sum(!is.na(x)),
  range  = function(x) { x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x) - min(x) },
  # v0.2.2: max drawup (rise) and max drawdown magnitude (fall).
  rise   = function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) return(NA_real_)
    if (length(x) == 1L) return(0)
    max(x - cummin(x))
  },
  fall   = function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) return(NA_real_)
    if (length(x) == 1L) return(0)
    max(cummax(x) - x)
  },
  # v0.2.3: relative variants — divide by cummin/cummax, skip pairs
  # where the denominator is non-positive.
  rise_pct = function(x) {
    x <- x[!is.na(x)]
    if (length(x) <= 1L) return(if (length(x) == 1L) 0 else NA_real_)
    cm <- cummin(x)
    safe <- cm > 0
    if (!any(safe)) return(0)
    ratio <- ifelse(safe, (x - cm) / ifelse(safe, cm, 1), 0)
    max(ratio)
  },
  fall_pct = function(x) {
    x <- x[!is.na(x)]
    if (length(x) <= 1L) return(if (length(x) == 1L) 0 else NA_real_)
    cm <- cummax(x)
    safe <- cm > 0
    if (!any(safe)) return(0)
    ratio <- ifelse(safe, (cm - x) / ifelse(safe, cm, 1), 0)
    max(ratio)
  }
)

.OP_FNS <- list(
  ">"  = function(a, b) !is.na(a) & a >  b,
  "<"  = function(a, b) !is.na(a) & a <  b,
  ">=" = function(a, b) !is.na(a) & a >= b,
  "<=" = function(a, b) !is.na(a) & a <= b,
  "==" = function(a, b) !is.na(a) & a == b,
  "!=" = function(a, b) !is.na(a) & a != b
)

tq_eval.AggregateExpr <- function(node, env) {
  .eval_aggregate(node, env, row_mask = NULL)
}

.eval_aggregate <- function(node, env, row_mask = NULL) {
  if (!node$column %in% names(env$dt)) {
    stop(sprintf("Column '%s' not in data", node$column))
  }
  col <- env$dt[[node$column]]
  pid <- env$pid
  if (!is.null(row_mask)) {
    col_sub <- col[row_mask]
    pid_sub <- pid[row_mask]
  } else {
    col_sub <- col
    pid_sub <- pid
  }
  agg_fn <- .AGG_FNS[[.agg_fn_key(node)]]
  op_fn  <- .OP_FNS[[node$op]]

  agg_dt <- data.table::data.table(p = pid_sub, v = col_sub)
  per_pid <- agg_dt[, .(agg = agg_fn(v)), by = p]

  matching_pids <- per_pid$p[op_fn(per_pid$agg, node$value)]
  pid %in% matching_pids
}

.eval_within_aggregate <- function(node, env) {
  agg_node <- node$child
  sliding  <- is.null(node$direction) && is.null(node$ref)

  if (sliding) {
    if (isTRUE(node$outside)) {
      stop("`outside` over a sliding aggregate is not supported in v0.2")
    }
    return(.eval_aggregate_sliding(agg_node, node$days, env))
  }

  # Anchored: row mask of rows within window of any ref event.
  R  <- .unwrap_quantifier(node$ref)
  rh <- .unwrap_shift(R$inner)
  ref_mask <- tq_eval(rh$inner, env)
  all_rows <- rep(TRUE, env$nrow)
  in_window <- eval_within_days(
    env, all_rows, ref_mask, node$days,
    min_days = node$min_days, direction = node$direction,
    ref_offset_days = rh$offset
  )
  if (isTRUE(node$outside)) {
    evaluable_pids <- unique(env$pid[ref_mask])
    out_window <- (env$pid %in% evaluable_pids) & !in_window
    return(.eval_aggregate(agg_node, env, row_mask = out_window))
  }
  .eval_aggregate(agg_node, env, row_mask = in_window)
}

.eval_aggregate_sliding <- function(node, days, env) {
  if (!node$column %in% names(env$dt)) {
    stop(sprintf("Column '%s' not in data", node$column))
  }
  pid   <- env$pid
  date  <- env$date
  vals  <- env$dt[[node$column]]
  agg_fn <- .AGG_FNS[[.agg_fn_key(node)]]
  op_fn  <- .OP_FNS[[node$op]]

  # For each row r, compute the aggregate over rows in same pid where
  # date in [date[r] - days, date[r]]. Implemented as a non-equi self-
  # join restricted to same-pid same-or-earlier-date within `days`.
  src <- data.table::data.table(
    pid   = pid,
    date  = date,
    v     = vals,
    rowid = seq_len(env$nrow)
  )
  data.table::setkey(src, pid, date)

  # For each row, find the window's start date.
  src[, win_start := date - as.integer(days)]
  # Self-join: every row from src joins to every row of same pid whose
  # date is within [win_start, date].
  joined <- src[src,
    on = .(pid, date >= win_start, date <= date),
    allow.cartesian = TRUE,
    .(rowid = i.rowid, v_in = x.v, d_in = x.date)
  ]
  # rise/fall (v0.2.2) need chronological order within each window.
  data.table::setorder(joined, rowid, d_in)
  per_row <- joined[, .(agg = agg_fn(v_in)), by = rowid]
  # Map back per row -> per pid: person matches if ANY row's window agg satisfies threshold.
  per_row[, match := op_fn(agg, node$value)]
  pid_of_row <- pid[per_row$rowid]
  matching_pids <- unique(pid_of_row[per_row$match])
  pid %in% matching_pids
}

tq_eval.InsideExpr <- function(node, env) {
  # v0.2.1: aggregate child → event-window aggregate (sliding or anchored).
  if (inherits(node$child, "AggregateExpr")) {
    return(.eval_inside_aggregate(node, env))
  }
  child_mask <- tq_eval(node$child, env)
  ref_mask   <- tq_eval(node$ref,   env)
  eval_inside_outside(env, child_mask, ref_mask,
                      inside = node$inside,
                      min_events = node$min_events,
                      max_events = node$max_events,
                      direction = node$direction)
}

.eval_inside_aggregate <- function(node, env) {
  agg_node <- node$child
  sliding  <- is.null(node$direction) && is.null(node$ref)

  if (sliding) {
    return(.eval_aggregate_sliding_events(agg_node, node$max_events, env))
  }
  # Anchored: build a row mask of rows in the event-window of any ref, then agg.
  ref_mask <- tq_eval(node$ref, env)
  all_rows <- rep(TRUE, env$nrow)
  in_window <- eval_inside_outside(env, all_rows, ref_mask,
                                   inside = TRUE,
                                   min_events = node$min_events,
                                   max_events = node$max_events,
                                   direction = node$direction)
  if (!isTRUE(node$inside)) {
    evaluable_pids <- unique(env$pid[ref_mask])
    in_window <- (env$pid %in% evaluable_pids) & !in_window
  }
  .eval_aggregate(agg_node, env, row_mask = in_window)
}

.eval_aggregate_sliding_events <- function(node, window_size, env) {
  if (!node$column %in% names(env$dt)) {
    stop(sprintf("Column '%s' not in data", node$column))
  }
  agg_fn <- .AGG_FNS[[.agg_fn_key(node)]]
  op_fn  <- .OP_FNS[[node$op]]

  pid   <- env$pid
  date  <- env$date
  vals  <- env$dt[[node$column]]

  src <- data.table::data.table(
    pid   = pid,
    date  = date,
    v     = vals,
    rowid = seq_len(env$nrow)
  )
  data.table::setkey(src, pid, date)
  # Per-pid event index (0-based)
  src[, pos := seq_len(.N) - 1L, by = pid]
  src[, lo_pos := pmax(0L, pos - (window_size - 1L))]

  # Self-join: each row joins to same-pid rows whose pos is in [lo_pos, pos].
  joined <- src[src,
    on = .(pid, pos >= lo_pos, pos <= pos),
    allow.cartesian = TRUE,
    .(rowid = i.rowid, v_in = x.v, p_in = x.pos)
  ]
  data.table::setorder(joined, rowid, p_in)
  per_row <- joined[, .(agg = agg_fn(v_in)), by = rowid]
  per_row[, match := op_fn(agg, node$value)]
  pid_of_row <- pid[per_row$rowid]
  matching_pids <- unique(pid_of_row[per_row$match])
  pid %in% matching_pids
}

# --- span / between ----------------------------------------------------

tq_eval.WithinSpanExpr <- function(node, env) {
  child_mask <- tq_eval(node$child, env)
  ref_mask   <- tq_eval(node$ref,   env)
  .eval_span(env, child_mask, ref_mask, ref_mask, isTRUE(node$outside))
}

tq_eval.BetweenExpr <- function(node, env) {
  child_mask <- tq_eval(node$child, env)
  start_mask <- tq_eval(node$bound_start, env)
  end_mask   <- tq_eval(node$bound_end,   env)
  .eval_span(env, child_mask, start_mask, end_mask, isTRUE(node$outside))
}

.eval_span <- function(env, child_mask, start_mask, end_mask, outside) {
  if (!any(child_mask) || !any(start_mask) || !any(end_mask)) {
    return(rep(FALSE, env$nrow))
  }
  pid   <- env$pid
  dates <- env$date

  s_dt <- data.table(pid = pid[start_mask], date = dates[start_mask])
  e_dt <- data.table(pid = pid[end_mask],   date = dates[end_mask])
  s_range <- s_dt[, .(s_min = min(date)), by = pid]
  e_range <- e_dt[, .(e_max = max(date)), by = pid]

  row_dt <- data.table(idx = seq_len(env$nrow), pid = pid, date = dates)
  row_dt[s_range, s_min := i.s_min, on = "pid"]
  row_dt[e_range, e_max := i.e_max, on = "pid"]
  setorder(row_dt, idx)

  hits <- !is.na(row_dt$s_min) & !is.na(row_dt$e_max) &
          row_dt$date >= row_dt$s_min & row_dt$date <= row_dt$e_max
  positive <- child_mask & hits
  if (!outside) return(positive)
  .row_complement_evaluable(env, child_mask, start_mask | end_mask, positive)
}

# --- quantifier in isolation (shouldn't normally be reached) ----------

tq_eval.Quantifier <- function(node, env) tq_eval(node$child, env)
