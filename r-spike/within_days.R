# R port spike: eval_within_days using data.table rolling join.
#
# Implements the WithinExpr semantics from spec/semantics.md for the
# row-anchored, existential, unsigned-window case (the most common path
# through the Python evaluator: `K50 inside N days [direction] [Y]`).
#
# Skipped in this spike (handled by the full R port later):
#   - universal quantifiers (every_left / every_right)
#   - the ref=NULL case (window from each person's first event)
#   - `outside` (row-level complement restricted to evaluable persons)

suppressPackageStartupMessages(library(data.table))

# Inclusive |delta| <= days, optional lower bound, optional direction.
# Returns logical vector of length nrow(dt) selecting matching CHILD rows.
#
# Direction → data.table rolling join mapping:
#   "after"  ⇒ child > ref ⇒ look BACKWARD from child to find ref
#              roll = +days, rollends = c(FALSE, TRUE)
#              (TRUE on the upper end so the prevailing past-ref still
#               matches when ref is the last x value in its group.)
#   "before" ⇒ child < ref ⇒ look FORWARD from child to find ref
#              roll = -days, rollends = c(TRUE, FALSE)
#   "around" ⇒ either direction ⇒ run both directions, union the matches.
#   NA       ⇒ "nearest" — same as around (the parser doesn't actually emit
#              this combination, but the path is here for completeness).
#
# Inclusivity note: pandas merge_asof with tolerance=N matches |delta| <= N
# (inclusive on both ends — verified empirically). data.table's roll has the
# same semantics in 1.18, but we still apply abs(delta) <= days post-filter
# to guarantee parity across data.table versions.
eval_within_days <- function(
  dt, child_mask, ref_mask, days,
  min_days = 0L, direction = NA_character_
) {
  stopifnot(
    is.logical(child_mask), is.logical(ref_mask),
    length(child_mask) == nrow(dt), length(ref_mask) == nrow(dt),
    inherits(dt$date, "Date")
  )

  if (!any(child_mask) || !any(ref_mask)) {
    return(rep(FALSE, nrow(dt)))
  }

  child_dt <- data.table(
    pid       = dt$pid[child_mask],
    date      = dt$date[child_mask],
    orig_idx  = which(child_mask)
  )
  ref_dt <- data.table(
    pid      = dt$pid[ref_mask],
    ref_date = dt$date[ref_mask]
  )
  setkey(child_dt, pid, date)
  setkey(ref_dt, pid, ref_date)

  roll_join <- function(roll_val, ends) {
    ref_dt[child_dt,
      .(orig_idx = i.orig_idx, child_date = i.date, ref_date = x.ref_date),
      on = .(pid, ref_date = date), roll = roll_val, rollends = ends
    ]
  }

  # Signed `around` (min_days < 0) uses signed delta; rolls are sized to
  # each side's bound. All other cases use unsigned |delta|.
  signed <- !is.na(direction) && direction == "around" && min_days < 0

  if (is.na(direction) || direction == "around") {
    bw_roll <- days
    fw_roll <- if (signed) -abs(min_days) else -days
    bw <- roll_join(bw_roll, c(FALSE, TRUE))
    fw <- roll_join(fw_roll, c(TRUE, FALSE))
    matched <- rbind(bw, fw)
  } else if (direction == "after") {
    matched <- roll_join( days, c(FALSE, TRUE))
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

  result <- rep(FALSE, nrow(dt))
  result[unique(matched$orig_idx)] <- TRUE
  result
}

# Helper: match a code pattern (plain or `*`-wildcard) against a character vector.
match_code <- function(values, pattern) {
  if (endsWith(pattern, "*")) {
    prefix <- substr(pattern, 1, nchar(pattern) - 1L)
    startsWith(as.character(values), prefix)
  } else {
    as.character(values) == pattern
  }
}
