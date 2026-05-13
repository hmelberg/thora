# Silence R CMD check NOTEs from data.table non-standard evaluation:
# bare column names inside DT[i, j, by] would otherwise look "undefined".
utils::globalVariables(c(
  # column names used in data.table NSE inside the package
  "pid", "date", "ref_date", "child_date", "orig_idx",
  "i.orig_idx", "i.date", "x.ref_date", "i.ref_date", "x.child_date",
  "delta", "m", "cs", "total", "ok",
  "s_min", "e_max", "i.s_min", "i.e_max",
  "first", "idx",
  ".N", ".SD", ".I",
  # aggregation result columns referenced by name in data.table expressions
  "l_min", "l_max", "r_min", "r_max",
  # data.table's "..varname" syntax for using local-variable column names
  "..cols",
  # v0.2 aggregate evaluator columns
  "v", "v_in", "win_start", "agg", "rowid", "i.v", "x.v",
  "i.rowid", "match", "p",
  # v0.2.1 event-window aggregate columns
  "pos", "lo_pos", "i.pos", "x.pos",
  # v0.2.2 chronological-ordering columns for rise/fall in sliding windows
  "d_in", "x.date", "p_in"
))
