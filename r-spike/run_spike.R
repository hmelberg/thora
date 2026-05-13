# R port spike: parity check against the Python golden fixtures.
#
# Loads spec/golden/dataset.csv and replays the `inside N days` queries
# from spec/golden/queries.jsonl through the R implementation in
# within_days.R. Prints per-query parity (R pids vs. expected pids).

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

script_path <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- regmatches(args, regexpr("--file=.*", args))
  if (length(m) > 0) sub("--file=", "", m) else "r-spike/run_spike.R"
})()
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
source(file.path(repo_root, "r-spike", "within_days.R"))

dataset_path <- file.path(repo_root, "spec", "golden", "dataset.csv")
queries_path <- file.path(repo_root, "spec", "golden", "queries.jsonl")

dt <- fread(dataset_path)
dt[, start_date := as.Date(start_date)]
setnames(dt, "start_date", "date")
setkey(dt, pid, date)

cat(sprintf("loaded %d rows / %d persons\n", nrow(dt), uniqueN(dt$pid)))

# Parse the golden fixtures
fixtures <- lapply(readLines(queries_path), fromJSON, simplifyVector = FALSE)

# --- Hand-encoded AST interpretation for the spike ----------------------
# We only evaluate WithinExpr nodes whose child + ref are simple CodeAtoms
# and whose `outside` is FALSE. Everything else is skipped — the full
# port will dispatch on every node type.
within_queries <- Filter(function(f) {
  ast <- f$ast
  if (is.null(ast$`_node`) || ast$`_node` != "WithinExpr") return(FALSE)
  if (!is.null(ast$outside) && isTRUE(ast$outside)) return(FALSE)
  if (is.null(ast$ref)) return(FALSE)
  ast$child$`_node` == "CodeAtom" && ast$ref$`_node` == "CodeAtom"
}, fixtures)

cat(sprintf("running %d WithinExpr fixtures\n\n", length(within_queries)))

passes <- 0L; failures <- list()
for (f in within_queries) {
  ast <- f$ast
  child_codes <- unlist(ast$child$codes)
  ref_codes   <- unlist(ast$ref$codes)
  days        <- as.integer(ast$days)
  min_days    <- if (is.null(ast$min_days)) 0L else as.integer(ast$min_days)
  direction   <- if (is.null(ast$direction)) NA_character_ else ast$direction

  child_mask <- Reduce(`|`, lapply(child_codes, match_code, values = dt$icd))
  ref_mask   <- Reduce(`|`, lapply(ref_codes,   match_code, values = dt$icd))

  result <- eval_within_days(dt, child_mask, ref_mask, days, min_days, direction)
  r_pids <- sort(unique(dt$pid[result]))
  py_pids <- sort(unlist(f$pids))

  ok <- identical(as.integer(r_pids), as.integer(py_pids))
  status <- if (ok) "OK " else "BAD"
  if (ok) passes <- passes + 1L
  else failures[[length(failures) + 1L]] <- list(
    q = f$query, r = r_pids, py = py_pids
  )
  cat(sprintf("[%s] %-60s  R=%d py=%d\n", status, f$query, length(r_pids), length(py_pids)))
}

cat(sprintf("\n%d/%d fixtures pass\n", passes, length(within_queries)))

if (length(failures) > 0L) {
  cat("\n--- failures ---\n")
  for (f in failures) {
    only_r  <- setdiff(f$r,  f$py)
    only_py <- setdiff(f$py, f$r)
    cat(sprintf("\nquery: %s\n", f$q))
    cat(sprintf("  only in R  (%d): %s\n", length(only_r),  paste(head(only_r,  10), collapse = ",")))
    cat(sprintf("  only in py (%d): %s\n", length(only_py), paste(head(only_py, 10), collapse = ",")))
  }
}

quit(status = if (passes == length(within_queries)) 0L else 1L)
