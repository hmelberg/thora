# Time the R port on the same benchmark query set as bench_python.py.
# Requires the tquery package to be installed (see r/tquery/).

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(tquery)
})

script_path <- (function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- regmatches(args, regexpr("--file=.*", args))
  if (length(m) > 0) sub("--file=", "", m) else "tools/bench_r.R"
})()
repo_root <- normalizePath(file.path(dirname(script_path), ".."))

dt <- fread(file.path(repo_root, "bench", "dataset_large.csv"))
dt[, start_date := as.Date(start_date)]
setkey(dt, pid, start_date)
cat(sprintf("loaded %d rows, %d persons\n\n", nrow(dt), uniqueN(dt$pid)))

QUERIES <- c(
  "K50",
  "K50*",
  "K50 and K51",
  "K50 or K51",
  "not K50",
  "min 3 of K50",
  "2-5 of K50",
  "1st K50",
  "last 2 of K50",
  "K50 before K51",
  "K50 after K51",
  "every K50 before K51",
  "K50 inside 30 days after K51",
  "K50 inside 30 to 90 days after K51",
  "K50 inside 100 days",
  "K50 outside 30 days after K51",
  "K50 inside -5 to 20 days around K51",
  "K50 inside 5 events after K51",
  "K50 inside last 5 events",
  "(K50 or K51) and K52"
)

out_path <- file.path(repo_root, "bench", "results_r.jsonl")
con <- file(out_path, "w")
on.exit(close(con))

cat(sprintf("%-55s  %7s  %10s\n", "query", "count", "time_ms"))
for (q in QUERIES) {
  tquery::tquery(dt, q, pid = "pid", date = "start_date", cols = "icd")  # warm-up
  t0 <- Sys.time()
  r <- tquery::tquery(dt, q, pid = "pid", date = "start_date", cols = "icd")
  elapsed_ms <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  cat(sprintf("%-55s  %7d  %10.2f\n", q, r$count, elapsed_ms))
  writeLines(toJSON(list(query = q, count = unbox(r$count), time_ms = unbox(elapsed_ms)),
                    auto_unbox = TRUE), con)
}
