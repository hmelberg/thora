test_that("R evaluator matches Python pids+count for every golden", {
  skip_if_not_installed("jsonlite")
  dpath <- system.file("extdata", "dataset.csv",   package = "tquery")
  qpath <- system.file("extdata", "queries.jsonl", package = "tquery")
  expect_true(nzchar(dpath) && nzchar(qpath))

  dt <- data.table::fread(dpath)
  dt[, start_date := as.Date(start_date)]
  data.table::setkey(dt, pid, start_date)

  fixtures <- lapply(readLines(qpath), jsonlite::fromJSON, simplifyVector = FALSE)

  for (f in fixtures) {
    py_pids  <- sort(as.integer(unlist(f$pids)))
    py_count <- as.integer(f$count)
    r <- tquery(dt, f$query, pid = "pid", date = "start_date", cols = "icd")
    expect_identical(as.integer(r$count), py_count, info = paste("query:", f$query))
    expect_identical(sort(as.integer(r$pids)), py_pids, info = paste("query:", f$query))
  }
})
