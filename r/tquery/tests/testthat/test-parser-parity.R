test_that("R parser produces ASTs matching the Python reference for every golden", {
  skip_if_not_installed("jsonlite")
  qpath <- system.file("extdata", "queries.jsonl", package = "tquery")
  expect_true(nzchar(qpath))

  fixtures <- lapply(readLines(qpath), jsonlite::fromJSON, simplifyVector = FALSE)
  expect_gt(length(fixtures), 0)

  for (f in fixtures) {
    py_ast <- ast_from_json(f$ast)
    r_ast  <- parse_query(f$query)
    # Lenient deep compare: jsonlite roundtrip may give double where parser
    # gives integer for `n`/`days`, so compare via re-serialised JSON.
    expect_equal(
      ast_to_json(r_ast),
      ast_to_json(py_ast),
      info = paste("query:", f$query)
    )
  }
})
