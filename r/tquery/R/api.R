# Public API: tquery(), count_persons() — entry points for users.

# Returns a list with class "tquery_result" — analogous to TQueryResult
# in tquery/_types.py but minimal for v1.
tquery <- function(dt, expr, pid = "pid", date = "start_date",
                   cols = NULL, sep = NULL, variables = list()) {
  if (!inherits(dt, "data.table")) dt <- as.data.table(dt)
  if (!inherits(dt[[date]], "Date")) {
    dt[[date]] <- as.Date(dt[[date]])
  }
  ast <- parse_query(expr)
  env <- make_eval_env(dt, pid, date, cols, sep, variables)
  mask <- tq_eval(ast, env)
  structure(list(
    expr  = expr,
    ast   = ast,
    mask  = mask,
    pids  = sort(unique(env$pid[mask])),
    count = length(unique(env$pid[mask])),
    env   = env
  ), class = "tquery_result")
}

count_persons <- function(dt, expr, ...) tquery(dt, expr, ...)$count

event_counts <- function(dt, expr, ...) {
  r <- tquery(dt, expr, ...)
  pid <- r$env$pid
  stats::setNames(tabulate(match(pid[r$mask], r$pids)), r$pids)
}

print.tquery_result <- function(x, ...) {
  cat(sprintf("tquery_result: expr='%s'  count=%d  rows=%d\n",
              x$expr, x$count, sum(x$mask)))
  invisible(x)
}
