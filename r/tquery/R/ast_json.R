# JSON ↔ AST conversion for cross-language testing.
# Mirrors tquery/_ast_json.py. Spec format: see spec/ast.md.


# Field name + (optional) factory metadata for each node kind.
# Order matters for reconstruction.
.AST_FIELDS <- list(
  CodeAtom        = list(fields = c("codes", "columns"),                                  ctor = new_code_atom),
  EventAtom       = list(fields = character(),                                            ctor = function() new_event_atom()),
  ComparisonAtom  = list(fields = c("column", "op", "value"),                             ctor = new_comparison_atom),
  AggregateExpr   = list(fields = c("func", "column", "op", "value", "relative"),         ctor = new_aggregate_expr),
  PrefixExpr      = list(fields = c("kind", "n", "child"),                                ctor = new_prefix_expr),
  RangePrefixExpr = list(fields = c("min_n", "max_n", "child"),                           ctor = new_range_prefix_expr),
  NotExpr         = list(fields = "child",                                                ctor = new_not_expr),
  BinaryLogical   = list(fields = c("op", "left", "right"),                               ctor = new_binary_logical),
  TemporalExpr    = list(fields = c("op", "left", "right"),                               ctor = new_temporal_expr),
  WithinExpr      = list(fields = c("child", "days", "min_days", "direction", "ref", "outside"), ctor = new_within_expr),
  WithinSpanExpr  = list(fields = c("child", "ref", "outside"),                           ctor = new_within_span_expr),
  InsideExpr      = list(fields = c("child", "inside", "min_events", "max_events", "direction", "ref"), ctor = new_inside_expr),
  BetweenExpr     = list(fields = c("child", "bound_start", "bound_end", "outside"),      ctor = new_between_expr),
  ShiftExpr       = list(fields = c("child", "offset_days"),                              ctor = new_shift_expr),
  Quantifier      = list(fields = c("kind", "child"),                                     ctor = new_quantifier)
)

ast_to_json <- function(node) {
  stopifnot(inherits(node, "tq_ast"))
  kind <- class(node)[1]
  fields <- .AST_FIELDS[[kind]]$fields
  out <- list(`_node` = kind)
  for (f in fields) {
    out[[f]] <- .encode(node[[f]])
  }
  out
}

ast_from_json <- function(obj) {
  if (!is.list(obj) || is.null(obj$`_node`)) {
    stop("Not an AST JSON object")
  }
  kind <- obj$`_node`
  spec <- .AST_FIELDS[[kind]]
  if (is.null(spec)) stop("Unknown AST node type: ", kind)
  args <- list()
  for (f in spec$fields) {
    args[[f]] <- .decode(obj[[f]], f)
  }
  do.call(spec$ctor, args)
}

.encode <- function(v) {
  if (is.null(v)) return(NULL)
  if (inherits(v, "tq_ast")) return(ast_to_json(v))
  # Code/column tuples: emit as JSON arrays. jsonlite serialises char
  # vectors as arrays which matches the Python convention.
  if (is.character(v) && length(v) >= 1L) return(as.list(v))
  # jsonlite::unbox makes length-1 vectors serialize as JSON scalars rather
  # than 1-element arrays. Only used if jsonlite is available.
  if (is.logical(v) || is.numeric(v) || is.integer(v)) {
    return(if (requireNamespace("jsonlite", quietly = TRUE)) jsonlite::unbox(v) else v)
  }
  if (is.character(v)) {
    return(if (requireNamespace("jsonlite", quietly = TRUE)) jsonlite::unbox(v) else v)
  }
  v
}

.decode <- function(v, field) {
  if (is.null(v)) return(NULL)
  if (is.list(v) && !is.null(v$`_node`)) return(ast_from_json(v))
  # Lists of strings (codes / columns tuples) come back as list-of-1-strings;
  # flatten to character vector.
  if (is.list(v) && length(v) >= 1L && all(vapply(v, is.character, logical(1)))) {
    return(unlist(v, use.names = FALSE))
  }
  v
}
