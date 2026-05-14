# R port of tquery/_ast.py — S3 AST node constructors.
#
# Each node is a named list with a `class` vector ending in "tq_ast".
# Constructors mirror the Python dataclasses in field name and order.
# Spec: see spec/ast.md at the repo root.

new_node <- function(.cls, ..., extra_classes = character()) {
  obj <- list(...)
  structure(obj, class = c(.cls, extra_classes, "tq_ast"))
}

new_code_atom <- function(codes, columns = NULL) {
  stopifnot(is.character(codes), length(codes) >= 1L)
  new_node("CodeAtom", codes = codes, columns = columns)
}

new_event_atom <- function() {
  new_node("EventAtom")
}

new_comparison_atom <- function(column, op, value) {
  stopifnot(op %in% c(">", "<", ">=", "<=", "==", "!="))
  new_node("ComparisonAtom", column = column, op = op, value = as.numeric(value))
}

# v0.2: person-level aggregate over a numeric column with comparison.
new_aggregate_expr <- function(func, column, op, value, relative = FALSE) {
  stopifnot(func %in% c("sum","mean","avg","min","max","median","sd","var","count","n","range","rise","fall"))
  stopifnot(op %in% c(">", "<", ">=", "<=", "==", "!="))
  # `relative=TRUE` flags a `%` threshold (rise/fall v0.2.3; range v0.2.4).
  if (isTRUE(relative) && !(func %in% c("rise", "fall", "range"))) {
    stop("relative=TRUE only supported for rise/fall/range, not ", func)
  }
  new_node("AggregateExpr",
    func = func, column = column, op = op,
    value = as.numeric(value), relative = isTRUE(relative))
}

new_prefix_expr <- function(kind, n, child) {
  stopifnot(kind %in% c("min", "max", "exactly", "ordinal", "first", "last"))
  new_node("PrefixExpr", kind = kind, n = as.integer(n), child = child)
}

new_range_prefix_expr <- function(min_n, max_n, child) {
  stopifnot(min_n <= max_n)
  new_node("RangePrefixExpr",
    min_n = as.integer(min_n), max_n = as.integer(max_n), child = child)
}

new_not_expr <- function(child) {
  new_node("NotExpr", child = child)
}

new_binary_logical <- function(op, left, right) {
  stopifnot(op %in% c("and", "or"))
  new_node("BinaryLogical", op = op, left = left, right = right)
}

new_temporal_expr <- function(op, left, right) {
  stopifnot(op %in% c("before", "after", "simultaneously"))
  new_node("TemporalExpr", op = op, left = left, right = right)
}

new_within_expr <- function(child, days, min_days = 0L, direction = NULL, ref = NULL, outside = FALSE) {
  new_node("WithinExpr",
    child = child, days = as.integer(days),
    min_days = as.integer(min_days), direction = direction,
    ref = ref, outside = isTRUE(outside))
}

new_within_span_expr <- function(child, ref, outside = FALSE) {
  new_node("WithinSpanExpr", child = child, ref = ref, outside = isTRUE(outside))
}

new_inside_expr <- function(child, inside, min_events, max_events,
                            direction = NULL, ref = NULL) {
  # v0.2.1: direction and ref may be NULL when child is an AggregateExpr
  # (sliding event-window aggregate). For other children, both are required.
  if (!is.null(direction)) {
    stopifnot(direction %in% c("before", "after", "around"))
  }
  new_node("InsideExpr",
    child = child, inside = isTRUE(inside),
    min_events = as.integer(min_events), max_events = as.integer(max_events),
    direction = direction, ref = ref)
}

new_between_expr <- function(child, bound_start, bound_end, outside = FALSE) {
  new_node("BetweenExpr",
    child = child, bound_start = bound_start, bound_end = bound_end,
    outside = isTRUE(outside))
}

new_shift_expr <- function(child, offset_days) {
  new_node("ShiftExpr", child = child, offset_days = as.integer(offset_days))
}

new_quantifier <- function(kind, child) {
  stopifnot(kind %in% c("any", "every"))
  new_node("Quantifier", kind = kind, child = child)
}

is_tq_ast <- function(x) inherits(x, "tq_ast")

# Predicate used by parser to validate `± N days` shift suffix.
is_single_date_expr <- function(node) {
  if (inherits(node, "ShiftExpr")) return(TRUE)
  if (inherits(node, "PrefixExpr") && node$kind == "ordinal") return(TRUE)
  FALSE
}
