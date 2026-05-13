# R port of tquery/_parser.py -- tokenizer + recursive descent parser.
#
# Translates the Python parser node-for-node. Grammar reference:
# spec/grammar.md. AST node reference: spec/ast.md. The functions in
# this file produce the same AST shapes as the Python parser.

# Loaded relative to this file's directory by source()'ing parser.R after
# ast.R. The test driver in r/tests/ takes care of that.

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

.KEYWORDS <- c(
  "before", "after", "simultaneously",
  "inside", "outside",
  "and", "or", "not", "never",
  "min", "max", "exactly",
  "of", "in", "to",
  "first", "last",
  "days", "event", "events",
  "around",
  "every", "any", "each", "always",
  # v0.2 aggregate functions. `min` / `max` already listed above; the
  # parser disambiguates aggregate vs prefix by 1-token lookahead.
  "sum", "mean", "avg", "median", "sd", "var", "count", "n",
  "range",  # v0.2.1
  "rise", "fall"  # v0.2.2 — signed range
)

.AGG_FUNCS <- c("sum", "mean", "avg", "min", "max",
                "median", "sd", "var", "count", "n",
                "range",  # v0.2.1
                "rise", "fall")  # v0.2.2

# Patterns: anchored at start of the remaining buffer. Order matters.
.TOKEN_PATTERNS <- list(
  list(re = "^\\s+",                                                  type = NULL),
  list(re = "^\\(",                                                   type = "LPAREN"),
  list(re = "^\\)",                                                   type = "RPAREN"),
  list(re = "^,",                                                     type = "COMMA"),
  list(re = "^@",                                                     type = "AT"),
  list(re = "^(>=|<=|!=|==|[><+\\-])",                                type = "OP"),
  list(re = "^\\d+(st|nd|rd|th)\\b",                                  type = "ORDINAL"),
  list(re = "^\\d+\\.\\d+",                                           type = "FLOAT"),
  list(re = "^\\d+-\\d+",                                             type = "INT_RANGE"),
  list(re = "^\\d+",                                                  type = "INT"),
  list(re = "^[A-Za-z_]\\w*(?:\\.\\w+)?\\*",                          type = "STAR_CODE"),
  list(re = "^[A-Za-z_]\\w*:[A-Za-z_]\\w*",                           type = "SLICE_CODE"),
  list(re = "^[A-Za-z]\\w*(?:\\.\\w+)?-[A-Za-z]\\w*(?:\\.\\w+)?",     type = "RANGE_CODE"),
  list(re = "^[A-Za-z_]\\w*(?:\\.\\w+)?",                             type = "CODE")
)

.syntax_error <- function(msg, expr, pos) {
  stop(structure(
    class = c("tq_syntax_error", "error", "condition"),
    list(message = sprintf("%s (at position %d in '%s')", msg, pos, expr),
         call = NULL, expr = expr, pos = pos, reason = msg)
  ))
}

tokenize <- function(expr) {
  tokens <- list()
  pos <- 1L  # 1-based position in `expr`
  n <- nchar(expr)
  while (pos <= n) {
    rest <- substr(expr, pos, n)
    matched <- FALSE
    for (pat in .TOKEN_PATTERNS) {
      m <- regexpr(pat$re, rest, perl = TRUE)
      if (m == 1L) {
        len <- attr(m, "match.length")
        raw <- substr(rest, 1L, len)
        if (!is.null(pat$type)) {
          ttype <- pat$type
          value <- raw
          if (ttype == "INT")     value <- as.integer(raw)
          else if (ttype == "FLOAT")   value <- as.numeric(raw)
          else if (ttype == "ORDINAL") value <- as.integer(sub("(st|nd|rd|th)$", "", raw))
          else if (ttype == "CODE" && tolower(raw) %in% .KEYWORDS) {
            ttype <- "KEYWORD"
            value <- tolower(raw)
          }
          tokens[[length(tokens) + 1L]] <- list(type = ttype, value = value, pos = pos - 1L)
        }
        pos <- pos + len
        matched <- TRUE
        break
      }
    }
    if (!matched) {
      .syntax_error(sprintf("Unexpected character %q", substr(rest, 1L, 1L)), expr, pos - 1L)
    }
  }
  tokens[[length(tokens) + 1L]] <- list(type = "EOF", value = NULL, pos = n)
  tokens
}

# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

# Mutable parser state lives in an environment so functions can mutate $pos.
.make_parser <- function(tokens, expr) {
  env <- new.env(parent = emptyenv())
  env$tokens <- tokens
  env$expr   <- expr
  env$pos    <- 1L
  env
}

.peek    <- function(p)        p$tokens[[p$pos]]
.advance <- function(p)        { tok <- p$tokens[[p$pos]]; p$pos <- p$pos + 1L; tok }
.at_type    <- function(p, types)    .peek(p)$type %in% types
.at_keyword <- function(p, values)   { t <- .peek(p); t$type == "KEYWORD" && t$value %in% values }
.error   <- function(p, msg)   .syntax_error(msg, p$expr, .peek(p)$pos)

.expect_keyword <- function(p, values) {
  t <- .peek(p)
  if (t$type == "KEYWORD" && t$value %in% values) return(.advance(p))
  .error(p, sprintf("Expected one of %s, got %s",
                    paste(sprintf("'%s'", values), collapse = ", "),
                    deparse(t$value)))
}

# --- Grammar rules -----------------------------------------------------

parse_query <- function(expr) {
  tokens <- tokenize(expr)
  p <- .make_parser(tokens, expr)
  node <- .parse_or(p)
  if (.peek(p)$type != "EOF") {
    .error(p, sprintf("Unexpected token: %s", deparse(.peek(p)$value)))
  }
  node
}

.parse_or <- function(p) {
  left <- .parse_and(p)
  while (.at_keyword(p, "or")) {
    .advance(p)
    right <- .parse_and(p)
    left <- new_binary_logical("or", left, right)
  }
  left
}

.parse_and <- function(p) {
  left <- .parse_temporal(p)
  while (.at_keyword(p, "and")) {
    .advance(p)
    right <- .parse_temporal(p)
    left <- new_binary_logical("and", left, right)
  }
  left
}

.parse_temporal <- function(p) {
  lhs_q <- .try_quantified(p)
  left <- if (!is.null(lhs_q)) lhs_q else .parse_not(p)
  if (.at_keyword(p, c("before", "after", "simultaneously"))) {
    op <- .advance(p)$value
    rhs_q <- .try_quantified(p)
    right <- if (!is.null(rhs_q)) rhs_q else .parse_not(p)
    return(new_temporal_expr(op, left, right))
  }
  if (inherits(left, "Quantifier")) {
    .error(p, sprintf("'%s' requires a temporal operator (before/after/simultaneously)", left$kind))
  }
  left
}

.try_quantified <- function(p) {
  if (!.at_keyword(p, c("every", "any", "each", "always"))) return(NULL)
  kind_keyword <- .advance(p)$value
  is_any <- kind_keyword == "any"

  tok <- .peek(p)
  if (tok$type == "LPAREN") {
    .error(p, sprintf("'%s' cannot be applied to a parenthesized group", kind_keyword))
  }
  if (tok$type == "KEYWORD" && tok$value %in% c("min","max","exactly","first","last","not","never")) {
    .error(p, sprintf("'%s' cannot be combined with '%s'", kind_keyword, tok$value))
  }
  if (tok$type %in% c("INT_RANGE", "ORDINAL")) {
    .error(p, sprintf("'%s' cannot be combined with a count prefix", kind_keyword))
  }

  atom <- .parse_code_expr(p)
  wrapped <- if (is_any) atom else new_quantifier("every", atom)
  if (.at_keyword(p, c("inside", "outside"))) {
    return(.parse_within(p, child = wrapped))
  }
  wrapped
}

.parse_not <- function(p) {
  if (.at_keyword(p, c("not", "never"))) {
    .advance(p)
    return(new_not_expr(.parse_prefix(p)))
  }
  .parse_prefix(p)
}

.parse_prefix <- function(p) {
  .maybe_attach_shift(p, .parse_prefix_core(p))
}

.maybe_attach_shift <- function(p, expr) {
  # Force eager evaluation: `expr = .parse_prefix_core(p)` at the call site
  # has a side effect on `p$pos`. R's lazy promises would otherwise let the
  # while loop run before `.parse_prefix_core` advances `p`, leaving `p$pos`
  # at the start of the prefix and the shift lookahead reading the wrong
  # tokens. force() makes the side effect happen first.
  force(expr)
  while (.lookahead_is_shift(p)) {
    sign <- if (.advance(p)$value == "+") 1L else -1L
    n <- .advance(p)$value
    .advance(p)  # 'days'
    if (!is_single_date_expr(expr)) {
      .error(p, "'+/- N days' requires a single-date anchor (an ordinal like `1st K51` or `-1st event`)")
    }
    expr <- new_shift_expr(expr, sign * n)
  }
  expr
}

.lookahead_is_shift <- function(p) {
  t0 <- .peek(p)
  if (!(t0$type == "OP" && t0$value %in% c("+", "-"))) return(FALSE)
  if (p$pos + 2L > length(p$tokens)) return(FALSE)
  t1 <- p$tokens[[p$pos + 1L]]
  t2 <- p$tokens[[p$pos + 2L]]
  t1$type == "INT" && t2$type == "KEYWORD" && t2$value == "days"
}

.parse_prefix_core <- function(p) {
  # Count range: 2-5 of K50
  if (.at_type(p, "INT_RANGE")) {
    raw <- .advance(p)$value
    parts <- strsplit(raw, "-", fixed = TRUE)[[1]]
    min_n <- as.integer(parts[1]); max_n <- as.integer(parts[2])
    if (.at_keyword(p, "of")) .advance(p)
    return(new_range_prefix_expr(min_n, max_n, .parse_within(p)))
  }

  if (.at_keyword(p, c("min", "max", "exactly"))) {
    # `min(` / `max(` are aggregates, not count prefixes.
    if (.peek(p)$value %in% c("min", "max") &&
        p$pos + 1L <= length(p$tokens) &&
        p$tokens[[p$pos + 1L]]$type == "LPAREN") {
      return(.parse_within(p))
    }
    kind <- .advance(p)$value
    tok <- .peek(p)
    if (tok$type != "INT") .error(p, sprintf("Expected integer after '%s', got %s", kind, deparse(tok$value)))
    n <- .advance(p)$value
    if (.at_keyword(p, "of")) .advance(p)
    return(new_prefix_expr(kind, n, .parse_within(p)))
  }

  if (.at_type(p, "ORDINAL")) {
    n <- .advance(p)$value
    if (.at_keyword(p, "of")) .advance(p)
    return(new_prefix_expr("ordinal", n, .parse_within(p)))
  }

  # Negative ordinal: -2nd X
  if (.at_type(p, "OP") && .peek(p)$value == "-" &&
      p$pos + 1L <= length(p$tokens) &&
      p$tokens[[p$pos + 1L]]$type == "ORDINAL") {
    .advance(p)  # '-'
    n <- .advance(p)$value
    if (n == 0L) .error(p, "Ordinal cannot be -0")
    if (.at_keyword(p, "of")) .advance(p)
    return(new_prefix_expr("ordinal", -n, .parse_within(p)))
  }

  if (.at_keyword(p, c("first", "last"))) {
    kind <- .advance(p)$value
    tok <- .peek(p)
    if (tok$type != "INT") .error(p, sprintf("Expected integer after '%s', got %s", kind, deparse(tok$value)))
    n <- .advance(p)$value
    if (.at_keyword(p, "of")) .advance(p)
    return(new_prefix_expr(kind, n, .parse_within(p)))
  }

  .parse_within(p)
}

.parse_within <- function(p, child = NULL) {
  if (is.null(child)) child <- .parse_atom(p)
  if (!.at_keyword(p, c("inside", "outside"))) return(child)

  keyword <- .advance(p)$value
  outside <- keyword == "outside"
  tok <- .peek(p)

  # Numeric: inside [-]N [to [-]M] days/events ...
  is_minus_int <- tok$type == "OP" && tok$value == "-" &&
    p$pos + 1L <= length(p$tokens) &&
    p$tokens[[p$pos + 1L]]$type == "INT"
  if (tok$type == "INT" || is_minus_int) {
    first <- .parse_signed_int(p)
    min_val <- 0L; max_val <- first
    if (.at_keyword(p, "to")) {
      .advance(p)
      min_val <- first
      max_val <- .parse_signed_int(p)
    }
    return(.finish_numeric_inside(p, child, min_val, max_val, outside))
  }

  # EXPR form: positional span or positional bounds
  bound_start <- .parse_prefix(p)
  if (.at_keyword(p, "and")) {
    .advance(p)
    bound_end <- .parse_prefix(p)
    return(new_between_expr(child, bound_start, bound_end, outside = outside))
  }
  new_within_span_expr(child, bound_start, outside = outside)
}

.parse_signed_int <- function(p) {
  negate <- FALSE
  tok <- .peek(p)
  if (tok$type == "OP" && tok$value == "-") { .advance(p); negate <- TRUE }
  tok <- .peek(p)
  if (tok$type != "INT") .error(p, sprintf("Expected integer, got %s", deparse(tok$value)))
  v <- .advance(p)$value
  if (negate) -v else v
}

.finish_numeric_inside <- function(p, child, min_val, max_val, outside) {
  unit_tok <- .peek(p)
  if (!(unit_tok$type == "KEYWORD" && unit_tok$value %in% c("days", "event", "events"))) {
    .error(p, sprintf("Expected 'days' or 'events', got %s", deparse(unit_tok$value)))
  }
  unit_raw <- .advance(p)$value
  unit <- if (unit_raw %in% c("event", "events")) "events" else unit_raw

  direction <- NULL; ref <- NULL
  if (.at_keyword(p, c("before", "after", "around"))) {
    direction <- .advance(p)$value
    if (.at_keyword(p, "of")) .advance(p)
    rhs_q <- .try_quantified(p)
    ref <- if (!is.null(rhs_q)) rhs_q else .parse_prefix(p)
  }

  has_negative <- min_val < 0L || max_val < 0L
  if (has_negative && (is.null(direction) || direction != "around")) {
    .error(p, "Negative offsets are only allowed with 'around' (use 'before' with a positive number instead)")
  }
  if (min_val > max_val) {
    .error(p, sprintf("Range must be ascending: got %d to %d", min_val, max_val))
  }

  if (unit == "days") {
    return(new_within_expr(child, max_val, min_val, direction, ref, outside = outside))
  }
  # events
  if (is.null(direction) || is.null(ref)) {
    # v0.2.1: sliding event-window over an AggregateExpr child.
    if (inherits(child, "AggregateExpr")) {
      if (outside) {
        .error(p, "`outside N events` over a sliding aggregate has no defined semantics")
      }
      if (min_val != 0L) {
        .error(p, "Sliding event window for aggregate cannot have a lower bound (use 'inside N events')")
      }
      return(new_inside_expr(child, TRUE, 0L, max_val, NULL, NULL))
    }
    .error(p, "Event window requires a direction (before/after/around) and a reference expression")
  }
  if (min_val == 0L && max_val > 0L && direction != "around") {
    min_events <- 1L; max_events <- max_val
  } else {
    min_events <- min_val; max_events <- max_val
  }
  new_inside_expr(child, !outside, min_events, max_events, direction, ref)
}

.parse_atom <- function(p) {
  tok <- .peek(p)
  if (tok$type == "LPAREN") {
    .advance(p)
    node <- .parse_or(p)
    if (.peek(p)$type != "RPAREN") .error(p, "Expected ')'")
    .advance(p)
    return(node)
  }
  if (.at_keyword(p, c("event", "events"))) {
    .advance(p)
    return(new_event_atom())
  }
  agg <- .try_aggregate_atom(p)
  if (!is.null(agg)) return(agg)
  if (.is_comparison_ahead(p)) return(.parse_comparison(p))
  .parse_code_expr(p)
}

.try_aggregate_atom <- function(p) {
  tok <- .peek(p)
  if (tok$type != "KEYWORD" || !(tok$value %in% .AGG_FUNCS)) return(NULL)
  if (p$pos + 1L > length(p$tokens)) return(NULL)
  if (p$tokens[[p$pos + 1L]]$type != "LPAREN") return(NULL)

  func <- .advance(p)$value
  .advance(p)  # '('
  col_tok <- .peek(p)
  if (!col_tok$type %in% c("CODE", "IDENT")) {
    .error(p, sprintf("Expected a column name inside %s(...), got %s",
                      func, deparse(col_tok$value)))
  }
  column <- .advance(p)$value
  if (.peek(p)$type != "RPAREN") {
    .error(p, sprintf("Expected ')' after column name in %s(...), got %s",
                      func, deparse(.peek(p)$value)))
  }
  .advance(p)  # ')'
  op_tok <- .peek(p)
  if (op_tok$type != "OP" ||
      !(op_tok$value %in% c(">", "<", ">=", "<=", "==", "!="))) {
    .error(p, sprintf(
      "Aggregate %s(%s) must be followed by a comparison (>, <, >=, <=, ==, !=), got %s",
      func, column, deparse(op_tok$value)))
  }
  op <- .advance(p)$value
  val_tok <- .peek(p)
  if (!(val_tok$type %in% c("INT", "FLOAT"))) {
    .error(p, sprintf("Expected a numeric threshold after '%s', got %s",
                      op, deparse(val_tok$value)))
  }
  .advance(p)
  new_aggregate_expr(func, column, op, as.numeric(val_tok$value))
}

.is_comparison_ahead <- function(p) {
  if (p$pos + 2L > length(p$tokens)) return(FALSE)
  t0 <- p$tokens[[p$pos]]
  t1 <- p$tokens[[p$pos + 1L]]
  if (!(t0$type == "CODE" && t1$type == "OP")) return(FALSE)
  if (!(t1$value %in% c(">", "<", ">=", "<=", "==", "!="))) return(FALSE)
  !(is.character(t0$value) && tolower(t0$value) %in% .KEYWORDS)
}

.parse_comparison <- function(p) {
  col_tok <- .advance(p)
  op_tok  <- .advance(p)
  val_tok <- .peek(p)
  if (!val_tok$type %in% c("INT", "FLOAT")) {
    .error(p, sprintf("Expected number after '%s', got %s", op_tok$value, deparse(val_tok$value)))
  }
  .advance(p)
  new_comparison_atom(col_tok$value, op_tok$value, as.numeric(val_tok$value))
}

.parse_code_expr <- function(p) {
  codes <- character()
  codes <- c(codes, .parse_code_item(p))
  while (.peek(p)$type == "COMMA") {
    .advance(p)
    codes <- c(codes, .parse_code_item(p))
  }

  COL_TYPES <- c("CODE", "IDENT", "STAR_CODE", "RANGE_CODE", "SLICE_CODE")
  columns <- NULL
  if (.at_keyword(p, "in")) {
    .advance(p)
    cols <- character()
    tok <- .peek(p)
    if (!tok$type %in% COL_TYPES) {
      .error(p, sprintf("Expected column name after 'in', got %s", deparse(tok$value)))
    }
    cols <- c(cols, .advance(p)$value)
    while (.peek(p)$type == "COMMA") {
      .advance(p)
      tok <- .peek(p)
      if (!tok$type %in% COL_TYPES) {
        .error(p, sprintf("Expected column name, got %s", deparse(tok$value)))
      }
      cols <- c(cols, .advance(p)$value)
    }
    columns <- cols
  }
  new_code_atom(codes, columns)
}

.parse_code_item <- function(p) {
  tok <- .peek(p)
  if (tok$type == "AT") {
    .advance(p)
    name_tok <- .peek(p)
    if (!name_tok$type %in% c("CODE", "IDENT")) {
      .error(p, sprintf("Expected variable name after '@', got %s", deparse(name_tok$value)))
    }
    return(paste0("@", .advance(p)$value))
  }
  if (tok$type %in% c("STAR_CODE", "RANGE_CODE")) return(.advance(p)$value)
  if (tok$type == "CODE") {
    v <- tok$value
    if (is.character(v) && tolower(v) %in% .KEYWORDS) {
      .error(p, sprintf("Expected a code, got keyword '%s'. Use parentheses if you mean a code that matches a keyword name.", v))
    }
    return(.advance(p)$value)
  }
  .error(p, sprintf("Expected a code expression, got %s", deparse(tok$value)))
}
