# R port of tquery/_codes.py — code resolution and row matching.
# v1: plain, wildcard (K50*), range (K50-K53), and @variable refs.
# Multi-column with sep is handled at the row-matching layer.

expand_codes <- function(pattern, all_codes = NULL, variables = list()) {
  # Variable reference
  if (startsWith(pattern, "@")) {
    name <- substr(pattern, 2L, nchar(pattern))
    if (!name %in% names(variables)) {
      stop(sprintf("Variable '%s' not found", name))
    }
    return(as.character(variables[[name]]))
  }

  # Wildcard
  if (endsWith(pattern, "*")) {
    prefix <- substr(pattern, 1L, nchar(pattern) - 1L)
    if (is.null(all_codes)) return(pattern)
    matched <- all_codes[startsWith(all_codes, prefix)]
    if (length(matched) == 0L) {
      stop(sprintf("Wildcard '%s' matched no codes in the dataset", pattern))
    }
    return(matched)
  }

  # Range (a-b, where a and b are letter-starting)
  if (grepl("-", pattern, fixed = TRUE) && !startsWith(pattern, "-")) {
    parts <- strsplit(pattern, "-", fixed = TRUE)[[1]]
    if (length(parts) == 2L && nzchar(parts[1]) && nzchar(parts[2])) {
      start <- parts[1]; end <- parts[2]
      if (is.null(all_codes)) return(pattern)
      matched <- all_codes[all_codes >= start & all_codes <= end]
      if (length(matched) == 0L) {
        stop(sprintf("Range '%s' matched no codes in the dataset", pattern))
      }
      return(matched)
    }
  }

  pattern
}

expand_all_codes <- function(patterns, all_codes = NULL, variables = list()) {
  out <- character()
  for (pat in patterns) {
    out <- c(out, expand_codes(pat, all_codes, variables))
  }
  unique(out)
}

# Resolve column patterns against the actual columns of the data.
resolve_columns <- function(patterns, all_columns) {
  out <- character()
  for (pat in patterns) {
    if (endsWith(pat, "*")) {
      prefix <- substr(pat, 1L, nchar(pat) - 1L)
      out <- c(out, all_columns[startsWith(all_columns, prefix)])
    } else if (grepl(":", pat, fixed = TRUE)) {
      parts <- strsplit(pat, ":", fixed = TRUE)[[1]]
      if (length(parts) == 2L && all(parts %in% all_columns)) {
        si <- which(all_columns == parts[1])
        ei <- which(all_columns == parts[2])
        out <- c(out, all_columns[si:ei])
      }
    } else if (grepl("-", pat, fixed = TRUE) && !startsWith(pat, "-")) {
      parts <- strsplit(pat, "-", fixed = TRUE)[[1]]
      if (length(parts) == 2L) {
        sorted_cols <- sort(all_columns)
        out <- c(out, sorted_cols[sorted_cols >= parts[1] & sorted_cols <= parts[2]])
      }
    } else if (pat %in% all_columns) {
      out <- c(out, pat)
    }
  }
  unique(out)
}

# Compute the row-level mask: TRUE for rows where ANY of the searched
# columns contains ANY of the codes. Wildcards inside `codes` survive as
# prefix tests applied directly to cell values.
get_matching_rows <- function(dt, codes, cols, sep = NULL) {
  mask <- rep(FALSE, nrow(dt))
  if (length(codes) == 0L) return(mask)

  wildcards <- codes[endsWith(codes, "*")]
  exact     <- codes[!endsWith(codes, "*")]

  for (col in cols) {
    if (!col %in% names(dt)) next
    values <- dt[[col]]
    if (is.null(sep)) {
      if (length(exact)) mask <- mask | (values %in% exact)
      for (w in wildcards) {
        prefix <- substr(w, 1L, nchar(w) - 1L)
        mask <- mask | (!is.na(values) & startsWith(as.character(values), prefix))
      }
    } else {
      # Cells contain a separator-joined list of codes.
      parts_per_row <- strsplit(as.character(values), sep, fixed = TRUE)
      row_hits <- vapply(parts_per_row, function(parts) {
        if (length(parts) == 0L) return(FALSE)
        parts <- trimws(parts)
        if (length(exact) && any(parts %in% exact)) return(TRUE)
        for (w in wildcards) {
          prefix <- substr(w, 1L, nchar(w) - 1L)
          if (any(startsWith(parts, prefix))) return(TRUE)
        }
        FALSE
      }, logical(1))
      mask <- mask | row_hits
    }
  }
  mask
}

# Collect every unique code seen across the configured columns. Used to
# resolve wildcards and ranges. With sep, splits cell values.
collect_unique_codes <- function(dt, cols, sep = NULL) {
  all <- character()
  for (col in cols) {
    if (!col %in% names(dt)) next
    vals <- as.character(dt[[col]])
    vals <- vals[!is.na(vals)]
    if (is.null(sep)) {
      all <- c(all, vals)
    } else {
      parts <- unlist(strsplit(vals, sep, fixed = TRUE), use.names = FALSE)
      all <- c(all, trimws(parts))
    }
  }
  sort(unique(all))
}
