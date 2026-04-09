"""Tests for string manipulation helpers."""

import pandas as pd
import pytest

from tquery._stringops import (
    del_repeats,
    del_singles,
    interleave_strings,
    left_justify,
    overlay_strings,
    shorten,
)


class TestInterleaveStrings:
    def test_two_columns(self):
        df = pd.DataFrame({"i": ["i  i"], "a": [" a  "]})
        result = interleave_strings(df, time_sep="|")
        # Each position interleaves chars: (i,' '), (' ',a), (' ',' '), (i,' ')
        assert result.iloc[0] == "i | a|  |i "

    def test_no_sep(self):
        df = pd.DataFrame({"i": ["i  i"], "a": [" a  "]})
        result = interleave_strings(df, time_sep="")
        assert result.iloc[0] == "i  a  i "

    def test_unequal_lengths(self):
        df = pd.DataFrame({"x": ["ab"], "y": ["cde"]})
        result = interleave_strings(df, time_sep="|", no_event=" ")
        # a+c, b+d, ' '+e
        assert result.iloc[0] == "ac|bd| e"

    def test_multiple_persons(self):
        df = pd.DataFrame({"x": ["ab", "cd"], "y": ["ef", "gh"]})
        result = interleave_strings(df, time_sep="|")
        assert result.iloc[0] == "ae|bf"
        assert result.iloc[1] == "cg|dh"


class TestOverlayStrings:
    def test_no_collision(self):
        df = pd.DataFrame({"x": ["a  "], "y": [" b "]})
        result = overlay_strings(df, no_event=" ")
        assert result.iloc[0] == "ab "

    def test_with_collision(self):
        df = pd.DataFrame({"x": ["a a"], "y": ["ba "]})
        result = overlay_strings(df, collision="x", no_event=" ")
        # pos 0: a+b → collision, pos 1: ' '+a → a, pos 2: a+' ' → a
        assert result.iloc[0] == "xaa"

    def test_all_empty(self):
        df = pd.DataFrame({"x": ["   "], "y": ["   "]})
        result = overlay_strings(df, no_event=" ")
        assert result.iloc[0] == "   "


class TestLeftJustify:
    def test_pads_to_max(self):
        s = pd.Series(["ab", "abcde", "a"])
        result = left_justify(s)
        assert all(result.str.len() == 5)
        assert result.iloc[0] == "ab   "
        assert result.iloc[2] == "a    "


class TestShorten:
    def test_agg_3(self):
        s = pd.Series(["i  i  a  "])
        result = shorten(s, agg=3, no_event=" ")
        # 'i  ' → 'i', 'i  ' → 'i', 'a  ' → 'a'
        assert result.iloc[0] == "iia"

    def test_agg_2(self):
        s = pd.Series(["i ia  "])
        result = shorten(s, agg=2, no_event=" ")
        # 'i ' → 'i', 'ia' → 'i', '  ' → ' '
        assert result.iloc[0] == "ii "

    def test_all_empty(self):
        s = pd.Series(["      "])
        result = shorten(s, agg=3, no_event=" ")
        assert result.iloc[0] == "  "


class TestDelRepeats:
    def test_basic(self):
        s = pd.Series(["aaabbc", "iiiai"])
        result = del_repeats(s)
        assert result.iloc[0] == "abc"
        assert result.iloc[1] == "iai"

    def test_uppercase(self):
        s = pd.Series(["AAABB"])
        result = del_repeats(s)
        assert result.iloc[0] == "AB"

    def test_no_repeats(self):
        s = pd.Series(["abc"])
        result = del_repeats(s)
        assert result.iloc[0] == "abc"


class TestDelSingles:
    def test_basic(self):
        s = pd.Series(["aabca"])
        result = del_singles(s)
        assert result.iloc[0] == "aa"

    def test_all_repeated(self):
        s = pd.Series(["aabb"])
        result = del_singles(s)
        assert result.iloc[0] == "aabb"

    def test_single_char(self):
        s = pd.Series(["a"])
        result = del_singles(s)
        assert result.iloc[0] == ""

    def test_empty(self):
        s = pd.Series([""])
        result = del_singles(s)
        assert result.iloc[0] == ""
