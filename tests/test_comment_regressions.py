"""Regression tests for opening-comment extraction across every supported language.

Each case below is a crafted input the extractor gets wrong today. The expected
values follow the opening-comment contract in ml4setk's
``docs/comment_extractor.md`` (skip a hashbang, find the first real comment near
the top of the file, expand across contiguous comments until code appears) and
each language's own comment syntax.

Known failures are marked ``expectedFailure`` so the suite stays green. When a
fix lands, unittest reports an "unexpected success": remove the marker
(or the language from the known-failure set) to lock the fix in.
"""

from __future__ import annotations

import unittest
from typing import Callable

from ml4setk.Parsing.Comments.registry import COMMENT_SYNTAXES

from commentminer.extractors import ML4SEOpeningCommentExtractor
from commentminer.models import InputRecord


def _extract(language: str, content: str) -> str | None:
    record = InputRecord(
        dataset="the-stack",
        record_id="regression",
        content=content,
        metadata={"selected_language": language},
    )
    return ML4SEOpeningCommentExtractor().extract_opening_comment(record)


# (test name, registry language, source text, expected opening comment, why it is a bug)
LANGUAGE_CASES: tuple[tuple[str, str, str, str | None, str], ...] = (
    # c_style family: these languages allow nested /* */ comments.
    ("rust_nested_block", "rust", "/* outer /* inner */ still outer */\nfn main() {}\n",
     "/* outer /* inner */ still outer */", "Rust block comments nest."),
    ("swift_nested_block", "swift", "/* outer /* inner */ still outer */\nlet x = 1\n",
     "/* outer /* inner */ still outer */", "Swift block comments nest."),
    ("kotlin_nested_block", "kotlin", "/* a /* b */ c */\nfun main() {}\n",
     "/* a /* b */ c */", "Kotlin block comments nest."),
    ("scala_nested_block", "scala", "/* a /* b */ c */\nobject A\n",
     "/* a /* b */ c */", "Scala block comments nest."),
    ("dart_nested_block", "dart", "/* a /* b */ c */\nvoid main() {}\n",
     "/* a /* b */ c */", "Dart block comments nest."),
    ("odin_nested_block", "odin", "/* a /* b */ c */\npackage main\n",
     "/* a /* b */ c */", "Odin block comments nest."),
    ("rust_inner_attribute_is_not_hashbang", "rust", "#![no_std]\n// crate header\nfn main() {}\n",
     None, "#![...] is a Rust inner attribute (code), not a hashbang to skip."),
    ("c_line_comment_continuation", "c", "// header \\\n   continued\nint x;\n",
     "// header \\\n   continued", "Backslash-newline splices the next line into a C line comment."),
    ("vue_html_comment", "vue", "<!-- header -->\n<template></template>\n",
     "<!-- header -->", "Vue single-file components use HTML comments at top level."),
    ("sass_indented_block", "sass", "/* header\n   continues\n.a\n  color: red\n",
     "/* header\n   continues", "Indented Sass /* comments close by dedent, not */."),
    # hash_line_style
    ("makefile_comment_continuation", "makefile", "# header \\\n  continued\nall:\n",
     "# header \\\n  continued", "Make continues a comment across a trailing backslash."),
    ("tcl_comment_continuation", "tcl", "# header \\\n  continued\nputs hi\n",
     "# header \\\n  continued", "Tcl continues a comment across a trailing backslash."),
    # hash_style
    ("python_single_quote_docstring", "python", "'''Module docstring.'''\nimport os\n",
     "'''Module docstring.'''", "''' docstrings are equivalent to the supported \"\"\" form."),
    ("python_raw_docstring", "python", 'r"""raw doc"""\nx = 1\n',
     'r"""raw doc"""', "Prefixed (raw) docstrings are still module docstrings."),
    ("nix_block_comment", "nix", "/* header */\n{ pkgs }: pkgs.hello\n",
     "/* header */", "Nix supports /* */ block comments."),
    # dash_style
    ("vhdl_2008_block_comment", "vhdl", "/* header */\nentity e is end;\n",
     "/* header */", "VHDL-2008 adds /* */ block comments."),
    ("asn1_comment_closed_by_double_dash", "asn1", "-- header -- Foo ::= INTEGER\n",
     "-- header --", "An ASN.1 -- comment ends at the next -- on the same line."),
    # nested_dash_style / haskell_style
    ("agda_line_comment_containing_block_opener", "agda", "-- uses {- syntax\n{- header -}\nmodule M where\n",
     "-- uses {- syntax\n{- header -}", "{- inside a line comment must not open a nested block."),
    ("agda_pragma_is_not_comment", "agda", "{-# OPTIONS --safe #-}\nmodule M where\n",
     None, "{-# ... #-} is a pragma, not a comment."),
    ("haskell_pragma_is_not_comment", "haskell", "{-# LANGUAGE GADTs #-}\nmodule Main where\n",
     None, "{-# ... #-} is a pragma, not a comment."),
    ("haskell_line_comment_containing_block_opener", "haskell", "-- uses {- syntax\n{- block header -}\nmodule X where\n",
     "-- uses {- syntax\n{- block header -}", "{- inside a line comment must not open a nested block."),
    # semicolon_style
    ("lisp_block_comment", "lisp", "#| header |#\n(defun f ())\n",
     "#| header |#", "Common Lisp supports #| |# block comments."),
    ("lisp_nested_block_comment", "lisp", "#| outer #| inner |# outer |#\n(defun f ())\n",
     "#| outer #| inner |# outer |#", "Common Lisp #| |# comments nest."),
    ("scheme_block_comment", "scheme", "#| header |#\n(define x 1)\n",
     "#| header |#", "R7RS Scheme supports #| |# block comments."),
    # nested_star_style
    ("ocaml_string_inside_comment", "ocaml", '(* header "*)" still *)\nlet x = 1\n',
     '(* header "*)" still *)', "OCaml lexes string literals inside comments."),
    # d_doc_style
    ("d_plain_line_comment", "d", "// header\nvoid main() {}\n",
     "// header", "D supports plain // comments, not only ///."),
    ("d_plain_block_comment", "d", "/* header */\nvoid main() {}\n",
     "/* header */", "D supports plain /* */ comments, not only /** */."),
    ("d_nested_plus_comment", "d", "/+ outer /+ inner +/ outer +/\nvoid main() {}\n",
     "/+ outer /+ inner +/ outer +/", "D /+ +/ comments nest and need not start with /++."),
    # fsharp_style
    ("fsharp_line_comment_containing_block_opener", "f#", "// uses (* here\n(* header *)\nlet x = 1\n",
     "// uses (* here\n(* header *)", "(* inside a line comment must not open a nested block."),
    ("ats_block_comment", "ats", "/* header */\nval x = 1\n",
     "/* header */", "ATS supports /* */ block comments."),
    # forth_style
    ("forth_paren_comment_without_trailing_space", "forth", "( header)\n: foo ;\n",
     "( header)", "Forth ( comments end at the first ), no space required before it."),
    # bang_style
    ("fortran_fixed_form_comment", "fortran", "C     fixed form header\n      PROGRAM X\n",
     "C     fixed form header", "Fixed-form Fortran uses C in column 1 for comments."),
    # julia_style
    ("julia_nested_block_comment", "julia", "#=\nouter #= inner =#\nstill outer\n=#\nx = 1\n",
     "#=\nouter #= inner =#\nstill outer\n=#", "Julia #= =# comments nest."),
    # matlab_style
    ("matlab_nested_block_comment", "matlab", "%{\nouter\n%{\ninner\n%}\nouter\n%}\nx = 1;\n",
     "%{\nouter\n%{\ninner\n%}\nouter\n%}", "MATLAB %{ %} block comments nest."),
    ("matlab_inline_percent_brace_is_line_comment", "matlab", "%{ not a block\nx = 1;\n%}\n",
     "%{ not a block", "%{ only opens a block when it is alone on its line."),
    # lua_style
    ("lua_long_bracket_level", "lua", "--[==[\nheader ]] still header\n]==]\nlocal x = 1\n",
     "--[==[\nheader ]] still header\n]==]", "Lua long comments may use --[==[ ... ]==] levels."),
    # perl_style / ruby_style
    ("perl_pod_cut_must_start_line", "perl", "=head1 NAME\n\nUse x=cut-off\n\n=cut\nuse strict;\n",
     "=head1 NAME\n\nUse x=cut-off\n\n=cut", "=cut only ends POD at the start of a line."),
    ("ruby_end_must_start_line", "ruby", "=begin\nsee x=end-y\n=end\nputs 1\n",
     "=begin\nsee x=end-y\n=end", "=end only closes =begin at the start of a line."),
    # raku_style
    ("raku_embedded_comment_backtick", "raku", "#`( embedded\nmulti-line )\nsay 1;\n",
     "#`( embedded\nmulti-line )", "Raku embedded comments are #`( ... ), which may span lines."),
    # sql_style
    ("tsql_nested_block", "tsql", "/* outer /* inner */ outer */\nSELECT 1;\n",
     "/* outer /* inner */ outer */", "T-SQL block comments nest."),
    ("plpgsql_nested_block", "plpgsql", "/* outer /* inner */ outer */\nSELECT 1;\n",
     "/* outer /* inner */ outer */", "PostgreSQL block comments nest."),
    # webassembly_style
    ("webassembly_nested_block", "webassembly", "(; outer (; inner ;) outer ;)\n(module)\n",
     "(; outer (; inner ;) outer ;)", "WebAssembly text-format block comments nest."),
    # batchfile_style
    ("batchfile_at_rem", "batchfile", "@REM header\necho hi\n",
     "@REM header", "@REM is the common echo-suppressed REM comment."),
    # quote_line_style
    ("smalltalk_comment_ends_at_quote", "smalltalk", '"header" Object subclass: #Foo.\n',
     '"header"', "Smalltalk comments end at the closing double quote."),
    ("smalltalk_multiline_comment", "smalltalk", '"multi\nline"\nObject subclass: #Foo.\n',
     '"multi\nline"', "Smalltalk double-quoted comments can span lines."),
    # autoit_style
    ("autoit_long_comment_directives", "autoit", "#comments-start\nheader\n#comments-end\nMsgBox(0, 1, 2)\n",
     "#comments-start\nheader\n#comments-end", "#comments-start/#comments-end are aliases of #cs/#ce."),
    # applescript_style
    ("applescript_line_comment_containing_block_opener", "applescript", "-- uses (* here\n(* header *)\ndisplay dialog 1\n",
     "-- uses (* here\n(* header *)", "(* inside a line comment must not open a nested block."),
    # hash_pipe_style
    ("racket_line_comment_containing_block_opener", "racket", "; uses #| here\n#| header |#\n(define x 1)\n",
     "; uses #| here\n#| header |#", "#| inside a line comment must not open a nested block."),
    # nim_style
    ("nim_doc_block_comment", "nim", "##[\nheader\n]##\necho 1\n",
     "##[\nheader\n]##", "Nim ##[ ]## is a multi-line documentation comment."),
    # freebasic_style
    ("freebasic_nested_block", "freebasic", "/' outer /' inner '/ outer '/\nPrint 1\n",
     "/' outer /' inner '/ outer '/", "FreeBASIC /' '/ comments nest."),
    # cmake_style
    ("cmake_bracket_comment_with_equals", "cmake", "#[=[\nheader\n]=]\nproject(x)\n",
     "#[=[\nheader\n]=]", "CMake bracket comments may use #[=[ ... ]=]."),
    # freemarker_style
    ("freemarker_square_bracket_syntax", "freemarker", "[#-- header --]\n[#assign x = 1]\n",
     "[#-- header --]", "FreeMarker's square-bracket syntax uses [#-- --]."),
    # sas_style
    ("sas_statement_comment_spans_lines", "sas", "* header\n  continues;\ndata x;\n",
     "* header\n  continues;", "A SAS * comment statement runs to the next semicolon."),
    # haml_style
    ("haml_silent_comment_block", "haml", "-# header\n   continues\n%p hi\n",
     "-# header\n   continues", "Lines indented under -# belong to the silent comment."),
    # lean_style
    ("lean_line_comment_containing_block_opener", "lean", "-- uses /- here\n/- header -/\ndef x := 1\n",
     "-- uses /- here\n/- header -/", "/- inside a line comment must not open a nested block."),
    # liquid_style
    ("liquid_whitespace_control_comment", "liquid", "{%- comment -%}\nheader\n{%- endcomment -%}\n<p>x</p>\n",
     "{%- comment -%}\nheader\n{%- endcomment -%}", "Liquid tags accept {%- -%} whitespace control."),
    # restructuredtext_style
    ("restructuredtext_directive_is_not_comment", "restructuredtext", ".. image:: logo.png\n\nTitle\n=====\n",
     None, ".. name:: is a directive, not a comment."),
    # stata_style
    ("stata_nested_block", "stata", "/* outer /* inner */ outer */\ndisplay 1\n",
     "/* outer /* inner */ outer */", "Stata /* */ comments nest."),
    # coldfusion_style
    ("coldfusion_cfc_script_comment", "coldfusion_cfc", "/** header */\ncomponent {}\n",
     "/** header */", "Script-syntax CFCs use /* */ and // comments."),
    # inform7_style
    ("inform7_nested_brackets", "inform_7", "[outer [inner] outer]\nThe Kitchen is a room.\n",
     "[outer [inner] outer]", "Inform 7 [ ] comments nest."),
    # tla_style
    ("tla_line_comment_containing_block_opener", "tla", "\\* uses (* here\n(* header *)\n---- MODULE M ----\n",
     "\\* uses (* here\n(* header *)", "(* inside a line comment must not open a nested block."),
    # nested_c_style
    ("dafny_line_comment_containing_block_opener", "dafny", "// glob src/*.dfy\n/* header */\nmethod M() {}\n",
     "// glob src/*.dfy\n/* header */", "/* inside a line comment must not open a nested block."),
    # gerber_style
    ("gerber_comment_ends_at_asterisk", "gerber_image", "G04 header*D10*\n",
     "G04 header*", "A G04 comment ends at its terminating *."),
    # asciidoc_style
    ("asciidoc_longer_comment_delimiter", "asciidoc", "//////\nheader //// inner\n//////\n= Title\n",
     "//////\nheader //// inner\n//////", "A comment block closes only on a delimiter matching its opener."),
)


def _make_language_case(language: str, content: str, expected: str | None, why: str) -> Callable:
    def test(self: unittest.TestCase) -> None:
        self.assertEqual(_extract(language, content), expected, why)

    return unittest.expectedFailure(test)


class LanguageSyntaxRegressionTests(unittest.TestCase):
    """Hand-crafted inputs that violate a specific language's comment syntax."""

    @unittest.expectedFailure
    def test_digital_command_language_dollar_prefixed_comment(self) -> None:
        # DCL command lines start with "$", so "$!" is the usual comment form.
        match = _extract("digital_command_language", "$! header\n$ set verify\n")
        self.assertIsNotNone(match)
        self.assertTrue(match.endswith("! header"))

    @unittest.expectedFailure
    def test_roff_dot_backslash_quote_comment(self) -> None:
        # Man pages conventionally start with `.\"` comment lines.
        match = _extract("roff", '.\\" header\n.TH FOO 1\n')
        self.assertIsNotNone(match)
        self.assertTrue(match.endswith('\\" header'))


for _name, _language, _content, _expected, _why in LANGUAGE_CASES:
    setattr(
        LanguageSyntaxRegressionTests,
        f"test_{_name}",
        _make_language_case(_language, _content, _expected, _why),
    )


def _seeded_opening_comment(syntax) -> str:
    examples = (
        syntax.canonical_regex_examples
        + syntax.shared_regex_examples
        + syntax.canonical_nested_examples
        + syntax.shared_nested_examples
    )
    example = next((item for item in examples if item.kind == "line"), examples[0])
    return example.expected_match


# Block-only languages whose comments do not end with a `.*` regex, so CRLF
# input does not leak a trailing "\r" into the match.
CRLF_SAFE_LANGUAGES = frozenset(
    {
        "ant_build_system", "antlers", "asl", "augeas", "blade", "collada",
        "coldfusion", "coldfusion_cfc", "coq", "css", "e_mail", "ejs",
        "freemarker", "groovy_server_pages", "handlebars", "html", "html_ecr",
        "html_eex", "html_erb", "html_razor", "inform_7", "isabelle",
        "isabelle_root", "jsoniq", "jsp", "latte", "liquid", "mathematica",
        "maven_pom", "mustache", "nunjucks", "ocaml", "postcss", "rexx",
        "rmarkdown", "smarty", "svelte", "svg", "twig", "wikitext", "xml",
        "xml_property_list", "xproc", "xquery", "xslt",
    }
)


def _make_baseline_case(language: str, comment: str) -> Callable:
    def test(self: unittest.TestCase) -> None:
        self.assertEqual(_extract(language, comment + "\nsuffix\n"), comment)

    return test


def _make_bom_case(language: str, comment: str) -> Callable:
    def test(self: unittest.TestCase) -> None:
        # Files with a UTF-8 byte order mark decode with a leading U+FEFF.
        self.assertEqual(_extract(language, "﻿" + comment + "\nsuffix\n"), comment)

    return unittest.expectedFailure(test)


def _make_crlf_case(language: str, comment: str) -> Callable:
    def test(self: unittest.TestCase) -> None:
        # Windows line endings must not leave a trailing "\r" in the comment.
        content = (comment + "\nsuffix\n").replace("\n", "\r\n")
        self.assertEqual(_extract(language, content), comment.replace("\n", "\r\n"))

    if language in CRLF_SAFE_LANGUAGES:
        return test
    return unittest.expectedFailure(test)


class AllLanguagesRegressionTests(unittest.TestCase):
    """One seeded comment per registry language, wrapped in hostile file encodings."""


for _syntax in COMMENT_SYNTAXES:
    _comment = _seeded_opening_comment(_syntax)
    for _language in _syntax.language_names:
        _slug = "".join(ch if ch.isalnum() else "_" for ch in _language)
        if _language in {"c#", "f#", "c++"}:
            _slug = _language.replace("#", "sharp").replace("+", "plus")
        setattr(AllLanguagesRegressionTests, f"test_{_slug}_baseline", _make_baseline_case(_language, _comment))
        setattr(AllLanguagesRegressionTests, f"test_{_slug}_utf8_bom", _make_bom_case(_language, _comment))
        setattr(AllLanguagesRegressionTests, f"test_{_slug}_crlf", _make_crlf_case(_language, _comment))


class LanguageResolutionRegressionTests(unittest.TestCase):
    """Failures in CommentMiner's own mapping from dataset language names to registry keys."""

    @unittest.expectedFailure
    def test_unresolved_candidate_cache_does_not_hide_later_candidates(self) -> None:
        # The first record resolves "c-sharp" -> None, caches that, then falls
        # through to lang="C#". The second record hits the cached None and
        # returns early without trying the remaining candidates.
        extractor = ML4SEOpeningCommentExtractor()
        matches = [
            extractor.extract_opening_comment(
                InputRecord(
                    dataset="the-stack",
                    record_id=str(index),
                    content="// header\nclass A {}\n",
                    language="C#",
                    metadata={"selected_language": "c-sharp", "ext": "cs", "lang": "C#"},
                )
            )
            for index in range(2)
        ]
        self.assertEqual(matches, ["// header", "// header"])


def _make_resolution_case(stack_language: str, registry_language: str) -> Callable:
    def test(self: unittest.TestCase) -> None:
        record = InputRecord(
            dataset="the-stack",
            record_id="r",
            content="",
            metadata={"selected_language": stack_language},
        )
        self.assertEqual(ML4SEOpeningCommentExtractor()._resolve_language(record), registry_language)

    return unittest.expectedFailure(test)


# The Stack directory names whose alias target is not a registry key.
STACK_LANGUAGE_NAMES = (
    ("c-sharp", "c#"),
    ("f-sharp", "f#"),
    ("visual-basic", "visual_basic_net"),
    ("java-server-pages", "jsp"),
    ("objective-c++", "objective_cpp"),
)

for _stack_language, _registry_language in STACK_LANGUAGE_NAMES:
    _slug = "".join(ch if ch.isalnum() else "_" for ch in _stack_language).replace("__", "_plusplus")
    setattr(
        LanguageResolutionRegressionTests,
        f"test_stack_name_{_slug}_resolves",
        _make_resolution_case(_stack_language, _registry_language),
    )


if __name__ == "__main__":
    unittest.main()
