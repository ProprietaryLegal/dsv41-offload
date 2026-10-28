"""Fault-oriented, CPU-only regression checks for documentation validation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.check_docs import check_briefs, check_links, check_window4, heading_ids, links


COMPARISON = """## Window 4 (date)
### W4-sync: controls
| short | a | b | −3.7 % | c | d |
| c32 | a | b | −3.0 % | c | d |
| conc | a | b | −1.0 % | c | d |
| c150 | a | b | −4.0 % | c | d |
### W4-E1p: process
| short | a | b | −6.3 % | c | d |
| c32 | a | b | −5.2 % | c | d |
| conc | a | b | −3.9 % | c | d |
| c150 | a | b | −5.1 % | c | d |
### W4-E0: sync
| short | a | b | −1.8 % | c | d |
| c32 | a | b | −0.8 % | c | d |
| conc | a | b | −1.4 % | c | d |
| c150 | a | b | −2.0 % | c | d |
## Window 5 (date)
"""


class DocumentationChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_missing_paths_and_fragments_fail_with_location(self) -> None:
        target = self.write("guide.md", "# A & B: `code` (one)\n\n# Repeated\n# Repeated\n")
        source = self.write("index.md", "[good](guide.md#a--b-code-one)\n[duplicate](guide.md#repeated-1)\n[bad](missing.md)\n[bad fragment](guide.md#no-such-heading)\n")
        result = check_links([source, target])
        self.assertEqual([r.line for r in result], [3, 4])
        self.assertIn("missing relative target", result[0].message)
        self.assertIn("missing heading fragment", result[1].message)

    def test_references_images_spaces_escapes_and_balanced_parentheses(self) -> None:
        for name in ("space name.md", "plot.png", "v(one).md"):
            self.write(name, "# Title\n")
        source = self.write("README.md", "[spaces](<space name.md#title> \"caption\")\n![plot][image]\n[paired](v(one).md)\n[escaped](v\\(one\\).md)\n[short]\n\n[image]: plot.png\n[short]: <space%20name.md#title>\n")
        self.assertFalse(check_links([source]))
        self.assertGreaterEqual(len(links(source.read_text())), 5)

    def test_code_comments_external_and_absolute_links_are_not_local_errors(self) -> None:
        source = self.write("README.md", "```md\n[x](absent.md)\n```\n`[x](absent.md)`\n<!-- [x](absent.md) -->\n[x](https://example.invalid/no)\n[x](mailto:x@example.invalid)\n[x](/host/local/file)\n\\[x](absent.md)\n")
        self.assertFalse(check_links([source]))

    def test_html_local_links_and_explicit_anchors(self) -> None:
        self.write("guide.md", '<a id="custom"></a>\n# Guide\n')
        source = self.write("README.md", '<a href="guide.md#custom">ok</a>\n<img src="missing.png">\n')
        findings = check_links([source])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line, 2)

    def test_unmatched_outer_bracket_cannot_hide_a_broken_link(self) -> None:
        source = self.write("README.md", "An unfinished [note\n\n[broken](missing.md)\n")
        findings = check_links([source])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].line, 3)

    def test_heading_ids_remain_unique_across_explicit_suffixes(self) -> None:
        self.assertEqual(heading_ids("# Same\n# Same\n# Same-1\n"), {"same", "same-1", "same-1-1"})

    def test_same_file_query_encoded_fragment_setext_and_code_heading(self) -> None:
        source = self.write("README.md", "Title\n=====\n\n# `Code` name\n\n[x](#title)\n[y](README.md?raw=1#code%20name)\n")
        # A URI fragment uses the actual generated slug, not heading display text.
        self.assertEqual(len(check_links([source])), 1)
        self.assertIn("code-name", heading_ids(source.read_text()))

    def test_brief_must_be_a_real_index_link(self) -> None:
        self.write("documents/plans/lanes/NEW.md", "# New\n")
        index = self.write("documents/README.md", "`plans/lanes/NEW.md`\n")
        self.assertEqual(len(check_briefs(self.root)), 1)
        index.write_text("[New](plans/lanes/NEW.md)\n")
        self.assertFalse(check_briefs(self.root))

    def test_corrected_ranges_pass_but_stale_arm_specific_ranges_fail(self) -> None:
        self.write("offload/results/COMPARISON.md", COMPARISON)
        readme = self.write("README.md", "Window 4: one-card async −1.0 to −4.0 %, TP2 sync −0.8 to −2.0 %, TP2 + async −3.9 to −6.3 %.\n")
        self.assertFalse(check_window4(self.root, [readme]))
        readme.write_text("Window 4: one-card async −0.9 to −3.9 %.\n")
        self.assertEqual(len(check_window4(self.root, [readme])), 1)
        readme.write_text("Window 4: one-card async −3.9 %.\n")
        self.assertEqual(len(check_window4(self.root, [readme])), 1)

    def test_stale_single_and_wrapped_promotion_claim_fail(self) -> None:
        self.write("offload/results/COMPARISON.md", COMPARISON)
        readme = self.write("README.md", "What window 4 measured: TP2 worker + process proxy + nap:\n−3.9 to −6.3 % per round.\n")
        self.assertFalse(check_window4(self.root, [readme]))
        readme.write_text("Window 4: TP2 + async −6.5 %.\n")
        self.assertEqual(len(check_window4(self.root, [readme])), 1)

    def test_unrelated_kernel_percentages_are_not_window4_claims(self) -> None:
        self.write("offload/results/COMPARISON.md", COMPARISON)
        readme = self.write("README.md", "Native kernel −6.5 % on a different workload.\n\nWindow 4 report is linked elsewhere.\n")
        self.assertFalse(check_window4(self.root, [readme]))

    def test_unsigned_faster_claims_are_checked_too(self) -> None:
        self.write("offload/results/COMPARISON.md", COMPARISON)
        readme = self.write("README.md", "Window 4: TP2 + async 3.9 to 6.3% faster.\n")
        self.assertFalse(check_window4(self.root, [readme]))
        readme.write_text("Window 4: TP2 + async 6.5% faster.\n")
        self.assertEqual(len(check_window4(self.root, [readme])), 1)

    def test_missing_canonical_table_is_a_loud_failure(self) -> None:
        self.write("offload/results/COMPARISON.md", "No Window-4 evidence.\n")
        result = check_window4(self.root, [])
        self.assertEqual(len(result), 1)
        self.assertIn("cannot read canonical", result[0].message)


if __name__ == "__main__":
    unittest.main()
