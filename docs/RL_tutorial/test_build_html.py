"""Check that Pages readers can reach tutorial source without private-repo access."""

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

import build_html


class PublicationLinksTest(unittest.TestCase):
    def test_source_link_preserves_query_and_line_anchor(self):
        result = build_html.rewrite_links(
            '<a href="../../scripts/rl_tutorial/plot_reward_curve.py?plain=1#L20">code</a>'
        )
        self.assertIn(
            f'href="{build_html.TUTORIAL_SOURCE}/scripts/rl_tutorial/plot_reward_curve.py?plain=1#L20"',
            result,
        )

    def test_raw_samples_open_json_instead_of_a_github_html_page(self):
        for suffix in ("json", "jsonl"):
            with self.subTest(suffix=suffix):
                result = build_html.rewrite_links(
                    f'<a href="../../scripts/rl_tutorial/sample_data/sample.{suffix}">data</a>'
                )
                self.assertIn(f'href="{build_html.TUTORIAL_RAW}/scripts/', result)

    def test_local_navigation_assets_and_framework_links_stay_local_or_pinned(self):
        unchanged = (
            '<a href="./alfworld_robot.html#step-2">player</a>'
            '<img src="./ch1_reward_curve.png">'
            '<a href="./build_html.py">builder</a>'
            '<a href="https://github.com/agentscope-ai/Trinity-RFT/commit/6513971">framework</a>'
        )
        self.assertEqual(build_html.rewrite_links(unchanged), unchanged)
        self.assertEqual(
            build_html.rewrite_links('<a href="./README.md">overview</a>'),
            '<a href="./index.html">overview</a>',
        )

    def test_all_rewritten_source_targets_exist_in_checkout(self):
        root = Path(__file__).resolve().parent
        repo = root.parents[1]
        checked = 0
        for page in build_html.CHAPTERS:
            body, _ = build_html.convert(build_html.preprocess((root / page["md"]).read_text()))
            result = build_html.rewrite_links(body)
            self.assertNotRegex(result, r'href="\.\./\.\./(?:scripts|examples)/')
            self.assertNotIn("github.com/agentscope-ai/agentic-rl", result)
            for link in re.findall(r'href="([^"]+)"', result):
                for prefix in (build_html.TUTORIAL_SOURCE, build_html.TUTORIAL_RAW):
                    if link.startswith(prefix + "/"):
                        path = unquote(urlsplit(link[len(prefix) + 1:]).path)
                        self.assertTrue((repo / path).is_file(), f'{page["md"]}: {link}')
                        checked += 1
        self.assertGreaterEqual(checked, 8)


if __name__ == "__main__":
    unittest.main()
