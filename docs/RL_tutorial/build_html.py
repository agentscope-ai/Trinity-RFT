#!/usr/bin/env python3
"""把 docs/RL_tutorial/ 下的 markdown 教程构建成带 Tailwind CSS 样式的离线 HTML。

用法:
    python3 docs/RL_tutorial/build_html.py          # 用已有的 tailwind.min.css 生成 HTML
    python3 docs/RL_tutorial/build_html.py --css    # 同时重新编译 Tailwind CSS（需要 node/npx）

- README.md -> index.html，章节与 LoRA 运行参考 -> 同名 .html（输出在同一目录）
- Markdown 是正文来源；交互图由本地图解组件增强；改动后重跑本脚本即可
- 生成的 HTML 完全离线可用：Tailwind（含 typography 插件）预编译后内嵌（tailwind.min.css），
  字体用系统字体栈，无任何 CDN / 外部请求；图片（ch1_reward_curve.png）用相对路径
- 只改 md 内容时不需要 node；改了模板里的 class 时才需要 --css 重编译
- 依赖: pip install markdown pygments；--css 时另需 node >= 18

结构与 docs/RL_tutorial/build_html.py（copaw 云教程）保持一致，便于两站对照。
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import markdown
from build_alfworld_player import build_standalone, player_assets
from build_lora_primer import build_fallbacks, enhance_lora_figures, primer_assets
from markdown.extensions.toc import slugify_unicode
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent

GITHUB_URL = "https://github.com/agentscope-ai/Trinity-RFT"          # 框架（依赖）
TUTORIAL_REPO = f"{GITHUB_URL}/tree/tutorial"                       # 公开教程分支
TUTORIAL_SOURCE = f"{GITHUB_URL}/blob/tutorial"
TUTORIAL_RAW = "https://raw.githubusercontent.com/agentscope-ai/Trinity-RFT/tutorial"

CHAPTERS = [
    dict(md="README.md", out="index.html", num="§",
         nav="教程总览", mode="导读", time="3 分钟"),
    dict(md="ch0_什么是强化学习.md", num="0",
         nav="什么是强化学习？", mode="概念", time="10 分钟阅读"),
    dict(md="ch1_跑通.md", num="1",
         nav="跑通：先看 RL 真的有用", mode="黑盒", time="1 小时装环境 + 约 15 小时训练"),
    dict(md="ch2_rollout与experience.md", num="2",
         nav="rollout：state/action/reward", mode="拆解", time="15 分钟阅读"),
    dict(md="ch3_reward怎么算.md", num="3",
         nav="reward：环境怎么给分", mode="拆解", time="15 分钟阅读"),
    dict(md="ch4_advantage怎么来.md", num="4",
         nav="advantage：为什么不用 reward 直接训", mode="拆解", time="20 分钟阅读"),
    dict(md="ch5_loss与权重更新.md", num="5",
         nav="loss 与更新：advantage 变梯度", mode="拆解", time="20 分钟阅读"),
    dict(md="ch6_动手实验.md", num="6",
         nav="LoRA 实测与开放实验", mode="实验", time="看图理解，再动手运行"),
    dict(md="lora32_reproduction.md", num="▶",
         nav="LoRA32 运行参考", mode="参考", time="环境、配置、日志与恢复"),
]
for _c in CHAPTERS:
    _c.setdefault("out", _c["md"].replace(".md", ".html"))

MODE_STYLE = {
    "导读": "bg-slate-100 text-slate-700 ring-slate-200",
    "概念": "bg-teal-100 text-teal-800 ring-teal-200",
    "决策": "bg-amber-100 text-amber-800 ring-amber-200",
    "黑盒": "bg-indigo-100 text-indigo-800 ring-indigo-200",
    "拆解": "bg-sky-100 text-sky-800 ring-sky-200",
    "实验": "bg-emerald-100 text-emerald-800 ring-emerald-200",
    "参考": "bg-slate-100 text-slate-700 ring-slate-200",
}

PYGMENTS_CSS = HtmlFormatter(style="github-dark").get_style_defs(".highlight")

FONT_SANS = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', "
             "'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans SC', sans-serif")
FONT_MONO = ("ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, "
             "'Liberation Mono', monospace")

TAILWIND_CONFIG_JS = """
module.exports = {
  // scrollspy 在运行时给 TOC 链接加的 class，扫描不到，需要 safelist
  safelist: ['border-indigo-500', 'text-indigo-700', 'font-medium',
             'border-transparent', 'text-slate-500'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'PingFang SC',
               'Hiragino Sans GB', 'Microsoft YaHei', 'Noto Sans SC', 'sans-serif'],
        mono: ['ui-monospace', 'SF Mono', 'SFMono-Regular', 'Menlo', 'Consolas',
               'Liberation Mono', 'monospace'],
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
"""

BASE_CSS = ("html { scroll-behavior: smooth; }\n"
            "body { font-family: " + FONT_SANS + "; }\n"
            "code, pre, kbd { font-family: " + FONT_MONO + "; }\n") + """

.prose { line-height: 1.85; }
.prose h2 { scroll-margin-top: 5rem; border-bottom: 1px solid #e2e8f0; padding-bottom: .45rem; }
.prose h3, .prose h4 { scroll-margin-top: 5rem; }
.prose hr { margin: 2.75em 0; border-color: #e2e8f0; }

/* 原文入口显示为一行链接，展开后仍保留原有排版。 */
.prose details.tutorial-original {
  border: 0; border-radius: 0; background: transparent; padding: 0;
  margin: 1.25rem 0 2rem;
}
.prose details.tutorial-original > summary {
  display: inline-block; font-size: .9rem; font-weight: 500;
  color: #4f46e5; text-decoration: underline; text-underline-offset: 4px;
  cursor: pointer; list-style: none;
}
.prose details.tutorial-original > summary::-webkit-details-marker { display: none; }
.prose details.tutorial-original[open] > summary { margin-bottom: 1rem; }
.prose details.tutorial-original > summary:focus-visible { outline: 2px solid #6366f1; outline-offset: 5px; }

/* 行内代码：小圆角芯片，去掉 typography 默认的反引号 */
.prose :where(code):not(:where(pre code)) {
  background: #eef2ff; color: #4338ca; padding: .12em .42em;
  border-radius: .375rem; font-weight: 500; font-size: .85em;
}
.prose :where(code)::before, .prose :where(code)::after { content: none !important; }

/* 代码块：深色卡片（配 pygments github-dark） */
.prose .highlight { margin: 1.6em 0; }
.prose .highlight pre {
  background: #0d1117; color: #e6edf3; border-radius: .75rem;
  padding: 1.05rem 1.3rem; overflow-x: auto; font-size: .85em;
  line-height: 1.7; margin: 0; border: 1px solid #1f2937;
}
.highlight { background: transparent !important; }

/* 引用块：柔和的左边条 callout */
.prose blockquote {
  border-left: 4px solid #6366f1; background: #f5f6ff;
  border-radius: 0 .75rem .75rem 0; padding: .75rem 1.25rem;
  font-style: normal; color: #3f3f6e; font-weight: 400;
}
.prose blockquote p:first-of-type::before,
.prose blockquote p:last-of-type::after { content: none; }

/* 表格：圆角卡片 + 斑马纹 + 移动端横向滚动 */
.table-wrap {
  overflow-x: auto; border: 1px solid #e2e8f0; border-radius: .75rem;
  margin: 1.7em 0; background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.table-wrap table { margin: 0 !important; width: 100%; font-size: .9em; }
.prose thead { background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
.prose thead th { padding: .7rem 1rem; color: #334155; white-space: nowrap; }
.prose tbody td { padding: .62rem 1rem; vertical-align: top; }
.prose tbody tr { border-bottom: 1px solid #f1f5f9; }
.prose tbody tr:last-child { border-bottom: none; }
.prose tbody tr:nth-child(even) { background: #fafbfd; }

/* 折叠块 */
.prose details {
  border: 1px solid #e2e8f0; border-radius: .75rem;
  padding: .85rem 1.15rem; background: #fafafa; margin: 1.3em 0;
}
.prose summary { cursor: pointer; font-weight: 600; color: #4f46e5; }

.prose img {
  border-radius: .75rem; border: 1px solid #e2e8f0;
  box-shadow: 0 4px 14px rgba(15,23,42,.06);
}
.prose a { color: #4f46e5; text-decoration-color: #c7d2fe; text-underline-offset: 3px; }
.prose a:hover { color: #3730a3; }
"""

SCROLLSPY_JS = """
(function () {
  var links = [].slice.call(document.querySelectorAll('#toc a'));
  if (!links.length) return;
  var pairs = links.map(function (a) {
    var id = decodeURIComponent((a.getAttribute('href') || '').slice(1));
    var el = document.getElementById(id);
    return el ? { el: el, a: a } : null;
  }).filter(Boolean);
  var ACTIVE = ['border-indigo-500', 'text-indigo-700', 'font-medium'];
  function onScroll() {
    var y = window.scrollY + 96, cur = pairs[0];
    pairs.forEach(function (p) { if (p.el.offsetTop <= y) cur = p; });
    pairs.forEach(function (p) {
      var on = (p === cur);
      ACTIVE.forEach(function (c) { p.a.classList.toggle(c, on); });
      p.a.classList.toggle('border-transparent', !on);
      p.a.classList.toggle('text-slate-500', !on);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
"""


# ---------------------------------------------------------------- markdown 处理

def split_title(text):
    m = re.match(r"^#\s+(.+?)\s*\n", text)
    if not m:
        raise ValueError("找不到一级标题")
    return m.group(1).strip(), text[m.end():]


def strip_trailing_nav(text):
    """去掉文件末尾的『上一章 / 下一章 / 返回 / 开始阅读』手写导航（改由模板统一生成）。"""
    chunks = re.split(r"\n---+\n", text)
    last = chunks[-1]
    if len(chunks) > 1 and re.search(r"\*\*(上一章|下一章|返回|开始阅读)\*\*", last) \
            and len(last.strip()) < 400:
        chunks = chunks[:-1]
    return "\n---\n".join(chunks)


def preprocess(text):
    text = text.replace("<details>", '<details markdown="1">')
    # pygments 没有 jsonc lexer，用 js 高亮注释 + 对象字面量效果一致
    text = re.sub(r"```jsonc\b", "```js", text)
    return text


def convert(body):
    md = markdown.Markdown(
        extensions=["extra", "codehilite", "toc", "sane_lists"],
        extension_configs={
            "codehilite": {"css_class": "highlight", "guess_lang": False},
            "toc": {"slugify": slugify_unicode, "toc_depth": "2-4"},
        },
    )
    html = md.convert(body)
    return html, md.toc_tokens


def rewrite_links(html):
    for c in CHAPTERS:
        for prefix in ("./", ""):
            html = html.replace(f'href="{prefix}{c["md"]}"', f'href="./{c["out"]}"')
    # Pages publishes /docs only; scripts, examples and raw data live in the public branch.
    def source_link(match):
        path = match.group(1)
        is_raw_data = Path(urlsplit(path).path).suffix in {".json", ".jsonl"}
        source = TUTORIAL_RAW if is_raw_data else TUTORIAL_SOURCE
        return f'href="{source}/{path}"'

    return re.sub(r'href="\.\./\.\./((?:scripts|examples)/[^"]+)"', source_link, html)


def wrap_tables(html):
    html = html.replace("<table>", '<div class="table-wrap"><table>')
    html = html.replace("</table>", "</table></div>")
    return html


# ---------------------------------------------------------------- 页面组件

def mode_pill(mode, extra=""):
    style = MODE_STYLE.get(mode, MODE_STYLE["导读"])
    return (f'<span class="inline-flex items-center rounded-full px-3 py-1 text-xs '
            f'font-semibold ring-1 ring-inset {style} {extra}">{mode}</span>')


def sidebar(active_idx):
    items = []
    for i, c in enumerate(CHAPTERS):
        if i == active_idx:
            row = "bg-indigo-50 text-indigo-900 font-semibold"
            bubble = "bg-indigo-600 text-white"
        else:
            row = "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
            bubble = "bg-slate-200 text-slate-500 group-hover:bg-slate-300"
        items.append(
            f'<a href="./{c["out"]}" class="group flex items-center gap-3 rounded-lg '
            f'px-3 py-2 text-sm transition {row}">'
            f'<span class="flex h-6 w-6 shrink-0 items-center justify-center rounded-md '
            f'text-xs font-bold {bubble}">{c["num"]}</span>'
            f'<span class="truncate">{c["nav"]}</span></a>'
        )
    nav = "\n".join(items)
    return f"""
<aside class="hidden lg:block w-72 shrink-0">
  <div class="sticky top-0 h-screen overflow-y-auto py-8 pl-6 pr-4 flex flex-col">
    <a href="./index.html" class="block px-3">
      <div class="text-lg font-black tracking-tight text-slate-900">RL Tutorial</div>
      <div class="mt-0.5 text-xs text-slate-500">跑通并看懂强化学习 · 以 ALFWorld 为实验环境</div>
    </a>
    <nav class="mt-6 space-y-1">{nav}</nav>
    <div class="mt-auto pt-8 px-3">
      <a href="{TUTORIAL_REPO}" target="_blank" rel="noopener"
         class="inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-slate-700 transition">
        <svg viewBox="0 0 16 16" class="h-4 w-4 fill-current"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
        教程代码 · tutorial 分支
      </a>
    </div>
  </div>
</aside>"""


def mobile_bar(active_idx):
    pills = []
    for i, c in enumerate(CHAPTERS):
        label = "总览" if i == 0 else ("运行参考" if c["num"] == "▶" else f"第{c['num']}章")
        cls = ("bg-indigo-600 text-white" if i == active_idx
               else "bg-slate-100 text-slate-600")
        pills.append(f'<a href="./{c["out"]}" class="shrink-0 rounded-full px-3 py-1 '
                     f'text-xs font-medium {cls}">{label}</a>')
    return f"""
<div class="lg:hidden sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
  <div class="flex items-center gap-3 px-4 py-2.5">
    <a href="./index.html" class="text-sm font-black tracking-tight text-slate-900 shrink-0">RL Tutorial</a>
    <div class="flex gap-1.5 overflow-x-auto py-0.5" style="scrollbar-width:none">{''.join(pills)}</div>
  </div>
</div>"""


def toc_rail(toc_tokens):
    entries = [t for t in toc_tokens if t["level"] == 2]
    if not entries:
        return ""
    links = "\n".join(
        f'<a href="#{t["id"]}" class="block border-l-2 border-transparent -ml-px pl-3 '
        f'py-1 leading-snug text-slate-500 hover:text-slate-900 transition">{t["name"]}</a>'
        for t in entries
    )
    return f"""
<aside class="hidden xl:block w-64 shrink-0">
  <div class="sticky top-8 max-h-[calc(100vh-4rem)] overflow-y-auto py-10 pr-6">
    <div class="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">本页目录</div>
    <nav id="toc" class="border-l border-slate-200 text-[13px]">{links}</nav>
  </div>
</aside>"""


def prev_next(active_idx):
    prev_c = CHAPTERS[active_idx - 1] if active_idx > 0 else None
    next_c = CHAPTERS[active_idx + 1] if active_idx < len(CHAPTERS) - 1 else None

    def card(c, kind):
        if c is None:
            return '<div class="hidden sm:block"></div>'
        if kind == "prev":
            tag, align = "← 上一章", ""
        else:
            tag = "开始阅读 →" if active_idx == 0 else "下一章 →"
            align = " sm:text-right"
        if c["num"] == "▶":
            label = c["nav"]
            tag = "运行参考 →"
        else:
            label = "教程总览" if c["num"] == "§" else f'第 {c["num"]} 章 · {c["nav"]}'
        return (f'<a href="./{c["out"]}" class="group block rounded-xl border '
                f'border-slate-200 bg-white p-5 transition hover:border-indigo-300 '
                f'hover:shadow-md{align}">'
                f'<div class="mb-1 text-xs text-slate-400">{tag}</div>'
                f'<div class="font-semibold text-slate-800 group-hover:text-indigo-700">'
                f'{label}</div></a>')

    return (f'<nav class="mt-14 grid gap-4 sm:grid-cols-2">'
            f'{card(prev_c, "prev")}{card(next_c, "next")}</nav>')


def render_page(active_idx, title, body_html, toc_tokens, tailwind_css):
    c = CHAPTERS[active_idx]
    animation_assets = player_assets() + '\n' if 'class="alf-player"' in body_html else ""
    if 'class="lora-figure"' in body_html:
        animation_assets += primer_assets() + '\n'
    clock = ('<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
             '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>')
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · RL Tutorial</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🏠</text></svg>">
<!-- Tailwind CSS（含 typography 插件）本地预编译内嵌，完全离线可用；重编译: build_html.py --css -->
<style>{tailwind_css}</style>
<style>{BASE_CSS}</style>
<style>{PYGMENTS_CSS}</style>
</head>
<body class="bg-slate-50 text-slate-800 antialiased">
{mobile_bar(active_idx)}
<div class="mx-auto max-w-[96rem] lg:flex">
{sidebar(active_idx)}
<main class="min-w-0 flex-1 px-4 py-10 sm:px-8 lg:py-14">
  <div class="mx-auto max-w-3xl">
    <header class="mb-10">
      <div class="mb-4 flex flex-wrap items-center gap-3">
        {mode_pill(c["mode"])}
        <span class="inline-flex items-center gap-1.5 text-sm text-slate-500">{clock}{c["time"]}</span>
      </div>
      <h1 class="text-3xl font-bold leading-snug tracking-tight text-slate-900 sm:text-4xl">{title}</h1>
    </header>
    <article class="prose prose-slate max-w-none">
{body_html}
    </article>
    {prev_next(active_idx)}
    <footer class="mt-12 border-t border-slate-200 pt-6 text-center text-xs text-slate-400">
      RL Tutorial · 基于 <a href="{GITHUB_URL}" class="underline decoration-slate-300 underline-offset-2 hover:text-slate-600">Trinity-RFT</a>
      · 本页由 build_html.py 从 Markdown 生成
    </footer>
  </div>
</main>
{toc_rail(toc_tokens)}
</div>
<script>{SCROLLSPY_JS}</script>
{animation_assets}</body>
</html>
"""


# ---------------------------------------------------------------- CSS 编译

CSS_PATH = ROOT / "tailwind.min.css"


def compile_tailwind(scan_pages):
    """用 Tailwind CLI 扫描渲染好的页面，编译出只含用到的 class 的最小 CSS。"""
    tmp = Path(tempfile.mkdtemp(prefix="rl_tutorial_tw_"))
    scan = tmp / "scan"
    scan.mkdir()
    for name, html in scan_pages:
        (scan / name).write_text(html, encoding="utf-8")
    (tmp / "input.css").write_text(
        "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n")
    (tmp / "tailwind.config.js").write_text(TAILWIND_CONFIG_JS)
    print("  [css] npm install tailwindcss ...")
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund", "--loglevel=error",
         "tailwindcss@3.4.17", "@tailwindcss/typography@0.5.15"],
        cwd=tmp, check=True)
    print("  [css] tailwindcss compile ...")
    subprocess.run(
        ["npx", "--no-install", "tailwindcss", "-c", "tailwind.config.js",
         "-i", "input.css", "-o", str(CSS_PATH),
         "--content", "scan/*.html", "--minify"],
        cwd=tmp, check=True)
    print(f"  [css] -> {CSS_PATH.name} ({CSS_PATH.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- main

def main():
    build_standalone()
    build_fallbacks()
    pages = []
    for i, c in enumerate(CHAPTERS):
        text = (ROOT / c["md"]).read_text(encoding="utf-8")
        title, body = split_title(text)
        body = preprocess(strip_trailing_nav(body))
        body_html, toc_tokens = convert(body)
        body_html = enhance_lora_figures(wrap_tables(rewrite_links(body_html)))
        pages.append((i, title, body_html, toc_tokens))

    if "--css" in sys.argv[1:] or not CSS_PATH.exists():
        scan_pages = [(CHAPTERS[i]["out"], render_page(i, t, b, tk, ""))
                      for i, t, b, tk in pages]
        compile_tailwind(scan_pages)

    css = CSS_PATH.read_text(encoding="utf-8")
    for i, t, b, tk in pages:
        out = ROOT / CHAPTERS[i]["out"]
        out.write_text(render_page(i, t, b, tk, css), encoding="utf-8")
        print(f"  {CHAPTERS[i]['md']:<40} -> {CHAPTERS[i]['out']}")


if __name__ == "__main__":
    main()
