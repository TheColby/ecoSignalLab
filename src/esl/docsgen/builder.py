"""Build Markdown documentation to hyperlink-rich HTML and PDF artifacts.

References:
- Markdown syntax and fenced code handling: https://python-markdown.github.io/
- Mermaid diagram runtime: https://mermaid.js.org/
- Browser-based PDF rendering: https://playwright.dev/python/docs/api/class-page#page-pdf
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown

MERMAID_BLOCK_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    flags=re.DOTALL,
)
LOCAL_MD_LINK_RE = re.compile(r'href="([^":#]+)\.md(#[^"]*)?"')
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FONT_ASSETS_DIR = Path(__file__).parent / "assets" / "fonts"


def _copy_font_assets(html_dir: Path) -> Path:
    """Copy the bundled book face beside generated HTML pages."""
    destination = html_dir / "assets" / "fonts"
    shutil.copytree(_FONT_ASSETS_DIR, destination, dirs_exist_ok=True)
    return destination


@dataclass(slots=True)
class DocsBuildReport:
    """Summary of generated documentation artifacts."""

    root: Path
    output_root: Path
    html_pages: list[Path]
    pdf_pages: list[Path]


@dataclass(slots=True)
class _RenderedPage:
    source: Path
    title: str
    body_html: str
    out_html: Path


def _discover_docs(root: Path) -> list[Path]:
    docs = [
        root / "README.md",
        root / "DESIGN.md",
        root / "ARCHITECTURE.md",
    ]
    docs.extend(sorted((root / "docs").glob("*.md")))
    return [p for p in docs if p.exists()]


def _read_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _rewrite_links(raw_html: str, repo_root: Path) -> str:
    root_str = str(repo_root.resolve()).replace("\\", "/")

    def replace_local(m: re.Match[str]) -> str:
        base = m.group(1)
        frag = m.group(2) or ""
        if base.startswith("http://") or base.startswith("https://"):
            return m.group(0)
        return f'href="{base}.html{frag}"'

    out = LOCAL_MD_LINK_RE.sub(replace_local, raw_html)

    out = out.replace(f'href="{root_str}/', 'href="../')
    out = out.replace(f'href="/{root_str}/', 'href="../')
    out = out.replace(".md\"", '.html"')
    return out


def _upgrade_mermaid_blocks(raw_html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        code = html.unescape(match.group(1)).strip()
        return f'<div class="mermaid">\n{code}\n</div>'

    return MERMAID_BLOCK_RE.sub(repl, raw_html)


def _strip_markdown_inline(text: str) -> str:
    out = text.strip()
    out = re.sub(r"`([^`]+)`", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)
    out = re.sub(r"[*_~]+", "", out)
    out = out.replace('"', "'")
    return out


def _extract_headings(markdown_text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    in_fence = False
    fence_marker = ""
    for raw in markdown_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(raw)
        if not m:
            continue
        level = len(m.group(1))
        title = _strip_markdown_inline(m.group(2))
        if title:
            headings.append((level, title))
    return headings


def _build_auto_visual_block(markdown_text: str, fallback_title: str, max_nodes: int = 28) -> str:
    headings = _extract_headings(markdown_text)
    selected = headings[:max_nodes]
    truncated = len(headings) > len(selected)

    lines: list[str] = []
    lines.append("flowchart TD")
    root_label = _strip_markdown_inline(fallback_title) or "Document"
    lines.append(f'    ROOT["{root_label}"]')

    stack: list[tuple[int, str]] = [(0, "ROOT")]
    for idx, (level, title) in enumerate(selected, start=1):
        node = f"H{idx}"
        safe = title.replace('"', "'")
        lines.append(f'    {node}["{safe}"]')
        while stack and level <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else "ROOT"
        lines.append(f"    {parent} --> {node}")
        stack.append((level, node))

    if truncated:
        lines.append('    ROOT --> MORE["... additional sections omitted for readability"]')

    diagram = "\n".join(lines)
    return (
        "\n\n## Visual Outline (Auto-generated)\n\n"
        "This auto-generated Mermaid graph summarizes this document structure.\n\n"
        "```mermaid\n"
        f"{diagram}\n"
        "```\n"
    )


def _ensure_visual_outline(markdown_text: str, page_title: str) -> str:
    if "```mermaid" in markdown_text:
        return markdown_text
    return markdown_text.rstrip() + _build_auto_visual_block(markdown_text, fallback_title=page_title)


def _equation_preview(latex: str) -> str:
    """Return a plain-English orientation sentence for a displayed equation."""
    normalized = latex.replace(" ", "").lower()
    if "hash" in normalized or "pipeline_hash" in normalized:
        return (
            "At a high level, this is a provenance fingerprint: it combines the processing "
            "choices that must match before two analyses can be called equivalent."
        )
    if "\\arg\\max" in normalized or "topk" in normalized or "operatorname{rank}" in normalized:
        return (
            "At a high level, this is a ranking rule: it turns candidate scores or distances "
            "into an explicit choice of one item or an ordered shortlist."
        )
    if "ndsi" in normalized or "e_{\\mathrm{bio}}" in normalized:
        return (
            "At a high level, this is a bounded contrast: it compares two selected energy "
            "bands so the sign indicates which chosen band dominates."
        )
    if "\\log" in normalized:
        return (
            "At a high level, this is a logarithmic scale conversion: it compresses a wide "
            "range of linear values into decibels or another interpretable log scale."
        )
    if "\\frac" in normalized:
        return (
            "At a high level, this is a normalized relationship: it compares a selected "
            "quantity with an appropriate reference, total, or scale factor."
        )
    if "\\sum" in normalized:
        return (
            "At a high level, this is an accumulation: it combines contributions over samples, "
            "channels, bins, or frames into one quantity."
        )
    if "\\approx" in normalized:
        return (
            "At a high level, this is a model rather than an exact identity: it summarizes an "
            "observed process with stated assumptions and an expected approximation error."
        )
    if "\\in" in normalized or "\\mathcal" in normalized:
        return (
            "At a high level, this defines a set or eligibility condition: it states which "
            "items qualify for a later analysis or decision step."
        )
    return (
        "At a high level, this expression makes the preceding method reproducible by stating "
        "the quantities and relationships used to compute the result."
    )


def _add_equation_previews(markdown_text: str) -> str:
    """Insert a high-level explanation before every displayed TeX block.

    The transformation deliberately ignores fenced code blocks, where dollar
    delimiters may be documentation examples rather than equations to render.
    """
    lines = markdown_text.splitlines()
    output: list[str] = []
    in_fence = False
    fence_marker = ""
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            output.append(lines[index])
            index += 1
            continue

        if not in_fence and stripped == "$$":
            end = index + 1
            while end < len(lines) and lines[end].strip() != "$$":
                end += 1
            if end < len(lines):
                # Only honor a preview immediately attached to this equation.
                # A prior equation's preview must not suppress the next one.
                recent = "\n".join(output[-3:])
                if "equation-preview" not in recent:
                    preview = _equation_preview("\n".join(lines[index + 1 : end]))
                    output.extend(
                        (
                            "",
                            '<p class="equation-preview"><strong>Equation preview.</strong> '
                            f"{preview}</p>",
                            "",
                        )
                    )
                output.extend(lines[index : end + 1])
                index = end + 1
                continue

        output.append(lines[index])
        index += 1

    return "\n".join(output)


def _markdown_extensions() -> tuple[list[str], dict[str, dict[str, bool]]]:
    extensions = [
        "fenced_code",
        "tables",
        "toc",
        "sane_lists",
        "admonition",
        "attr_list",
    ]
    configs: dict[str, dict[str, bool]] = {}
    try:
        import pymdownx.arithmatex  # type: ignore  # noqa: F401
    except Exception:
        return extensions, configs
    extensions.append("pymdownx.arithmatex")
    # Generic mode keeps TeX delimiters for MathJax runtime rendering.
    configs["pymdownx.arithmatex"] = {"generic": True}
    return extensions, configs


def _render_markdown(markdown_text: str, repo_root: Path) -> str:
    extensions, extension_configs = _markdown_extensions()
    markdown_text = _add_equation_previews(markdown_text)
    rendered = markdown.markdown(
        markdown_text,
        extensions=extensions,
        extension_configs=extension_configs,
        output_format="html5",
    )
    rendered = _rewrite_links(rendered, repo_root)
    rendered = _upgrade_mermaid_blocks(rendered)
    return rendered


def _slug(doc_path: Path, repo_root: Path) -> str:
    rel = doc_path.relative_to(repo_root)
    return str(rel.with_suffix(".html")).replace("\\", "/")


def _render_page_template(
    title: str,
    nav_html: str,
    body_html: str,
    page_title: str,
    font_dir_href: str,
) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{page_title} - {title}</title>
  <style>
    /* TeX Gyre Schola is the open New Century Schoolbook-compatible book face. */
    @font-face {{
      font-family: "TeX Gyre Schola";
      font-style: normal;
      font-weight: 400;
      src: url("{font_dir_href}/texgyreschola-regular.otf") format("opentype");
    }}
    @font-face {{
      font-family: "TeX Gyre Schola";
      font-style: normal;
      font-weight: 700;
      src: url("{font_dir_href}/texgyreschola-bold.otf") format("opentype");
    }}
    @font-face {{
      font-family: "TeX Gyre Schola";
      font-style: italic;
      font-weight: 400;
      src: url("{font_dir_href}/texgyreschola-italic.otf") format("opentype");
    }}
    @font-face {{
      font-family: "TeX Gyre Schola";
      font-style: italic;
      font-weight: 700;
      src: url("{font_dir_href}/texgyreschola-bolditalic.otf") format("opentype");
    }}
    :root {{
      --bg: #fdfcf8;
      --panel: #fffefb;
      --ink: #171717;
      --muted: #242424;
      --line: #9c998f;
      --link: #17365d;
      --code: #f0eee7;
      --terminal-bg: #06150d;
      --terminal-border: #287a48;
      --terminal-green: #8dffa9;
      --book-font: "TeX Gyre Schola", "New Century Schoolbook",
        "New Century Schoolbook Std", "Century Schoolbook", "URW Bookman", serif;
      --code-font: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas,
        "Liberation Mono", monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: var(--book-font); font-size: 17px; background: var(--bg); color: var(--ink); }}
    .layout {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: 100vh; }}
    nav {{ border-right: 1px solid var(--line); background: #f6f3eb; padding: 20px 16px; position: sticky; top: 0; height: 100vh; overflow: auto; }}
    nav h1 {{ font-size: 1.05rem; margin: 0 0 14px 0; letter-spacing: 0.01em; }}
    nav a {{ display: block; color: var(--link); text-decoration: none; padding: 6px 0 6px 10px; word-break: break-word; border-left: 2px solid transparent; margin-left: -10px; }}
    nav a:hover {{ text-decoration: underline; background: #eeeadf; }}
    nav a[aria-current="page"] {{ font-weight: 700; background: #eeeadf; border-left-color: var(--link); }}
    .skip-link {{ position: absolute; top: -40px; left: 10px; background: var(--link); color: #fff; padding: 10px 14px; z-index: 100; transition: top 0.2s ease; border-radius: 0 0 5px 5px; text-decoration: none; font-weight: 700; }}
    .skip-link:focus {{ top: 0; }}
    main {{ padding: 28px 32px 48px; }}
    article {{ max-width: 900px; margin: 0 auto; background: var(--panel); padding: 34px 48px 48px; }}
    h1, h2, h3, h4 {{ color: var(--ink); font-family: var(--book-font); font-weight: 700; line-height: 1.18; }}
    h1 {{ font-size: 2.15rem; margin: 1.7em 0 0.55em; }}
    h2 {{ font-size: 1.55rem; margin: 1.55em 0 0.5em; }}
    h3 {{ font-size: 1.22rem; margin: 1.4em 0 0.45em; }}
    h4 {{ font-size: 1.05rem; margin: 1.25em 0 0.4em; }}
    p, li {{ color: var(--muted); line-height: 1.55; }}
    p {{ text-align: justify; hyphens: auto; }}
    p, li, blockquote {{ orphans: 3; widows: 3; }}
    a {{ color: var(--link); }}
    button {{ font: inherit; }}
    pre {{ position: relative; font-family: var(--code-font); background: var(--terminal-bg); color: var(--terminal-green); border: 1px solid var(--terminal-border); box-shadow: inset 0 0 0 1px rgba(141, 255, 169, 0.08); padding: 14px; border-radius: 2px; overflow: auto; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; padding: 4px 8px; font-size: 0.7rem; background: rgba(255, 255, 255, 0.1); color: #fff; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 4px; cursor: pointer; opacity: 0; transition: opacity 0.2s ease; }}
    pre:hover .copy-btn, .copy-btn:focus {{ opacity: 1; }}
    .copy-btn:hover {{ background: rgba(255, 255, 255, 0.2); }}
    code {{ font-family: var(--code-font); background: var(--code); border-radius: 2px; padding: 0.1rem 0.3rem; }}
    pre code {{ background: transparent; color: inherit; font-family: var(--code-font); padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px; border-top: 1.5px solid var(--ink); border-bottom: 1.5px solid var(--ink); }}
    th, td {{ border: 0; border-bottom: 1px solid #c8c3b9; padding: 8px 10px; text-align: left; vertical-align: top; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ background: transparent; font-weight: 700; }}
    .assignment-start, .textbook-page-start {{ height: 0; }}
    .mermaid {{ background: #faf8f1; border: 1px solid #c8c3b9; border-radius: 0; padding: 12px; margin: 18px 0; }}
    .textbook-title-page {{ min-height: 235mm; display: flex; flex-direction: column; justify-content: center; text-align: center; padding: 24mm 16mm; background: var(--panel); border-top: 3px double var(--ink); border-bottom: 3px double var(--ink); }}
    .textbook-title-logo {{ display: block; width: 118px; height: auto; margin: 0 auto 20px; }}
    .textbook-title-page h1 {{ font-size: 3.2rem; letter-spacing: 0.02em; margin: 0; }}
    .textbook-title-page h2 {{ font-size: 1.8rem; max-width: 760px; margin: 1.2rem auto; }}
    .textbook-title-page h3 {{ color: var(--muted); font-size: 1.05rem; font-weight: 500; margin: 0.4rem auto 2rem; }}
    .textbook-title-page p {{ max-width: 620px; margin: 0.5rem auto; }}
    figure.textbook-figure {{ margin: 22px auto; max-width: 100%; text-align: center; break-inside: avoid; page-break-inside: avoid; }}
    figure.textbook-figure img {{ display: block; max-width: 100%; max-height: 168mm; margin: 0 auto 8px; border: 1px solid var(--line); border-radius: 0; }}
    figure.textbook-figure figcaption {{ color: var(--muted); font-size: 0.9rem; line-height: 1.45; }}
    p.equation-preview {{ background: #f4f1e9; border-left: 3px solid var(--line); margin: 1.2rem 0 0.45rem; padding: 0.6rem 0.85rem; text-align: left; }}
    .math-block {{ overflow-x: auto; }}
    mjx-container[jax="SVG"][display="true"] {{ margin: 1rem 0; overflow-x: auto; overflow-y: hidden; }}
    mjx-container[jax="SVG"] svg {{ max-width: 100%; }}
    mjx-container {{ font-size: 100% !important; }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      nav {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
      main {{ padding: 16px; }}
      article {{ padding: 16px; }}
    }}
    @media print {{
      /* A book-sized face improves legibility in the long-form PDF editions. */
      body {{ background: #fffefb; font-size: 11.8pt; }}
      body.textbook-edition {{ font-size: 11.8pt; }}
      body.textbook-edition, body.textbook-edition p, body.textbook-edition li {{ color: #000; }}
      body.textbook-edition p, body.textbook-edition li {{ line-height: 1.48; }}
      body.textbook-edition a {{ color: #000; text-decoration-color: #555; }}
      body.textbook-edition .mermaid {{ background: #fff; border-color: #555; }}
      body.textbook-edition .textbook-title-page {{
        min-height: 9.05in;
        padding: 0.55in 0.35in;
        border-color: #000;
      }}
      body.textbook-edition .textbook-title-page h1 {{ letter-spacing: 0.025em; }}
      nav, .skip-link {{ display: none; }}
      .layout {{ display: block; }}
      main {{ padding: 0; }}
      article {{ max-width: none; border: 0; border-radius: 0; box-shadow: none; padding: 0; }}
      .textbook-title-page {{ break-after: page; page-break-after: always; border-radius: 0; }}
      h1 {{ break-before: page; page-break-before: always; }}
      .textbook-title-page h1 {{ break-before: auto; page-break-before: auto; }}
      h1, h2, h3, h4 {{ break-after: avoid-page; page-break-after: avoid; }}
      h1 + p, h1 + ul, h1 + ol, h1 + table, h1 + figure,
      h2 + p, h2 + ul, h2 + ol, h2 + table, h2 + figure,
      h3 + p, h3 + ul, h3 + ol, h3 + table, h3 + figure,
      h4 + p, h4 + ul, h4 + ol, h4 + table, h4 + figure {{
        break-before: avoid-page;
        page-break-before: avoid;
      }}
      table, pre, .mermaid {{ break-inside: avoid; page-break-inside: avoid; }}
      li {{ break-inside: avoid; page-break-inside: avoid; }}
      .assignment-start, .textbook-page-start {{ break-before: page; page-break-before: always; }}
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["\\\\(", "\\\\)"], ["$", "$"]],
        displayMath: [["\\\\[", "\\\\]"], ["$$", "$$"]],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <script type=\"module\">
    import mermaid from \"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs\";
    mermaid.initialize({{
      startOnLoad: true,
      securityLevel: \"loose\",
      theme: \"neutral\",
      fontFamily: 'TeX Gyre Schola, New Century Schoolbook, New Century Schoolbook Std, Century Schoolbook, URW Bookman, serif'
    }});
    window.__esl_mermaid_ready = (async () => {{
      try {{
        await mermaid.run({{ querySelector: '.mermaid' }});
      }} catch (err) {{
        console.error('mermaid render failed', err);
      }}
    }})();
    document.addEventListener('DOMContentLoaded', () => {{
      const fallbackCopy = (text) => {{
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'absolute';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }};
      document.querySelectorAll('pre').forEach((pre) => {{
        const code = pre.querySelector('code');
        if (!code || code.classList.contains('language-mermaid')) {{
          return;
        }}
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.type = 'button';
        btn.innerText = 'Copy';
        btn.setAttribute('aria-label', 'Copy code to clipboard');
        btn.addEventListener('click', async () => {{
          try {{
            if (navigator.clipboard && window.isSecureContext) {{
              await navigator.clipboard.writeText(code.innerText);
            }} else {{
              fallbackCopy(code.innerText);
            }}
            const originalText = btn.innerText;
            btn.innerText = 'Copied';
            setTimeout(() => {{ btn.innerText = originalText; }}, 1500);
          }} catch (err) {{
            console.error('copy failed', err);
          }}
        }});
        pre.appendChild(btn);
      }});
    }});
  </script>
</head>
<body class="{'textbook-edition' if 'Textbook' in title else ''}">
  <a href="#main-content" class="skip-link">Skip to content</a>
  <div class=\"layout\">
    <nav aria-label=\"Main Documentation\">
      <h1>{title}</h1>
      {nav_html}
    </nav>
    <main id=\"main-content\" tabindex=\"-1\">
      <article>
        {body_html}
      </article>
    </main>
  </div>
</body>
</html>
"""


def _build_nav(rendered_pages: list[_RenderedPage], current_html: Path) -> str:
    items = []
    current_resolved = current_html.resolve()
    for page in rendered_pages:
        label = page.title
        href = os.path.relpath(page.out_html, start=current_html.parent).replace("\\", "/")
        current_attr = ' aria-current="page"' if page.out_html.resolve() == current_resolved else ""
        items.append(f'<a href="{href}"{current_attr}>{html.escape(label)}</a>')
    return "\n".join(items)


def _write_html_pages(root: Path, docs: list[Path], html_dir: Path, title: str) -> list[_RenderedPage]:
    pages: list[_RenderedPage] = []
    html_dir.mkdir(parents=True, exist_ok=True)
    font_dir = _copy_font_assets(html_dir)

    for doc in docs:
        rel = doc.relative_to(root)
        out_html = html_dir / rel.with_suffix(".html")
        out_html.parent.mkdir(parents=True, exist_ok=True)

        markdown_text = doc.read_text(encoding="utf-8")
        page_title = _read_title(markdown_text, rel.stem)
        enriched_markdown = _ensure_visual_outline(markdown_text, page_title=page_title)
        body_html = _render_markdown(enriched_markdown, root)

        pages.append(_RenderedPage(source=doc, title=page_title, body_html=body_html, out_html=out_html))

    for page in pages:
        nav_html = _build_nav(pages, page.out_html)
        font_dir_href = os.path.relpath(font_dir, start=page.out_html.parent).replace("\\", "/")
        page_html = _render_page_template(
            title=title,
            nav_html=nav_html,
            body_html=page.body_html,
            page_title=page.title,
            font_dir_href=font_dir_href,
        )
        page.out_html.write_text(page_html, encoding="utf-8")

    combined_sections = []
    for page in pages:
        source_rel = page.source.relative_to(root)
        combined_sections.append(
            f"<section id='{html.escape(source_rel.as_posix().replace('/', '-'))}'>"
            f"<h1>{html.escape(page.title)}</h1>"
            f"<p><strong>Source:</strong> {html.escape(source_rel.as_posix())}</p>"
            f"{page.body_html}"
            "</section><hr />"
        )
    combined_html = _render_page_template(
        title=title,
        nav_html=_build_nav(pages, html_dir / "ecoSignalLab_docs.html"),
        body_html="\n".join(combined_sections),
        page_title="Combined Documentation",
        font_dir_href=os.path.relpath(font_dir, start=html_dir).replace("\\", "/"),
    )
    (html_dir / "ecoSignalLab_docs.html").write_text(combined_html, encoding="utf-8")

    return pages


async def _render_pdf_pages(html_paths: list[Path], pdf_dir: Path) -> list[Path]:
    try:
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "PDF generation requires Playwright. Install with: pip install -e '.[docs]' "
            "and run: python -m playwright install chromium"
        ) from exc

    pdf_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    async with async_playwright() as pw:  # pragma: no cover - browser runtime
        browser = await pw.chromium.launch()
        context = await browser.new_context(viewport={"width": 1500, "height": 2200})
        for html_path in html_paths:
            page = await context.new_page()
            await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            await page.wait_for_timeout(1300)
            await page.evaluate(
                """
                async () => {
                  // Wait for bundled webfonts before Chromium snapshots the PDF.
                  await document.fonts.ready;
                  if (window.__esl_mermaid_ready) {
                    await window.__esl_mermaid_ready;
                  }
                  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                    await window.MathJax.startup.promise;
                    await window.MathJax.typesetPromise();
                  }
                }
                """
            )
            math_errors = await page.locator("mjx-merror").all_text_contents()
            if math_errors:
                raise RuntimeError(
                    f"MathJax failed to typeset {html_path.name}: " + "; ".join(math_errors)
                )

            # Persist MathJax SVG output so generated HTML remains readable without
            # a network connection or a JavaScript-capable viewer.
            html_path.write_text(await page.content(), encoding="utf-8")
            await page.wait_for_timeout(400)
            out_pdf = pdf_dir / f"{html_path.stem}.pdf"
            if html_path.stem == "TEXTBOOK":
                # A Letter-size master is practical for classroom and office
                # printing. The symmetric 0.9in horizontal margin is binding
                # safe regardless of whether a printer imposes facing pages.
                title_page_pdf = pdf_dir / f"{html_path.stem}.title-page.pdf"
                body_pages_pdf = pdf_dir / f"{html_path.stem}.body-pages.pdf"
                await page.pdf(
                    path=str(title_page_pdf),
                    width="8.5in",
                    height="11in",
                    print_background=True,
                    display_header_footer=False,
                    margin={"top": "0.62in", "bottom": "0.72in", "left": "0.9in", "right": "0.9in"},
                    page_ranges="1",
                )
                await page.pdf(
                    path=str(body_pages_pdf),
                    width="8.5in",
                    height="11in",
                    print_background=True,
                    display_header_footer=True,
                    header_template=(
                        '<div style="width:100%; font-size:8px; color:#111; '
                        'font-family:New Century Schoolbook,Century Schoolbook,TeX Gyre Schola,serif; '
                        'letter-spacing:0.04em; padding:0 0.9in; display:flex; '
                        'justify-content:space-between;">'
                        '<span>ecoSignalLab</span><span>Acoustic Analysis Textbook</span></div>'
                    ),
                    footer_template=(
                        '<div style="width:100%; font-size:8px; color:#111; '
                        'font-family:New Century Schoolbook,Century Schoolbook,TeX Gyre Schola,serif; '
                        'padding:0 0.9in; text-align:center;">'
                        'Colby Leider and ecoSignalLab contributors | '
                        '<span class="pageNumber"></span></div>'
                    ),
                    margin={"top": "0.62in", "bottom": "0.72in", "left": "0.9in", "right": "0.9in"},
                    page_ranges="2-",
                )
                _merge_pdf_parts((title_page_pdf, body_pages_pdf), out_pdf)
            else:
                await page.pdf(
                    path=str(out_pdf),
                    format="A4",
                    print_background=True,
                    display_header_footer=True,
                    footer_template=(
                        '<div style="width:100%; font-size:8px; color:#475569; '
                        'font-family:New Century Schoolbook,Century Schoolbook,TeX Gyre Schola,serif; '
                        'padding:0 10mm; text-align:center;">'
                        'ecoSignalLab documentation | page <span class="pageNumber"></span> '
                        'of <span class="totalPages"></span></div>'
                    ),
                    header_template="<div></div>",
                    margin={"top": "12mm", "bottom": "16mm", "left": "10mm", "right": "10mm"},
                )
            outputs.append(out_pdf)
            await page.close()
        await context.close()
        await browser.close()

    return outputs


def _merge_pdf_parts(parts: tuple[Path, ...], output: Path) -> None:
    """Combine PDF page ranges while retaining clickable annotations.

    The title page intentionally omits running matter. Playwright can apply a
    different header/footer template only per render, so the print master is
    assembled from a clean title range and a paginated body range.
    """
    try:
        import logging
        import warnings

        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:  # pragma: no cover - optional docs dependency
        raise RuntimeError("Textbook print assembly requires pypdf. Install with: pip install -e '.[docs]'.") from exc

    writer = PdfWriter()
    prior_logging_disable = logging.root.manager.disable
    try:
        # pypdf compares annotation dictionaries while combining the ranges.
        # The comparison diagnostics are expected for a document with many
        # links; annotations are retained and validated after the build.
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for part in parts:
                writer.append(PdfReader(part))
        with output.open("wb") as handle:
            writer.write(handle)
    finally:
        logging.disable(prior_logging_disable)
        writer.close()
        for part in parts:
            part.unlink(missing_ok=True)


def build_docs(
    root: str | Path = ".",
    output_root: str | Path = "docs/build",
    formats: set[str] | None = None,
    title: str = "ecoSignalLab Documentation",
    docs_files: list[str | Path] | None = None,
) -> DocsBuildReport:
    """Build project documentation from Markdown into HTML and optional PDF.

    Supported formats:
    - `html`
    - `pdf` (via Playwright Chromium render)
    """
    root_path = Path(root).resolve()
    out_root = Path(output_root).resolve()
    wanted = {x.lower().strip() for x in (formats or {"html", "pdf"}) if x.strip()}
    invalid = wanted - {"html", "pdf"}
    if invalid:
        raise ValueError(f"Unsupported formats: {sorted(invalid)}")

    if docs_files:
        docs = [Path(p).resolve() for p in docs_files]
    else:
        docs = _discover_docs(root_path)
    if not docs:
        raise RuntimeError(f"No markdown docs discovered from root: {root_path}")

    html_dir = out_root / "html"
    pdf_dir = out_root / "pdf"

    rendered_pages = _write_html_pages(root_path, docs, html_dir, title=title)
    html_paths = [p.out_html for p in rendered_pages]
    html_paths.append(html_dir / "ecoSignalLab_docs.html")

    pdf_paths: list[Path] = []
    if "pdf" in wanted:
        pdf_paths = asyncio.run(_render_pdf_pages(html_paths, pdf_dir))

    return DocsBuildReport(
        root=root_path,
        output_root=out_root,
        html_pages=html_paths,
        pdf_pages=pdf_paths,
    )
