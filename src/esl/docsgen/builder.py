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
from dataclasses import dataclass
from pathlib import Path

import markdown

MERMAID_BLOCK_RE = re.compile(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    flags=re.DOTALL,
)
LOCAL_MD_LINK_RE = re.compile(r'href="([^":#]+)\.md(#[^"]*)?"')
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


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


def _render_page_template(title: str, nav_html: str, body_html: str, page_title: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{page_title} - {title}</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #334155;
      --line: #dbe3ef;
      --link: #0b4ea2;
      --code: #e8eef8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background: linear-gradient(135deg, #f7fafc, #eef3fb); color: var(--ink); }}
    .layout {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: 100vh; }}
    nav {{ border-right: 1px solid var(--line); background: #f8fbff; padding: 20px 16px; position: sticky; top: 0; height: 100vh; overflow: auto; }}
    nav h1 {{ font-size: 1rem; margin: 0 0 14px 0; letter-spacing: 0.02em; }}
    nav a {{ display: block; color: var(--link); text-decoration: none; padding: 6px 0; word-break: break-word; }}
    nav a:hover {{ text-decoration: underline; }}
    main {{ padding: 28px 32px 48px; }}
    article {{ max-width: 1060px; margin: 0 auto; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 28px; box-shadow: 0 4px 24px rgba(15, 23, 42, 0.05); }}
    h1, h2, h3, h4 {{ color: #0b2545; }}
    p, li {{ color: var(--muted); line-height: 1.6; }}
    a {{ color: var(--link); }}
    pre {{ background: #0b1b33; color: #eef6ff; padding: 14px; border-radius: 10px; overflow: auto; position: relative; }}
    .copy-btn {{ position: absolute; top: 8px; right: 8px; padding: 4px 8px; background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; cursor: pointer; font-size: 12px; }}
    .copy-btn:hover {{ background: rgba(255,255,255,0.2); }}
    .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border-width: 0; }}
    code {{ background: var(--code); border-radius: 6px; padding: 0.1rem 0.35rem; }}
    pre code {{ background: transparent; padding: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 20px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f6ff; }}
    .mermaid {{ background: #f9fbff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin: 14px 0; }}
    .math-block {{ overflow-x: auto; }}
    mjx-container[jax="CHTML"][display="true"] {{ margin: 1rem 0; overflow-x: auto; overflow-y: hidden; }}
    mjx-container {{ font-size: 100% !important; }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      nav {{ position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }}
      main {{ padding: 16px; }}
      article {{ padding: 16px; }}
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
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <script type=\"module\">
    import mermaid from \"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs\";
    mermaid.initialize({{ startOnLoad: true, securityLevel: \"loose\", theme: \"neutral\" }});
    window.__esl_mermaid_ready = (async () => {{
      try {{
        await mermaid.run({{ querySelector: '.mermaid' }});
      }} catch (err) {{
        console.error('mermaid render failed', err);
      }}
    }})();
    document.addEventListener('DOMContentLoaded', () => {{
      const status = document.createElement('div');
      status.className = 'sr-only';
      status.setAttribute('aria-live', 'polite');
      document.body.appendChild(status);

      document.querySelectorAll('pre').forEach(pre => {{
        const codeElement = pre.querySelector('code');
        if (!codeElement) return;

        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.innerText = 'Copy';
        btn.setAttribute('aria-label', 'Copy code to clipboard');

        btn.addEventListener('click', () => {{
          if (!navigator.clipboard) {{
             console.warn('Clipboard API not available');
             return;
          }}
          const code = codeElement.innerText;
          navigator.clipboard.writeText(code).then(() => {{
            btn.innerText = 'Copied!';
            status.innerText = 'Code copied to clipboard';
            setTimeout(() => {{
              btn.innerText = 'Copy';
              status.innerText = '';
            }}, 2000);
          }}).catch(err => {{
            console.error('Copy failed', err);
          }});
        }});
        pre.appendChild(btn);
      }});
    }});
  </script>
</head>
<body>
  <div class=\"layout\">
    <nav>
      <h1>{title}</h1>
      {nav_html}
    </nav>
    <main>
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
    for page in rendered_pages:
        label = page.title
        href = os.path.relpath(page.out_html, start=current_html.parent).replace("\\", "/")
        items.append(f'<a href="{href}">{html.escape(label)}</a>')
    return "\n".join(items)


def _write_html_pages(root: Path, docs: list[Path], html_dir: Path, title: str) -> list[_RenderedPage]:
    pages: list[_RenderedPage] = []
    html_dir.mkdir(parents=True, exist_ok=True)

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
        page_html = _render_page_template(title=title, nav_html=nav_html, body_html=page.body_html, page_title=page.title)
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
                  if (window.__esl_mermaid_ready) {
                    await window.__esl_mermaid_ready;
                  }
                  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
                    await window.MathJax.startup.promise;
                  }
                }
                """
            )
            await page.wait_for_timeout(400)
            out_pdf = pdf_dir / f"{html_path.stem}.pdf"
            await page.pdf(
                path=str(out_pdf),
                format="A4",
                print_background=True,
                margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
            )
            outputs.append(out_pdf)
            await page.close()
        await context.close()
        await browser.close()

    return outputs


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
