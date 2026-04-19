#!/usr/bin/env python3
"""
Convert HDC-Brain book markdown files to HTML.
Uses the same visual style as ~/Documents/ml-book.

Usage:
  python3 convert_book.py

Output: docs/book-html/
"""

import re
import os
import shutil
from pathlib import Path

# ─────────────────────────────────────────────
# BOOK STRUCTURE (for sidebar + index)
# ─────────────────────────────────────────────

EXPEDITIONS = [
    {
        "num": "I",
        "title": "Разбираемся что вообще происходит",
        "subtitle": "Цель: понять что такое языковая модель и зачем нужны векторы",
        "color": "#7c6af7",
        "chapters": [
            ("chapter-01", "01", "Что делает языковая модель"),
            ("chapter-02", "02", "Вектор — единственный язык компьютера"),
            ("chapter-03", "03", "Нейронная сеть — функция с параметрами"),
            ("chapter-04", "04", "Как нейронная сеть учится"),
            ("project-01", "🏗️ I", "Проект: Наивная языковая модель"),
        ],
    },
    {
        "num": "II",
        "title": "HDC — когда обычных нейросетей недостаточно",
        "subtitle": "Цель: понять почему появился HDC и как три операции строят всё остальное",
        "color": "#60a5fa",
        "chapters": [
            ("chapter-05", "05", "Зачем придумали HDC"),
            ("chapter-06", "06", "Три операции: Bind, Bundle, Permute"),
            ("chapter-07", "07", "Bipolar — почему +1 и −1"),
            ("chapter-08", "08", "STE — трюк чтобы обучать необучаемое"),
            ("project-02", "🏗️ II", "Проект: HDC-классификатор"),
        ],
    },
    {
        "num": "III",
        "title": "Язык — как текст становится векторами",
        "subtitle": "Цель: понять токенизацию, позиции и кодбук",
        "color": "#4ade80",
        "chapters": [
            ("chapter-09", "09", "Токенизация — BPE с нуля"),
            ("chapter-10", "10", "Позиционное кодирование"),
            ("chapter-11", "11", "Codebook — словарь как отпечатки пальцев"),
            ("project-03", "🏗️ III", "Проект: Мини-кодбук для русских слов"),
        ],
    },
    {
        "num": "IV",
        "title": "Блоки — собираем архитектуру",
        "subtitle": "Цель: понять как каждый блок HDC-Brain работает изнутри",
        "color": "#fbbf24",
        "chapters": [
            ("chapter-12", "12", "HDC Memory — как модель помнит прошлое"),
            ("chapter-13", "13", "Attention — самый важный механизм в NLP"),
            ("chapter-14", "14", "HDC Binding Attention — наш главный трюк"),
            ("chapter-15", "15", "HDCBlock — собираем блок целиком"),
            ("chapter-16", "16", "Thought Loops — многопроходное мышление"),
            ("project-04", "🏗️ IV", "Проект: Мини-HDCBrain (4M параметров)"),
        ],
    },
    {
        "num": "V",
        "title": "Доказательства — почему это работает",
        "subtitle": "Цель: научиться читать логи, понимать метрики, и увидеть живую модель",
        "color": "#f472b6",
        "chapters": [
            ("chapter-17", "17", "Доказательства: почему 3 мысли работают"),
            ("chapter-18", "18", "Loss и метрики — как читать тренировочные логи"),
            ("chapter-19", "19", "AdamW и расписание скорости обучения"),
            ("chapter-20", "20", "Эффективное обучение на GPU"),
        ],
    },
    {
        "num": "VI",
        "title": "Вершина — полная картина",
        "subtitle": "Цель: собрать всё вместе и реализовать с нуля",
        "color": "#f87171",
        "chapters": [
            ("chapter-21", "21", "Трансформер vs HDC-Brain: детальное сравнение"),
            ("chapter-22", "22", "Полный путь токена: от символа до предсказания"),
            ("chapter-23", "23", "Финальный проект: реализуй HDC-Brain с нуля"),
        ],
    },
]

APPENDICES = [
    ("appendix-a", "A", "Глоссарий всех терминов"),
    ("appendix-b", "B", "Математика: линейная алгебра и теория информации"),
    ("appendix-c", "C", "Полный аннотированный код v14.1"),
]

# Flat list of all pages in order: (slug, display_num, title)
ALL_PAGES = []
for exp in EXPEDITIONS:
    ALL_PAGES.extend(exp["chapters"])
for a in APPENDICES:
    ALL_PAGES.append(a)


# ─────────────────────────────────────────────
# MARKDOWN → HTML CONVERTER
# ─────────────────────────────────────────────

def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_format(text):
    """Apply inline markdown formatting."""
    # Escape HTML first (but keep entities we add later)
    # We process inline code first to protect its content
    parts = re.split(r'(`[^`]+`)', text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside backtick — inline code
            inner = part[1:-1]
            result.append(f'<code>{escape_html(inner)}</code>')
        else:
            # Outside backtick — apply other formatting
            p = part
            # Bold (** or __)
            p = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', p)
            p = re.sub(r'__(.+?)__', r'<strong>\1</strong>', p)
            # Italic (* or _) — be careful not to match **)
            p = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', p)
            p = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', p)
            # Links [text](url)
            p = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', p)
            # Escape remaining HTML special chars (but not what we already added)
            # We do a targeted escape: only & that aren't already entities
            p = re.sub(r'&(?!amp;|lt;|gt;|quot;|#)', '&amp;', p)
            result.append(p)
    return ''.join(result)


def detect_callout_type(lines):
    """Detect what kind of callout a blockquote should be."""
    text = ' '.join(lines).lower()
    if any(w in text for w in ['важно', 'warning', 'предупреждение', 'осторожно', 'ошибка', 'баг', 'bug']):
        return 'warning', '⚠️', 'Важно'
    if any(w in text for w in ['совет', 'tip', 'подсказка', 'попробуй']):
        return 'tip', '💡', 'Совет'
    if any(w in text for w in ['аналогия', 'представь', 'история', 'помни', 'как если']):
        return 'analogy', '🎭', 'Аналогия'
    if any(w in text for w in ['формула', 'math', 'математика', 'вычисл']):
        return 'math', '📐', 'Математика'
    return 'info', 'ℹ️', 'Заметка'


def table_to_html(lines):
    """Convert markdown table lines to HTML table."""
    rows = []
    for line in lines:
        # Split by | and strip
        cells = [c.strip() for c in line.split('|')]
        # Remove empty first/last if line starts/ends with |
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        rows.append(cells)

    if not rows:
        return ''

    html = ['<table>']
    # First row is header
    html.append('<thead><tr>')
    for cell in rows[0]:
        html.append(f'<th>{inline_format(cell)}</th>')
    html.append('</tr></thead>')

    # Skip separator row (row with dashes)
    data_rows = [r for r in rows[1:] if not all(re.match(r'^[-: ]+$', c) for c in r)]

    if data_rows:
        html.append('<tbody>')
        for row in data_rows:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td>{inline_format(cell)}</td>')
            html.append('</tr>')
        html.append('</tbody>')

    html.append('</table>')
    return '\n'.join(html)


def list_to_html(lines, ordered=False):
    """Convert list lines to HTML."""
    tag = 'ol' if ordered else 'ul'
    html = [f'<{tag}>']
    for line in lines:
        # Strip list marker
        content = re.sub(r'^(\s*[-*+]|\s*\d+\.)\s+', '', line)
        html.append(f'<li>{inline_format(content)}</li>')
    html.append(f'</{tag}>')
    return '\n'.join(html)


def parse_markdown(md_text):
    """Convert markdown text to HTML body content."""
    html_parts = []

    # Split into fenced code blocks vs regular text
    # Pattern: ```lang\n...\n```
    code_pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    last_end = 0
    segments = []  # (is_code, lang, content)

    for m in code_pattern.finditer(md_text):
        if m.start() > last_end:
            segments.append((False, None, md_text[last_end:m.start()]))
        lang = m.group(1) or 'text'
        code = m.group(2)
        segments.append((True, lang, code))
        last_end = m.end()

    if last_end < len(md_text):
        segments.append((False, None, md_text[last_end:]))

    section_counter = [0]  # mutable for nested function

    for is_code, lang, content in segments:
        if is_code:
            lang_label = lang.upper() if lang and lang != 'text' else 'CODE'
            escaped = escape_html(content.rstrip())
            html_parts.append(f'''<pre>
  <div class="code-header">
    <span class="code-lang">{lang_label}</span>
    <button class="code-copy">Копировать</button>
  </div>
  <code class="language-{lang}">{escaped}</code>
</pre>''')
            continue

        # Process regular markdown content
        # Split into logical blocks (separated by blank lines)
        lines = content.split('\n')
        i = 0

        # State machine
        while i < len(lines):
            line = lines[i]

            # ── Headings ──
            if line.startswith('# '):
                title = inline_format(line[2:].strip())
                # h1 is usually the chapter title — skip it (we generate header separately)
                # but if it appears mid-content, show as h2
                i += 1
                continue

            if line.startswith('## '):
                section_counter[0] += 1
                title = inline_format(line[3:].strip())
                slug = re.sub(r'[^\w\s-]', '', line[3:].strip().lower())
                slug = re.sub(r'[\s_]+', '-', slug)[:50]
                html_parts.append(
                    f'<div class="section-heading" id="{slug}">\n'
                    f'  <h2>{title}</h2>\n'
                    f'</div>'
                )
                i += 1
                continue

            if line.startswith('### '):
                title_raw = line[4:].strip()
                title = inline_format(title_raw)

                # Detect special blocks
                title_lower = title_raw.lower()
                if any(w in title_lower for w in ['попробуй сам', 'задание', 'упражнение', 'домашнее']):
                    # Exercise block — collect until next ## or end
                    exercise_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith('## ') and not lines[i].startswith('### '):
                        exercise_lines.append(lines[i])
                        i += 1
                    inner_html = parse_markdown('\n'.join(exercise_lines))
                    html_parts.append(
                        f'<div class="exercise">\n'
                        f'  <div class="exercise-header">\n'
                        f'    <span class="exercise-badge">Задание</span>\n'
                        f'    <span class="exercise-title">{title}</span>\n'
                        f'  </div>\n'
                        f'  {inner_html}\n'
                        f'</div>'
                    )
                    continue

                if any(w in title_lower for w in ['итоги главы', 'итоги', 'что мы узнали', 'резюме', 'выводы']):
                    # Summary box
                    summary_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].startswith('## ') and not lines[i].startswith('### '):
                        summary_lines.append(lines[i])
                        i += 1
                    inner_html = parse_markdown('\n'.join(summary_lines))
                    html_parts.append(
                        f'<div class="summary-box">\n'
                        f'  <h4>✅ {title}</h4>\n'
                        f'  {inner_html}\n'
                        f'</div>'
                    )
                    continue

                html_parts.append(f'<h3>{title}</h3>')
                i += 1
                continue

            if line.startswith('#### '):
                title = inline_format(line[5:].strip())
                html_parts.append(f'<h4>{title}</h4>')
                i += 1
                continue

            # ── Horizontal rule ──
            if re.match(r'^[-*_]{3,}\s*$', line):
                html_parts.append('<hr class="chapter-divider" />')
                i += 1
                continue

            # ── Blockquote ──
            if line.startswith('>'):
                bq_lines = []
                while i < len(lines) and lines[i].startswith('>'):
                    raw = lines[i]
                    # Strip `> ` or just `>`
                    bq_lines.append(raw[2:] if raw.startswith('> ') else raw[1:])
                    i += 1
                # Detect if it's a key concept box (first line has **bold** definition)
                first = bq_lines[0] if bq_lines else ''
                if re.match(r'^\*\*[^*]+\*\*', first) or first.startswith('**'):
                    # Concept / definition box
                    inner = '\n'.join(bq_lines)
                    inner_html = parse_markdown(inner)
                    html_parts.append(
                        f'<div class="concept-box">\n'
                        f'  <div class="concept-label">Ключевое понятие</div>\n'
                        f'  {inner_html}\n'
                        f'</div>'
                    )
                else:
                    # Regular blockquote or callout
                    ctype, icon, clabel = detect_callout_type(bq_lines)
                    inner = '\n'.join(bq_lines)
                    # Check if it's a short single-line note or multi-line callout
                    if len(bq_lines) == 1:
                        # Simple blockquote
                        html_parts.append(f'<blockquote><p>{inline_format(first)}</p></blockquote>')
                    else:
                        inner_html = parse_markdown(inner)
                        html_parts.append(
                            f'<div class="callout {ctype}">\n'
                            f'  <div class="callout-icon">{icon}</div>\n'
                            f'  <div class="callout-body">\n'
                            f'    <div class="callout-title">{clabel}</div>\n'
                            f'    {inner_html}\n'
                            f'  </div>\n'
                            f'</div>'
                        )
                continue

            # ── Unordered list ──
            if re.match(r'^[\s]*[-*+] ', line):
                list_lines = []
                while i < len(lines) and re.match(r'^[\s]*[-*+] ', lines[i]):
                    list_lines.append(lines[i])
                    i += 1
                html_parts.append(list_to_html(list_lines, ordered=False))
                continue

            # ── Ordered list ──
            if re.match(r'^[\s]*\d+\. ', line):
                list_lines = []
                while i < len(lines) and re.match(r'^[\s]*\d+\. ', lines[i]):
                    list_lines.append(lines[i])
                    i += 1
                html_parts.append(list_to_html(list_lines, ordered=True))
                continue

            # ── Table ──
            if '|' in line and re.match(r'^\s*\|', line):
                table_lines = []
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                html_parts.append(table_to_html(table_lines))
                continue

            # ── Blank line ──
            if not line.strip():
                i += 1
                continue

            # ── Regular paragraph ──
            # Collect consecutive non-blank, non-special lines
            para_lines = []
            while i < len(lines):
                l = lines[i]
                if not l.strip():
                    break
                if l.startswith('#') or l.startswith('>') or re.match(r'^[-*_]{3,}\s*$', l) or l.strip() == '>':
                    break
                if re.match(r'^[\s]*[-*+] ', l) or re.match(r'^[\s]*\d+\. ', l):
                    break
                if '|' in l and re.match(r'^\s*\|', l):
                    break
                para_lines.append(l)
                i += 1

            if para_lines:
                text = ' '.join(para_lines)
                html_parts.append(f'<p>{inline_format(text)}</p>')

    return '\n\n'.join(p for p in html_parts if p.strip())


# ─────────────────────────────────────────────
# SIDEBAR GENERATOR
# ─────────────────────────────────────────────

def build_sidebar(current_slug, is_index=False):
    """Build sidebar HTML."""
    prefix = '' if is_index else '../'

    items = []
    items.append(f'''
    <div class="sidebar-header">
      <div class="sidebar-logo">HDC Brain</div>
      <div class="sidebar-title">Думающие векторы</div>
      <div class="sidebar-subtitle">HDC-Brain v14.1 — языковая модель с нуля</div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">
        <div class="nav-section-title">Начало</div>
        <a href="{prefix}index.html" class="nav-item{' active' if is_index else ''}">
          <span class="chapter-num">—</span>
          Оглавление
        </a>
      </div>''')

    for exp in EXPEDITIONS:
        items.append(f'''
      <div class="nav-section">
        <div class="nav-section-title">Экспедиция {exp["num"]} — {exp["title"]}</div>''')
        for slug, num, title in exp["chapters"]:
            active = ' active' if slug == current_slug else ''
            path = f'{prefix}chapters/{slug}.html'
            items.append(
                f'        <a href="{path}" class="nav-item{active}">\n'
                f'          <span class="chapter-num">{num}</span>\n'
                f'          {title}\n'
                f'        </a>'
            )
        items.append('      </div>')

    items.append('''
      <div class="nav-section">
        <div class="nav-section-title">Приложения</div>''')
    for slug, num, title in APPENDICES:
        active = ' active' if slug == current_slug else ''
        path = f'{prefix}chapters/{slug}.html'
        items.append(
            f'        <a href="{path}" class="nav-item{active}">\n'
            f'          <span class="chapter-num">{num}</span>\n'
            f'          {title}\n'
            f'        </a>'
        )
    items.append('      </div>\n    </nav>')

    return '<aside class="sidebar">' + ''.join(items) + '</aside>'


# ─────────────────────────────────────────────
# PAGE GENERATORS
# ─────────────────────────────────────────────

def get_expedition_for_chapter(slug):
    """Return expedition info for a chapter slug."""
    for exp in EXPEDITIONS:
        for ch_slug, _, _ in exp["chapters"]:
            if ch_slug == slug:
                return exp
    return None


def get_chapter_info(slug):
    """Return (display_num, title) for a chapter slug."""
    for _, num, title in ALL_PAGES:
        # all_pages entries match slug-less comparison
        pass
    for page_slug, num, title in ALL_PAGES:
        if page_slug == slug:
            return num, title
    return '?', 'Unknown'


def get_adjacent_chapters(slug):
    """Return (prev_slug, prev_title, next_slug, next_title)."""
    idx = None
    for i, (page_slug, _, _) in enumerate(ALL_PAGES):
        if page_slug == slug:
            idx = i
            break
    if idx is None:
        return None, None, None, None
    prev_slug, prev_title = None, None
    next_slug, next_title = None, None
    if idx > 0:
        prev_slug, _, prev_title = ALL_PAGES[idx - 1]
    if idx < len(ALL_PAGES) - 1:
        next_slug, _, next_title = ALL_PAGES[idx + 1]
    return prev_slug, prev_title, next_slug, next_title


def extract_chapter_title(md_text):
    """Extract the first meaningful heading from markdown."""
    for line in md_text.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            return line[3:].strip()
        if line.startswith('# '):
            t = line[2:].strip()
            # Skip if it's just the book title
            if 'Глава' in t or 'Appendix' in t or not t.startswith('Думающие'):
                return t
    return ''


def generate_chapter_page(md_file, slug, out_path):
    """Convert a markdown chapter file to HTML and write it."""
    md_text = Path(md_file).read_text(encoding='utf-8')

    num, title = get_chapter_info(slug)
    exp = get_expedition_for_chapter(slug)
    exp_label = f'Экспедиция {exp["num"]}' if exp else 'Приложение'
    is_project = slug.startswith('project-')
    page_type = 'Проект' if is_project else 'Глава'
    page_num_display = num.replace('🏗️ ', '') if is_project else num

    total_chapters = len(ALL_PAGES)
    current_idx = next((i for i, (s, _, _) in enumerate(ALL_PAGES) if s == slug), 0)
    progress_pct = round((current_idx / total_chapters) * 100)

    sidebar = build_sidebar(slug)

    prev_slug, prev_title, next_slug, next_title = get_adjacent_chapters(slug)

    # Build chapter nav
    nav_html = '<div class="chapter-nav">'
    if prev_slug:
        nav_html += (
            f'<a href="{prev_slug}.html" class="chapter-nav-item prev">\n'
            f'  <div class="nav-direction">← Предыдущая</div>\n'
            f'  <div class="nav-chapter-title">{prev_title}</div>\n'
            f'</a>'
        )
    else:
        nav_html += '<div></div>'

    if next_slug:
        nav_html += (
            f'<a href="{next_slug}.html" class="chapter-nav-item next">\n'
            f'  <div class="nav-direction">Следующая →</div>\n'
            f'  <div class="nav-chapter-title">{next_title}</div>\n'
            f'</a>'
        )
    else:
        nav_html += '<div></div>'
    nav_html += '</div>'

    # Convert markdown to HTML (skip the first h1 heading — we render it separately)
    # Remove the first # heading line
    lines = md_text.split('\n')
    # Find and skip first h1
    filtered = []
    skipped_h1 = False
    for line in lines:
        if not skipped_h1 and line.startswith('# '):
            skipped_h1 = True
            continue
        filtered.append(line)
    clean_md = '\n'.join(filtered)

    body_html = parse_markdown(clean_md)

    # Determine subtitle (first paragraph after h2 chapter heading)
    subtitle = ''
    in_subtitle = False
    for line in clean_md.split('\n'):
        if line.startswith('## '):
            in_subtitle = True
            continue
        if in_subtitle and line.strip() and not line.startswith('#') and not line.startswith('>'):
            subtitle = line.strip()[:200]
            break

    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{page_type} {page_num_display} — {title} | HDC Brain</title>
  <link rel="stylesheet" href="../assets/css/book.css" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" />
</head>
<body>

<div class="layout">

  {sidebar}

  <main class="main">
    <div class="topbar">
      <div class="topbar-breadcrumb">
        <a href="../index.html">HDC Brain</a> /
        <span>{exp_label}</span> /
        <span>{page_type} {page_num_display}</span>
      </div>
      <div class="topbar-progress">
        <span>{page_num_display} из {total_chapters}</span>
        <div class="progress-bar">
          <div class="progress-bar-fill" style="width: {progress_pct}%"></div>
        </div>
      </div>
    </div>

    <div class="content">

      <div class="chapter-meta">
        <span class="chapter-tag">{exp_label}</span>
        <span class="chapter-number">{page_type} {page_num_display}</span>
      </div>

      <h1 class="chapter-title">{title}</h1>
      <p class="chapter-subtitle">{inline_format(subtitle)}</p>

      <hr class="chapter-divider" />

      {body_html}

      {nav_html}

    </div>
  </main>

</div>

<script src="../assets/js/book.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
  document.querySelectorAll('pre code').forEach(block => {{
    hljs.highlightElement(block);
  }});
</script>
</body>
</html>'''

    Path(out_path).write_text(html, encoding='utf-8')
    print(f'  ✓ {slug}.html')


def generate_index(out_path):
    """Generate index.html."""
    total = len(ALL_PAGES)
    sidebar = build_sidebar('', is_index=True)

    exp_cards = []
    for exp in EXPEDITIONS:
        color = exp["color"]
        pills = ''
        for slug, num, title in exp["chapters"]:
            pills += f'<a href="chapters/{slug}.html" class="chapter-pill available">Гл. {num} — {title}</a>\n'

        exp_cards.append(f'''
        <div class="expedition-card" style="border-top: 3px solid {color};">
          <div class="expedition-card-header">
            <div class="expedition-num">{exp["num"]}</div>
            <div class="expedition-card-title">{exp["title"]}</div>
          </div>
          <div class="expedition-card-desc">{exp["subtitle"]}</div>
          <div class="expedition-chapters">
            {pills}
          </div>
        </div>''')

    # Appendices card
    app_pills = ''
    for slug, num, title in APPENDICES:
        app_pills += f'<a href="chapters/{slug}.html" class="chapter-pill available">Прил. {num} — {title}</a>\n'

    exp_cards.append(f'''
        <div class="expedition-card" style="border-top: 3px solid #94a3b8;">
          <div class="expedition-card-header">
            <div class="expedition-num" style="font-size:0.65rem;">ПРИ</div>
            <div class="expedition-card-title">Приложения</div>
          </div>
          <div class="expedition-card-desc">Справочные материалы: глоссарий, математика, полный код</div>
          <div class="expedition-chapters">
            {app_pills}
          </div>
        </div>''')

    cards_html = '\n'.join(exp_cards)

    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Думающие векторы — HDC-Brain v14.1</title>
  <link rel="stylesheet" href="assets/css/book.css" />
  <style>
    .content {{ max-width: 960px; }}
  </style>
</head>
<body>

<div class="layout">

  {sidebar}

  <main class="main">
    <div class="topbar">
      <div class="topbar-breadcrumb">HDC Brain</div>
      <div class="topbar-progress">
        <span>{total} глав + 3 приложения</span>
        <div class="progress-bar">
          <div class="progress-bar-fill" style="width: 0%"></div>
        </div>
      </div>
    </div>

    <div class="content">

      <div class="hero">
        <div class="hero-eyebrow">Персональное руководство по архитектуре</div>
        <h1 class="hero-title">Думающие<br>векторы</h1>
        <p class="hero-desc">
          Строим языковую модель HDC-Brain v14.1 с нуля — на принципах
          Hyperdimensional Computing, а не трансформерах. Без воды,
          с реальным кодом, реальными багами и реальными результатами.
        </p>
        <div class="hero-stats">
          <div>
            <div class="hero-stat-value">23</div>
            <div class="hero-stat-label">Главы</div>
          </div>
          <div>
            <div class="hero-stat-value">299M</div>
            <div class="hero-stat-label">Параметров</div>
          </div>
          <div>
            <div class="hero-stat-value">6</div>
            <div class="hero-stat-label">Экспедиций</div>
          </div>
          <div>
            <div class="hero-stat-value">BPB 4.2</div>
            <div class="hero-stat-label">Результат</div>
          </div>
        </div>
      </div>

      <h2 style="margin-top:0; border:none; padding:0; font-size:1.3rem; margin-bottom:24px;">Маршрут</h2>

      <div class="expedition-grid">
        {cards_html}
      </div>

      <div class="concept-box" style="margin-top: 48px;">
        <div class="concept-label">Как устроена каждая глава</div>
        <h3>Принцип: история → термин → код → формула → "вот что это буквально значит"</h3>
        <p>Ни одна концепция не появится без предварительной истории. Ни одна формула не появится без кода. Ты читаешь на уровне понимания, а математика догоняет.</p>
      </div>

    </div>
  </main>

</div>

<script src="assets/js/book.js"></script>
</body>
</html>'''

    Path(out_path).write_text(html, encoding='utf-8')
    print('  ✓ index.html')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    book_src = Path(__file__).parent / 'book'
    ml_book_assets = Path.home() / 'Documents' / 'ml-book' / 'assets'
    out_dir = Path(__file__).parent / 'book-html'

    print('📚 HDC-Brain Book → HTML Converter')
    print('=' * 40)

    # Create output directories
    (out_dir / 'chapters').mkdir(parents=True, exist_ok=True)
    (out_dir / 'assets' / 'css').mkdir(parents=True, exist_ok=True)
    (out_dir / 'assets' / 'js').mkdir(parents=True, exist_ok=True)

    # Copy CSS and JS from ml-book
    print('\n📋 Копирую ассеты из ml-book...')
    shutil.copy(ml_book_assets / 'css' / 'book.css', out_dir / 'assets' / 'css' / 'book.css')
    shutil.copy(ml_book_assets / 'js' / 'book.js', out_dir / 'assets' / 'js' / 'book.js')
    print('  ✓ book.css')
    print('  ✓ book.js')

    # Generate index
    print('\n📄 Генерирую index.html...')
    generate_index(out_dir / 'index.html')

    # Generate chapter pages
    print('\n📖 Конвертирую главы...')
    all_slugs = [(s, n, t) for s, n, t in ALL_PAGES]

    for slug, num, title in all_slugs:
        md_file = book_src / f'{slug}.md'
        out_file = out_dir / 'chapters' / f'{slug}.html'
        if md_file.exists():
            generate_chapter_page(md_file, slug, out_file)
        else:
            print(f'  ⚠ MISSING: {slug}.md')

    print(f'\n✅ Готово! Открой: {out_dir}/index.html')
    print(f'   Всего страниц: {len(all_slugs) + 1}')


if __name__ == '__main__':
    main()
