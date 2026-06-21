"""
Security tests: verifying that XSS payloads in document filenames and chunk
text do not survive into the rendered HTML markup.

html.escape() prevents XSS by escaping <, >, ", &, and ' — making it
impossible for injected strings to break out of their text context and
form valid HTML tags or attributes. The tests here verify exactly that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.html_utils import esc


def test_esc_escapes_opening_angle_bracket():
    """< must become &lt; — prevents opening new HTML tags."""
    assert "&lt;" in esc("<script>")
    assert "<script>" not in esc("<script>")


def test_esc_escapes_closing_angle_bracket():
    """The full <script>...</script> tag pair must be fully neutralised."""
    result = esc('<script>alert("xss")</script>')
    assert "<script>" not in result
    assert "</script>" not in result
    assert "&lt;script&gt;" in result


def test_esc_escapes_double_quotes():
    """Double quotes must be escaped to prevent attribute injection."""
    result = esc('"malicious"')
    assert '"malicious"' not in result
    assert "&quot;malicious&quot;" in result


def test_esc_escapes_ampersand():
    assert esc("a & b") == "a &amp; b"


def test_esc_handles_empty_string():
    assert esc("") == ""


def test_esc_handles_plain_text():
    assert esc("normal text") == "normal text"


def test_malicious_img_tag_breaks_out_prevention():
    """An <img onerror=...> payload must not produce a valid HTML tag.

    After escaping, the string must not contain a literal '<img' sequence
    because the '<' has been replaced with '&lt;'.
    """
    evil_filename = '<img src=x onerror=alert(1)>.pdf'
    escaped = esc(evil_filename)
    # The tag opener < is escaped, so no valid HTML tag can be formed
    assert "<img" not in escaped
    assert "&lt;img" in escaped


def test_malicious_script_url_scheme_in_href():
    """A javascript: href must not produce a literal '<a href' tag."""
    evil_text = '<a href="javascript:void(0)">click me</a>'
    escaped = esc(evil_text)
    # The angle brackets are escaped — no raw HTML tag survives
    assert "<a href" not in escaped
    assert "&lt;a" in escaped


def test_full_xss_payload_contains_no_raw_tags():
    """Common XSS probe strings must not produce any literal < or > chars."""
    payloads = [
        '"><script>alert(1)</script>',
        "'; DROP TABLE documents;--",
        '<svg onload=alert(1)>',
        '{{7*7}}',  # template injection probe
    ]
    for payload in payloads:
        result = esc(payload)
        assert "<" not in result, f"Raw '<' survived in: {result!r}"
        assert ">" not in result, f"Raw '>' survived in: {result!r}"
