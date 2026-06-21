"""
HTML escaping utilities for safe rendering in Streamlit's unsafe_allow_html blocks.

Centralises all HTML-escape logic so that no unescaped user-supplied or
document-derived string ever reaches a `st.markdown(..., unsafe_allow_html=True)`
call.  This prevents the stored-XSS vector that was identified in the evaluation:
filenames and chunk text from uploaded documents were rendered unescaped inside
ref-card HTML blocks.
"""
import html


def esc(value: str) -> str:
    """HTML-escape a single string value for safe embedding in markup."""
    return html.escape(str(value), quote=True)
