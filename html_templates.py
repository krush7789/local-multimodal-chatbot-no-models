"""HTML/CSS templates for Streamlit chat message rendering."""

from __future__ import annotations

from pathlib import Path

BOT_AVATAR_URL = "https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png"
DEFAULT_USER_AVATAR_URL = "https://i.ibb.co/rdZC7LZ/Photo-logo-1.png"
LOCAL_USER_AVATAR_FILE = Path("image.txt")

CSS = """
<style>
.chat-message {
  padding: 1.5rem;
  border-radius: 0.5rem;
  margin-bottom: 1rem;
  display: flex;
}
.chat-message.user {
  background-color: #2b313e;
}
.chat-message.bot {
  background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
"""

# Backward-compatible alias used by existing imports.
css = CSS


def get_bot_template(message: str) -> str:
    """Render bot message card HTML."""
    return f"""
    <div class="chat-message bot">
      <div class="avatar">
        <img src="{BOT_AVATAR_URL}" alt="Bot avatar" />
      </div>
      <div class="message">{message}</div>
    </div>
    """


def get_user_template(message: str) -> str:
    """Render user message card HTML."""
    avatar_source = _resolve_user_avatar()
    return f"""
    <div class="chat-message user">
      <div class="avatar">
        <img src="{avatar_source}" width="350" alt="User avatar" />
      </div>
      <div class="message">{message}</div>
    </div>
    """


def _resolve_user_avatar() -> str:
    if not LOCAL_USER_AVATAR_FILE.exists():
        return DEFAULT_USER_AVATAR_URL
    return LOCAL_USER_AVATAR_FILE.read_text(encoding="utf-8").strip() or DEFAULT_USER_AVATAR_URL
