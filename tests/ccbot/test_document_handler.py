"""Tests for document_handler — relaying .txt attachments as prompts."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.bot import _is_text_document, _safe_file_name


def _make_update(
    data: bytes,
    *,
    file_name: str = "prompt.txt",
    mime_type: str = "text/plain",
    caption: str | None = None,
    user_id: int = 1,
    thread_id: int = 42,
) -> MagicMock:
    """Build a mock Update carrying a document in a forum topic."""
    tg_file = MagicMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(data))

    doc = MagicMock()
    doc.file_name = file_name
    doc.mime_type = mime_type
    doc.file_size = len(data)
    doc.get_file = AsyncMock(return_value=tg_file)

    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.text = None
    update.message.caption = caption
    update.message.document = doc
    update.message.message_thread_id = thread_id
    update.message.chat = MagicMock()
    update.message.chat.type = "supergroup"
    update.message.chat.id = 100
    update.message.chat.send_action = AsyncMock()
    update.effective_chat = update.message.chat
    return update


def _make_context() -> MagicMock:
    context = MagicMock()
    context.bot = AsyncMock()
    context.user_data = {}
    return context


class _Harness:
    """Patches around document_handler with a bound window."""

    def __init__(self, bound: bool = True):
        self.bound = bound

    def __enter__(self):
        self._patches = [
            patch("ccbot.bot.is_user_allowed", return_value=True),
            patch("ccbot.bot._get_thread_id", return_value=42),
            patch("ccbot.bot.session_manager"),
            patch("ccbot.bot.tmux_manager"),
            patch("ccbot.bot.safe_reply", new_callable=AsyncMock),
            patch("ccbot.bot.clear_status_msg_info"),
        ]
        (
            self.allowed,
            self.thread,
            self.sm,
            self.tmux,
            self.reply,
            self.clear_status,
        ) = [p.start() for p in self._patches]
        self.sm.get_window_for_thread.return_value = "@5" if self.bound else None
        self.sm.get_display_name.return_value = "project"
        self.sm.paste_to_window = AsyncMock(return_value=(True, "Pasted to project"))
        self.tmux.find_window_by_id = AsyncMock(return_value=MagicMock())
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    @property
    def pasted(self) -> str:
        self.sm.paste_to_window.assert_called_once()
        return self.sm.paste_to_window.call_args[0][1]

    @property
    def reply_text(self) -> str:
        self.reply.assert_called()
        return self.reply.call_args[0][1]


class TestIsTextDocument:
    def test_mime_text_accepted(self):
        assert _is_text_document("notes", "text/plain")

    def test_txt_extension_accepted_without_mime(self):
        assert _is_text_document("prompt.TXT", "")

    def test_markdown_extension_accepted(self):
        assert _is_text_document("plan.md", "application/octet-stream")

    def test_pdf_rejected(self):
        assert not _is_text_document("doc.pdf", "application/pdf")


class TestSafeFileName:
    def test_strips_directories_and_odd_chars(self):
        assert _safe_file_name("../../etc/pa ss;wd.txt") == "pa_ss_wd.txt"

    def test_empty_falls_back(self):
        assert _safe_file_name("") == "file.txt"


class TestDocumentHandler:
    @pytest.mark.asyncio
    async def test_relays_content_as_prompt(self):
        """Multi-line .txt content is pasted verbatim as the prompt."""
        update = _make_update(b"line one\nline two\n")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_called_once_with("@5", "line one\nline two")
            assert "prompt.txt" in h.reply_text

    @pytest.mark.asyncio
    async def test_normalizes_crlf(self):
        """CRLF/CR are normalized — a bare CR would submit early in the TUI."""
        update = _make_update(b"a\r\nb\rc")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            assert h.pasted == "a\nb\nc"

    @pytest.mark.asyncio
    async def test_caption_prepended(self):
        update = _make_update(b"the body", caption="Refactor this:")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            assert h.pasted == "Refactor this:\n\nthe body"

    @pytest.mark.asyncio
    async def test_non_text_document_rejected(self):
        update = _make_update(
            b"%PDF-1.4", file_name="a.pdf", mime_type="application/pdf"
        )
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_not_called()
            assert "not a text file" in h.reply_text

    @pytest.mark.asyncio
    async def test_non_utf8_rejected(self):
        update = _make_update(b"\xff\xfe\x00bad")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_not_called()
            assert "UTF-8" in h.reply_text

    @pytest.mark.asyncio
    async def test_empty_file_rejected(self):
        update = _make_update(b"   \n\n")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_not_called()
            assert "empty" in h.reply_text

    @pytest.mark.asyncio
    async def test_oversized_file_sent_as_path(self):
        """Beyond the inline limit the file is saved and referenced by path."""
        from ccbot.bot import MAX_INLINE_TEXT_CHARS

        body = b"x" * (MAX_INLINE_TEXT_CHARS + 10)
        update = _make_update(body, caption="read it")
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            sent = h.pasted
            assert sent.startswith("read it\n\n(text file attached: ")
            saved = Path(sent.split("(text file attached: ")[1].rstrip(")"))
            assert saved.read_text(encoding="utf-8") == "x" * (
                MAX_INLINE_TEXT_CHARS + 10
            )
            saved.unlink()

    @pytest.mark.asyncio
    async def test_too_large_to_download_rejected(self):
        from ccbot.bot import MAX_TEXT_FILE_BYTES

        update = _make_update(b"small")
        update.message.document.file_size = MAX_TEXT_FILE_BYTES + 1
        with _Harness() as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            update.message.document.get_file.assert_not_called()
            h.sm.paste_to_window.assert_not_called()
            assert "too large" in h.reply_text

    @pytest.mark.asyncio
    async def test_unbound_topic_asks_for_text_first(self):
        update = _make_update(b"hello")
        with _Harness(bound=False) as h:
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_not_called()
            assert "No session bound" in h.reply_text

    @pytest.mark.asyncio
    async def test_paste_failure_reported(self):
        update = _make_update(b"hello")
        with _Harness() as h:
            h.sm.paste_to_window = AsyncMock(return_value=(False, "Failed to paste"))
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            assert "Failed to paste" in h.reply_text

    @pytest.mark.asyncio
    async def test_unauthorized_user_ignored(self):
        update = _make_update(b"hello")
        with _Harness() as h:
            h.allowed.return_value = False
            from ccbot.bot import document_handler

            await document_handler(update, _make_context())

            h.sm.paste_to_window.assert_not_called()
            assert "not authorized" in h.reply_text
