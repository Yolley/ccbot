"""Tests for TmuxManager.paste_text — multi-line injection via tmux buffer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccbot.tmux_manager import TmuxManager


def _ok_proc() -> MagicMock:
    proc = MagicMock()
    proc.stderr = []
    return proc


def _err_proc(msg: str = "unknown flag -p") -> MagicMock:
    proc = MagicMock()
    proc.stderr = [msg]
    return proc


class TestPasteText:
    @pytest.mark.asyncio
    async def test_loads_buffer_and_pastes_bracketed_then_enter(self):
        mgr = TmuxManager(session_name="ccbot-test")
        pane = MagicMock()
        pane.cmd.return_value = _ok_proc()
        server = MagicMock()
        server.cmd.return_value = _ok_proc()

        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=pane),
            patch.object(TmuxManager, "server", server),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await mgr.paste_text("@5", "line one\nline two") is True

        # Buffer staged from a temp file, keyed by a sanitized window id
        load_args = server.cmd.call_args[0]
        assert load_args[0] == "load-buffer"
        assert load_args[1] == "-b"
        assert load_args[2] == "ccbot-paste-5"
        # Content written to the staged file is the pasted text
        paste_args = pane.cmd.call_args_list[0][0]
        assert paste_args == ("paste-buffer", "-b", "ccbot-paste-5", "-d", "-p")
        # Enter submits the prompt after the paste
        pane.send_keys.assert_called_once_with("", enter=True, literal=False)

    @pytest.mark.asyncio
    async def test_stages_file_with_exact_content(self):
        mgr = TmuxManager(session_name="ccbot-test")
        pane = MagicMock()
        pane.cmd.return_value = _ok_proc()
        server = MagicMock()
        staged: list[str] = []

        def _capture(*args):
            if args[0] == "load-buffer":
                with open(args[3], encoding="utf-8") as f:
                    staged.append(f.read())
            return _ok_proc()

        server.cmd.side_effect = _capture

        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=pane),
            patch.object(TmuxManager, "server", server),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            await mgr.paste_text("@5", "a\nb\n\nc")

        assert staged == ["a\nb\n\nc"]

    @pytest.mark.asyncio
    async def test_falls_back_to_unbracketed_paste(self):
        """Older tmux without paste-buffer -p still delivers the text."""
        mgr = TmuxManager(session_name="ccbot-test")
        pane = MagicMock()
        pane.cmd.side_effect = [_err_proc(), _ok_proc()]
        server = MagicMock()
        server.cmd.return_value = _ok_proc()

        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=pane),
            patch.object(TmuxManager, "server", server),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await mgr.paste_text("@5", "hello") is True

        assert pane.cmd.call_args_list[1][0] == (
            "paste-buffer",
            "-b",
            "ccbot-paste-5",
            "-d",
        )

    @pytest.mark.asyncio
    async def test_load_buffer_failure_returns_false(self):
        mgr = TmuxManager(session_name="ccbot-test")
        pane = MagicMock()
        server = MagicMock()
        server.cmd.return_value = _err_proc("no such file")

        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=pane),
            patch.object(TmuxManager, "server", server),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await mgr.paste_text("@5", "hello") is False

        pane.cmd.assert_not_called()
        pane.send_keys.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_pane_returns_false(self):
        mgr = TmuxManager(session_name="ccbot-test")
        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=None),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await mgr.paste_text("@5", "hello") is False

    @pytest.mark.asyncio
    async def test_enter_false_skips_submit(self):
        mgr = TmuxManager(session_name="ccbot-test")
        pane = MagicMock()
        pane.cmd.return_value = _ok_proc()
        server = MagicMock()
        server.cmd.return_value = _ok_proc()

        with (
            patch.object(TmuxManager, "_resolve_pane", return_value=pane),
            patch.object(TmuxManager, "server", server),
            patch("ccbot.tmux_manager.asyncio.sleep", new_callable=AsyncMock),
        ):
            assert await mgr.paste_text("@5", "hello", enter=False) is True

        pane.send_keys.assert_not_called()
