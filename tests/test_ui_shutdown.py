import pytest

from nicegui import app

from voice_app import ui as page


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_fails', [False, True])
async def test_shutdown_cancels_sessions_then_closes_asr(monkeypatch, cancel_fails):
    assert page.shutdown in app._shutdown_handlers
    calls = []

    class Session:
        async def cancel(self):
            calls.append('session')
            if cancel_fails:
                raise RuntimeError('cancel failed')

    async def close_asr():
        calls.append('asr')

    monkeypatch.setattr(page, 'SESSIONS', {Session()})
    monkeypatch.setattr(page.local_asr, 'close', close_asr)
    if cancel_fails:
        with pytest.raises(RuntimeError, match='cancel failed'):
            await page.shutdown()
    else:
        await page.shutdown()
    assert calls == ['session', 'asr']
