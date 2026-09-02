import asyncio
import gzip
import json
import struct
import wave

import aiohttp
import pytest
from aiohttp import web

from astrbot.core.provider.sources.volcengine_stt import (
    ProviderVolcengineSTT,
    _build_audio_request,
    _build_full_request,
    _extract_texts,
    _parse_response,
)

TEST_AUDIO_FRAMES = b"\x01\x02" * 16000  # 1 second of fake stereo 16-bit PCM


def _make_stt_provider(overrides: dict | None = None) -> ProviderVolcengineSTT:
    provider_config = {
        "id": "test-volcengine-stt",
        "type": "volcengine_stt",
        "model": "doubao-seed-asr-2.0",
        "api_key": "test-key",
        "timeout": 5,
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderVolcengineSTT(
        provider_config=provider_config,
        provider_settings={},
    )


def _server_frame(seq: int, payload: dict | None, is_last: bool = False) -> bytes:
    """Build a server response frame the way the ASR service does."""
    compressed = gzip.compress(json.dumps(payload).encode()) if payload else b""
    flags = 0b0011 if is_last else 0b0001
    header = bytes(
        (
            (0b0001 << 4) | 0b0001,
            (0b1001 << 4) | flags,
            (0b0001 << 4) | 0b0001,
            0,
        )
    )
    return (
        header
        + struct.pack(">i", seq)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _error_frame(code: int, message: str) -> bytes:
    compressed = gzip.compress(json.dumps({"message": message}).encode())
    header = bytes(
        (
            (0b0001 << 4) | 0b0001,
            (0b1111 << 4) | 0b0001,
            (0b0001 << 4) | 0b0001,
            0,
        )
    )
    return (
        header
        + struct.pack(">i", -1)
        + struct.pack(">i", code)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def test_provider_registered_with_speech_to_text_type():
    from astrbot.core.provider.register import provider_cls_map

    meta = provider_cls_map["volcengine_stt"]
    assert meta.desc == "火山引擎 STT"
    assert meta.provider_type.value == "speech_to_text"


def test_default_config():
    provider = _make_stt_provider()
    assert provider.api_base == (
        "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_nostream"
    )
    assert provider.resource_id == "volc.seedasr.sauc.duration"
    assert provider.model_name == "doubao-seed-asr-2.0"


def test_headers_new_console_auth():
    provider = _make_stt_provider()
    headers = provider._build_headers()
    assert headers["X-Api-Key"] == "test-key"
    assert headers["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
    assert headers["X-Api-Sequence"] == "-1"
    assert "X-Api-App-Key" not in headers


def test_headers_legacy_console_auth():
    provider = _make_stt_provider({"appid": "123456", "api_key": "token-value"})
    headers = provider._build_headers()
    assert headers["X-Api-App-Key"] == "123456"
    assert headers["X-Api-Access-Key"] == "token-value"
    assert "X-Api-Key" not in headers


def test_full_request_frame_round_trip():
    frame = _build_full_request(1)
    assert frame[0] == 0b0001_0001
    is_last, code, payload = _parse_response(frame)
    assert not is_last
    assert code == 0
    assert payload["audio"] == {
        "format": "pcm",
        "codec": "raw",
        "rate": 16000,
        "bits": 16,
        "channel": 1,
    }
    assert payload["request"]["model_name"] == "bigmodel"


def test_audio_request_frames_mark_last_packet():
    normal = _build_audio_request(7, b"\x00" * 10, is_last=False)
    last = _build_audio_request(7, b"\x00" * 10, is_last=True)
    assert normal[1] == (0b0010 << 4) | 0b0001
    assert last[1] == (0b0010 << 4) | 0b0011
    # The last packet carries a negative sequence number.
    assert struct.unpack(">i", last[4:8])[0] == -7


def test_parse_server_response_and_text_join():
    frame1 = _server_frame(1, {"result": [{"text": "你好"}]})
    frame2 = _server_frame(-2, {"result": [{"text": "世界"}]}, is_last=True)
    is_last, code, payload1 = _parse_response(frame1)
    assert not is_last and code == 0
    is_last, code, payload2 = _parse_response(frame2)
    assert is_last and code == 0
    texts = _extract_texts(payload1) + _extract_texts(payload2)
    assert "".join(texts) == "你好世界"


def test_parse_server_ack_frame():
    # The service acknowledges the full request with a SERVER_ACK (0b1011)
    # frame that still carries the payload size prefix.
    header = bytes(
        (
            (0b0001 << 4) | 0b0001,
            (0b1011 << 4) | 0b0001,
            (0b0001 << 4) | 0b0001,
            0,
        )
    )
    compressed = gzip.compress(json.dumps({"seq": 1}).encode())
    frame = (
        header + struct.pack(">i", 1) + struct.pack(">I", len(compressed)) + compressed
    )
    is_last, code, payload = _parse_response(frame)
    assert not is_last and code == 0 and payload == {"seq": 1}


def test_extract_texts_shapes():
    # Non-streaming: result is a plain object.
    assert _extract_texts({"result": {"text": "你好"}}) == ["你好"]
    # Streaming: result is a list of utterance objects.
    assert _extract_texts({"result": [{"text": "你"}, {"text": "好"}]}) == [
        "你",
        "好",
    ]
    # Tolerate a plain string list.
    assert _extract_texts({"result": ["你好"]}) == ["你好"]
    # No result or non-dict payload yields no text.
    assert _extract_texts({"audio_info": {"duration": 1}}) == []
    assert _extract_texts(None) == []


def test_parse_error_response():
    frame = _error_frame(45000001, "bad key")
    is_last, code, payload = _parse_response(frame)
    assert code == 45000001
    assert payload["message"] == "bad key"


def test_parse_empty_ack_frame():
    header = bytes(
        (
            (0b0001 << 4) | 0b0001,
            (0b1001 << 4) | 0b0001,
            (0b0001 << 4) | 0b0001,
            0,
        )
    )
    frame = header + struct.pack(">i", 1) + struct.pack(">I", 0)
    is_last, code, payload = _parse_response(frame)
    assert not is_last and code == 0 and payload is None


def _write_test_wav(path, frames: bytes, channels: int, rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)


def test_get_text_streams_audio_and_joins_results(tmp_path):
    """End-to-end get_text against a mock ASR WebSocket server."""
    audio_path = tmp_path / "audio.wav"
    _write_test_wav(audio_path, TEST_AUDIO_FRAMES, channels=1, rate=16000)

    received = {"full": None, "audio_seqs": [], "last_seq": None}
    headers_seen = {}

    async def ws_handler(request):
        headers_seen.update(request.headers)
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        msg = await ws.receive()
        received["full"] = _parse_response(msg.data)
        await ws.send_bytes(_server_frame(1, None))

        seq = 1
        while True:
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.BINARY:
                break
            seq = struct.unpack(">i", msg.data[4:8])[0]
            if seq < 0:
                received["last_seq"] = seq
                break
            received["audio_seqs"].append(seq)

        # Non-streaming responses return the final result once, with
        # "result" as a plain object (not a list).
        await ws.send_bytes(
            _server_frame(1, {"result": {"text": "你好，世界"}}, is_last=True)
        )
        await ws.close()
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            provider = _make_stt_provider({"api_base": f"ws://127.0.0.1:{port}/ws"})
            return await provider.get_text(str(audio_path))
        finally:
            await runner.cleanup()

    text = asyncio.run(run())

    assert text == "你好，世界"
    assert headers_seen["X-Api-Key"] == "test-key"
    assert headers_seen["X-Api-Resource-Id"] == "volc.seedasr.sauc.duration"
    full_payload = received["full"][2]
    assert full_payload["audio"]["rate"] == 16000
    assert received["last_seq"] == -(len(received["audio_seqs"]) + 2)


def test_get_text_raises_on_error_response(tmp_path):
    audio_path = tmp_path / "audio.wav"
    _write_test_wav(audio_path, TEST_AUDIO_FRAMES, channels=1, rate=16000)

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_bytes(_error_frame(45000001, "invalid api key"))
        await ws.close()
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            provider = _make_stt_provider({"api_base": f"ws://127.0.0.1:{port}/ws"})
            await provider.get_text(str(audio_path))
        finally:
            await runner.cleanup()

    with pytest.raises(Exception, match="45000001.*invalid api key"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("channels", "rate"),
    [(2, 44100), (1, 8000)],
)
def test_load_wav_pcm_normalizes_to_16k_mono(tmp_path, channels, rate):
    import audioop

    frames = b"\x01\x02" * rate  # one second
    audio_path = tmp_path / "audio.wav"
    _write_test_wav(audio_path, frames, channels=channels, rate=rate)

    pcm = ProviderVolcengineSTT._load_wav_pcm(str(audio_path))

    if channels == 2:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    if rate != 16000:
        frames, _ = audioop.ratecv(frames, 2, 1, rate, 16000, None)
    assert pcm == frames
