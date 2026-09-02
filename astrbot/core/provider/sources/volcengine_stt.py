import asyncio
import audioop
import gzip
import json
import struct
import uuid
import wave

import aiohttp

from astrbot import logger
from astrbot.core.utils.media_utils import MediaResolver

from ..entities import ProviderType
from ..provider import STTProvider
from ..register import register_provider_adapter

# Binary protocol constants of the Volcengine bigmodel streaming ASR service
# (大模型流式语音识别 API, wss://openspeech.bytedance.com/api/v3/...).
_PROTOCOL_VERSION = 0b0001
_HEADER_SIZE = 0b0001
_FULL_CLIENT_REQUEST = 0b0001
_AUDIO_ONLY_REQUEST = 0b0010
_FULL_SERVER_RESPONSE = 0b1001
_SERVER_ACK = 0b1011
_ERROR_RESPONSE = 0b1111
_JSON_SERIALIZATION = 0b0001
_GZIP_COMPRESSION = 0b0001
_POS_SEQUENCE = 0b0001
_NEG_WITH_SEQUENCE = 0b0011

_SAMPLE_RATE = 16000
# 16-bit mono PCM: 32000 bytes per second; 200ms segments are recommended
# by the official API docs.
_SEGMENT_BYTES = _SAMPLE_RATE * 2 // 5


def _build_header(message_type: int, flags: int) -> bytes:
    return bytes(
        (
            (_PROTOCOL_VERSION << 4) | _HEADER_SIZE,
            (message_type << 4) | flags,
            (_JSON_SERIALIZATION << 4) | _GZIP_COMPRESSION,
            0x00,
        )
    )


def _build_full_request(seq: int) -> bytes:
    """Build the initial full client request frame with audio config.

    Args:
        seq: Packet sequence number.

    Returns:
        The full request frame bytes.
    """
    payload = json.dumps(
        {
            "user": {"uid": str(uuid.uuid4())},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": _SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_punc": True,
                "enable_itn": True,
            },
        }
    ).encode("utf-8")
    compressed = gzip.compress(payload)
    return (
        _build_header(_FULL_CLIENT_REQUEST, _POS_SEQUENCE)
        + struct.pack(">i", seq)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _build_audio_request(seq: int, segment: bytes, is_last: bool) -> bytes:
    """Build an audio-only request frame.

    Args:
        seq: Packet sequence number.
        segment: Raw PCM audio bytes of this packet.
        is_last: True to mark the packet as the last one (negative sequence).

    Returns:
        The audio request frame bytes.
    """
    if is_last:
        header = _build_header(_AUDIO_ONLY_REQUEST, _NEG_WITH_SEQUENCE)
        seq = -seq
    else:
        header = _build_header(_AUDIO_ONLY_REQUEST, _POS_SEQUENCE)
    compressed = gzip.compress(segment)
    return (
        header
        + struct.pack(">i", seq)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def _parse_response(data: bytes) -> tuple[bool, int, dict | None]:
    """Parse a server response frame.

    Args:
        data: The raw binary frame received from the server.

    Returns:
        A tuple of (is_last, error_code, payload_json).

    Raises:
        ValueError: If the frame is too short to parse.
    """
    if len(data) < 4:
        raise ValueError("Volcengine STT response frame is too short")
    header_size = data[0] & 0x0F
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    payload = data[header_size * 4 :]

    code = 0
    if flags & 0x01:
        payload = payload[4:]
    is_last = bool(flags & 0x02)
    if message_type == _ERROR_RESPONSE:
        code = struct.unpack(">i", payload[:4])[0]
        payload = payload[4:]
    if message_type in (
        _FULL_SERVER_RESPONSE,
        _SERVER_ACK,
        _ERROR_RESPONSE,
        _FULL_CLIENT_REQUEST,
        _AUDIO_ONLY_REQUEST,
    ):
        payload = payload[4:]

    msg: dict | None = None
    if payload:
        if compression == _GZIP_COMPRESSION:
            payload = gzip.decompress(payload)
        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.debug("Volcengine STT non-JSON payload: %s", payload[:200])
    return is_last, code, msg


def _extract_texts(payload: dict | None) -> list[str]:
    """Extract transcribed text pieces from a response payload.

    The service returns ``result`` as a dict in non-streaming mode
    (bigmodel_nostream) and as a list of utterance dicts in streaming
    modes. Both shapes are handled here.

    Args:
        payload: The parsed response JSON.

    Returns:
        Text pieces in arrival order.
    """
    if not isinstance(payload, dict):
        return []
    result = payload.get("result")
    if isinstance(result, dict):
        return [result.get("text") or ""]
    if isinstance(result, list):
        return [
            item.get("text") or "" if isinstance(item, dict) else item
            for item in result
            if item
        ]
    return []


@register_provider_adapter(
    "volcengine_stt",
    "火山引擎 STT",
    provider_type=ProviderType.SPEECH_TO_TEXT,
)
class ProviderVolcengineSTT(STTProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.api_key = provider_config.get("api_key", "")
        self.appid = provider_config.get("appid", "")
        self.resource_id = provider_config.get(
            "volcengine_resource_id",
            "volc.seedasr.sauc.duration",
        )
        self.api_base = provider_config.get(
            "api_base",
            "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_nostream",
        )
        self.timeout = int(provider_config.get("timeout", 60))
        self.set_model(provider_config.get("model", "doubao-seed-asr-2.0"))

    def _build_headers(self) -> dict:
        """Build the WebSocket handshake headers.

        Returns:
            Headers supporting both the new console (X-Api-Key) and the legacy
            console (X-Api-App-Key + X-Api-Access-Key) authentication.
        """
        headers = {
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        if self.appid:
            headers["X-Api-App-Key"] = self.appid
            headers["X-Api-Access-Key"] = self.api_key
        else:
            headers["X-Api-Key"] = self.api_key
        return headers

    @staticmethod
    def _load_wav_pcm(audio_path: str) -> bytes:
        """Read a WAV file and normalize it to 16 kHz mono 16-bit PCM.

        Args:
            audio_path: Local WAV file path.

        Returns:
            Raw PCM bytes ready to be streamed to the ASR service.
        """
        with wave.open(str(audio_path), "rb") as wav:
            channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

        if not frames:
            raise ValueError("Audio data is empty")
        if sampwidth != 2:
            frames = audioop.lin2lin(frames, sampwidth, 2)
        if channels == 2:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
        if rate != _SAMPLE_RATE:
            frames, _ = audioop.ratecv(frames, 2, 1, rate, _SAMPLE_RATE, None)
        return frames

    async def get_text(self, audio_url: str) -> str:
        async with MediaResolver(
            audio_url,
            media_type="audio",
            default_suffix=".wav",
        ).as_path(target_format="wav") as resolved:
            pcm_data = self._load_wav_pcm(resolved.path)

        texts: list[str] = []
        async with asyncio.timeout(self.timeout):
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(
                    self.api_base,
                    headers=self._build_headers(),
                ) as ws,
            ):
                await ws.send_bytes(_build_full_request(1))

                seq = 2
                for offset in range(0, len(pcm_data), _SEGMENT_BYTES):
                    segment = pcm_data[offset : offset + _SEGMENT_BYTES]
                    is_last = offset + _SEGMENT_BYTES >= len(pcm_data)
                    await ws.send_bytes(_build_audio_request(seq, segment, is_last))
                    seq += 1

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        is_last, code, payload = _parse_response(msg.data)
                        if code != 0:
                            error_msg = (
                                payload.get("message", "unknown error")
                                if isinstance(payload, dict)
                                else "unknown error"
                            )
                            raise Exception(
                                f"火山引擎 STT API 返回错误: {code}, {error_msg}"
                            )
                        texts.extend(_extract_texts(payload))
                        if is_last:
                            break
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break

        return "".join(texts).strip()
