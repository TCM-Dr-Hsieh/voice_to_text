from dataclasses import asdict, dataclass, field, fields, replace
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from .speaker import tuning

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / 'data' / 'settings.json'
MAX_VOCABULARY_CHARS = 25_000
# Relative to the project folder, so the project can be copied or moved. setup.ps1 downloads the models here.
DEFAULT_ASR_DIR = 'models/Qwen3-ASR-1.7B'
DEFAULT_ALIGNER_DIR = 'models/Qwen3-ForcedAligner-0.6B'
DEFAULT_SPEAKER_MODEL = 'models/speaker/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'
MIN_TRANSLATION_LAG = 3      # the three newest segments may still be revised, so a segment is final once 3 newer ones exist


def resolve_path(value) -> Path:
    """Expand ~ and resolve relative paths against the project root (never the process CWD)."""
    path = Path(str(value).strip().strip('"')).expanduser()
    return path if path.is_absolute() else ROOT / path


def listen_host() -> str:
    """The address the page is served on: every interface by default (so other computers on the network can open it); set
    VOICE_APP_HOST=127.0.0.1 to serve this computer only."""
    return os.environ.get('VOICE_APP_HOST', '').strip() or '0.0.0.0'


def shared_mode() -> bool:
    return os.environ.get('VOICE_APP_SHARED') == '1'


def source_allowed(kind: str) -> bool:
    remote_or_file = ('remote_microphone', 'remote_system', 'file')
    return kind in (remote_or_file if shared_mode() else
                    ('microphone', 'loopback', *remote_or_file))


@dataclass
class ModelConfig:
    model: str
    base_url: str = 'http://127.0.0.1:1234/v1'
    api_key: str = 'KEY'
    temperature: float = 0.0
    domain_terms: str = ''

    def validate(self):
        self.base_url = self.base_url.strip().rstrip('/，, ')
        self.model = self.model.strip()
        url = urlsplit(self.base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment:
            raise ValueError('Base URL 必須是完整的 http(s) 網址，不含查詢或片段。')
        if not self.model:
            raise ValueError('模型名稱不可留空。')
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError('Temperature 須為 0–2 的數值。')
        self.temperature = float(self.temperature)
        if len(self.domain_terms) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'領域常見詞彙最多 {MAX_VOCABULARY_CHARS} 字元。')


@dataclass
class MeetingSettings:
    """Meeting mode: who speaks (voice groups), the optional text fill of unknown speakers, and the live translation."""
    speaker_model_path: str = DEFAULT_SPEAKER_MODEL
    unknown_percentile: float = 10.0
    max_speakers: int = 12
    text_fill: bool = True               # an LLM reads the text to name the speaker of sentences the voice could not decide
    text_fill2: bool = False             # a second, blind pass over what the first left unknown (needs text_fill)
    background_filter: bool = True       # quiet talk that fits no known voice (a side conversation a microphone picked up): grey, not translated
    background_rel_db: float = 6.0       # ... when it is this many dB quieter than the usual speech level
    background_max_cos: float = 0.45     # ... and fits no voice group better than this similarity (lower = stricter)
    target_language: str = 'English'
    live_translation: bool = True        # the newest three segments are translated as they come (a draft, one text, no speakers)
    translation_lag: int = 6             # the speaker-labelled translation runs this many segments behind the transcript (at least 3)
    translation_context: int = 50        # how many already translated segments the translator sees
    translation_batch: int = 20          # segments per call when the translation has to catch up
    autosave_seconds: float = 20.0
    advanced: dict = field(default_factory=dict)   # the speaker-marking numbers that differ from their default (see speaker/tuning.py)

    def validate(self):
        self.advanced = tuning.validate_advanced(self.advanced)
        self.speaker_model_path = self.speaker_model_path.strip().strip('"')
        if not self.speaker_model_path:
            raise ValueError('說話者聲紋模型路徑不可留空。')
        if not math.isfinite(self.unknown_percentile) or not 0 <= self.unknown_percentile <= 50:
            raise ValueError('說話者「不明」百分位須介於 0–50。')
        if not isinstance(self.max_speakers, int) or isinstance(self.max_speakers, bool) or not 2 <= self.max_speakers <= 30:
            raise ValueError('說話者人數上限須為 2–30 的整數。')
        if not isinstance(self.background_filter, bool):
            raise ValueError('背景人聲過濾須為開或關。')
        if not isinstance(self.live_translation, bool):
            raise ValueError('即時暫定翻譯須為開或關。')
        if not math.isfinite(self.background_rel_db) or not 1 <= self.background_rel_db <= 30:
            raise ValueError('背景人聲「比常態小聲」須介於 1–30 dB。')
        if not math.isfinite(self.background_max_cos) or not 0.1 <= self.background_max_cos <= 0.9:
            raise ValueError('背景人聲「與說話者的相似度上限」須介於 0.1–0.9。')
        self.target_language = ' '.join(self.target_language.split())
        if not self.target_language or len(self.target_language) > 60:
            raise ValueError('翻譯語言不可留空，且最多 60 字。')
        if (not isinstance(self.translation_lag, int) or isinstance(self.translation_lag, bool)
                or not MIN_TRANSLATION_LAG <= self.translation_lag <= 60):
            raise ValueError(f'延遲語音數須為 {MIN_TRANSLATION_LAG}–60 的整數（最新 3 段仍可能被校稿修改）。')
        if not isinstance(self.translation_context, int) or not 0 <= self.translation_context <= 200:
            raise ValueError('翻譯上下文段數須為 0–200 的整數。')
        if not isinstance(self.translation_batch, int) or not 1 <= self.translation_batch <= 50:
            raise ValueError('補翻每批段數須為 1–50 的整數。')
        if not math.isfinite(self.autosave_seconds) or not 5 <= self.autosave_seconds <= 600:
            raise ValueError('會議自動存檔間隔須為 5–600 秒。')


@dataclass
class Settings:
    asr_model_dir: str = DEFAULT_ASR_DIR
    asr_device: str = 'auto'
    aligner_model_dir: str = DEFAULT_ALIGNER_DIR
    window_seconds: float = 12.0
    vocabulary: str = ''
    editor: ModelConfig = field(default_factory=lambda: ModelConfig('qwen_qwen3.5-9b'))
    translator: ModelConfig = field(default_factory=lambda: ModelConfig('qwen_qwen3.5-9b'))
    meeting: MeetingSettings = field(default_factory=MeetingSettings)
    overlap_seconds: float = 1.0
    context_chars: int = 2000
    streaming: bool = False
    timeout_seconds: float = 120.0

    def validate(self):
        self.asr_model_dir = self.asr_model_dir.strip().strip('"')
        if not self.asr_model_dir:
            raise ValueError('ASR 模型資料夾不可留空。')
        if self.asr_device not in ('auto', 'cuda', 'cpu'):
            raise ValueError('ASR 運算裝置須為 auto、cuda 或 cpu。')
        self.aligner_model_dir = self.aligner_model_dir.strip().strip('"')
        if not self.aligner_model_dir:
            raise ValueError('ForcedAligner 模型資料夾不可留空。')
        if not math.isfinite(self.window_seconds) or not 5 <= self.window_seconds <= 30:
            raise ValueError('window_seconds 須介於 5–30。')
        if len(self.vocabulary) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'專有詞提示最多 {MAX_VOCABULARY_CHARS} 字元。')
        self.editor.validate()
        self.translator.validate()
        self.meeting.validate()
        if not math.isfinite(self.overlap_seconds) or not (0 <= self.overlap_seconds <= self.window_seconds - 3):
            raise ValueError('左側參考 Overlap 須大於等於 0，且至少保留 3 秒新音訊（Z ≤ L−3）。')
        if not isinstance(self.context_chars, int) or not 100 <= self.context_chars <= 20000:
            raise ValueError('上下文長度須為 100–20000 的整數。')
        if not math.isfinite(self.timeout_seconds) or not 5 <= self.timeout_seconds <= 600:
            raise ValueError('API 逾時須為 5–600 秒。')

    def save(self, path=SETTINGS_PATH):
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    @classmethod
    def load(cls, path=SETTINGS_PATH):
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('設定檔最外層必須是 JSON 物件。')
        editor_data = data.get('editor', {})
        if not isinstance(editor_data, dict):
            raise ValueError('校稿模型設定必須是 JSON 物件。')
        editor_defaults = asdict(cls().editor)
        editor_names = {item.name for item in fields(ModelConfig)}
        editor_defaults.update({key: value for key, value in editor_data.items() if key in editor_names})
        settings_names = {item.name for item in fields(cls)}
        current = {key: value for key, value in data.items() if key in settings_names and key not in ('editor', 'translator', 'meeting')}
        current['editor'] = ModelConfig(**editor_defaults)
        translator_data = data.get('translator')
        if translator_data is None:                      # settings saved before meeting mode: translate with the proofreading service
            current['translator'] = replace(current['editor'], domain_terms='')
        elif isinstance(translator_data, dict):
            translator_defaults = asdict(cls().translator)
            translator_defaults.update({key: value for key, value in translator_data.items() if key in editor_names})
            current['translator'] = ModelConfig(**translator_defaults)
        else:
            raise ValueError('翻譯模型設定必須是 JSON 物件。')
        meeting_data = data.get('meeting', {})
        if not isinstance(meeting_data, dict):
            raise ValueError('會議模式設定必須是 JSON 物件。')
        meeting_names = {item.name for item in fields(MeetingSettings)}
        current['meeting'] = MeetingSettings(**{key: value for key, value in meeting_data.items() if key in meeting_names})
        settings = cls(**current)
        settings.validate()
        return settings
