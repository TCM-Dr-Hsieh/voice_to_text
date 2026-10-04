from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / 'data' / 'settings.json'
MAX_VOCABULARY_CHARS = 25_000
# Relative to the project folder, so the project can be copied or moved. setup.ps1 downloads the models here.
DEFAULT_ASR_DIR = 'models/Qwen3-ASR-1.7B'
DEFAULT_ALIGNER_DIR = 'models/Qwen3-ForcedAligner-0.6B'


def resolve_path(value) -> Path:
    """Expand ~ and resolve relative paths against the project root (never the process CWD)."""
    path = Path(str(value).strip().strip('"')).expanduser()
    return path if path.is_absolute() else ROOT / path


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
class Settings:
    asr_model_dir: str = DEFAULT_ASR_DIR
    asr_device: str = 'auto'
    aligner_model_dir: str = DEFAULT_ALIGNER_DIR
    window_seconds: float = 12.0
    vocabulary: str = ''
    editor: ModelConfig = field(default_factory=lambda: ModelConfig('qwen_qwen3.5-9b'))
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
        current = {key: value for key, value in data.items() if key in settings_names and key != 'editor'}
        current['editor'] = ModelConfig(**editor_defaults)
        settings = cls(**current)
        settings.validate()
        return settings
