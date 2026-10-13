"""The tunable numbers of speaker marking: one registry that feeds the settings pop-up, the validation and the tracker.

Every knob has a default (the value the pipeline was built and measured with), a range, a plain-words description of what it does and of
what raising or lowering it does. Only the values that differ from the default are stored in the settings file (`meeting.advanced`), so
a knob that was never touched follows the default of the version in use. Four knobs live in settings fields of their own, because they
were there first and have their own boxes in the meeting tab (`top_level`); the pop-up shows them with all the others.

The keys are the names of the fields of `Params` (and of `UnitParams` for the first group): `build_params` in service.py hands them on.
This module imports nothing from the project (the settings module imports it); tests/test_speaker_tuning.py checks that every default
here is the default the consuming code really has, that every range holds its default, and that every knob reaches its consumer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

STAGES = ('單位與句子', '建立與更新說話者', '句子標記與「不明」', '背景人聲', '文字補標')


@dataclass(frozen=True)
class Knob:
    key: str
    label: str
    stage: str
    default: float
    low: float
    high: float
    step: float
    what: str                    # what it does
    higher: str                  # what raising it does
    lower: str                   # what lowering it does
    integer: bool = False
    star: bool = False           # a knob that matters most
    top_level: bool = False      # stored in its own field of the meeting settings, not in `advanced`


KNOBS: tuple[Knob, ...] = (
    # ---- 1. units and sentences (units.py)
    Knob('target', '單位目標長度（秒）', STAGES[0], 0.8, 0.3, 3.0, 0.1,
         '一個單位（取一次聲紋的一段聲音）累積到這麼長就結束。',
         '聲紋較穩，但一個單位更容易跨到兩個人。', '切得更細，但每個聲紋更雜、更不穩。'),
    Knob('pause', '單位斷點的停頓（秒）', STAGES[0], 0.3, 0.1, 1.5, 0.05,
         '停頓超過這麼久就結束一個單位。',
         '更少在停頓處切開，一個單位可能混進兩個人。', '切得更碎，碎片多半被併走或丟棄。'),
    Knob('min_unit', '最短單位（秒）', STAGES[0], 0.4, 0.1, 1.5, 0.05,
         '比這更短的碎片併進鄰居，併不了就丟棄（沒有聲紋）。',
         '更多字沒有聲紋，「不明」變多。', '很短的聲音也算聲紋，可靠度下降。'),
    Knob('max_merged', '合併後單位上限（秒）', STAGES[0], 1.3, 0.5, 4.0, 0.1,
         '碎片併進鄰居之後，單位長度的上限。',
         '允許更長的合併單位。', '更多碎片因為併不進去而被丟棄。'),
    Knob('sent_pause', '句子斷點的停頓（秒）', STAGES[0], 0.5, 0.2, 3.0, 0.1,
         '停頓超過這麼久就結束一句（一句一個標籤）。',
         '句子變長，一句可能夾著兩個人。', '句子變短，每句的根據少，「不明」變多。'),
    # ---- 2. the speakers (tracker.py)
    Knob('max_speakers', '說話者人數上限', STAGES[1], 12, 2, 30, 1,
         '一場會議最多分出幾位說話者。',
         '允許更多人，也更容易長出誤增的說話者。', '超過的人併進既有的說話者。', integer=True, top_level=True),
    Knob('first_units', '新說話者所需單位數', STAGES[1], 6, 3, 30, 1,
         '「誰都不像」的聲音單位，至少有這麼多個彼此相像，才會成為新的一位說話者。',
         '更晚才認出新的人，說話很少的人可能一直是「不明」。', '更快認出新的人，但一陣雜音就可能被當成一個人。', integer=True),
    Knob('first_seconds', '新說話者所需說話時間（秒）', STAGES[1], 4.0, 1.0, 20.0, 0.5,
         '上面這些單位加起來至少要這麼多秒。',
         '同上，更嚴。', '同上，更鬆。'),
    Knob('first_sentences', '新說話者所需句數', STAGES[1], 2, 1, 10, 1,
         '上面這些單位要來自至少這麼多句不同的話（一陣短暫的雜音不算一個人）。',
         '同上，更嚴。', '同上，更鬆。', integer=True),
    Knob('link_cos', '算同一位新說話者的相似度', STAGES[1], 0.50, 0.2, 0.9, 0.01,
         '「誰都不像」的單位，與其中最典型的一個相似度達到這個值，才算同一個人。',
         '要求更像才算同一個人：不易把不同的人湊成一群，但新說話者更難成立。',
         '比較寬鬆：容易把不同的人湊成一位，或把雜音當成人。', star=True),
    Knob('tight_cos', '新說話者的緊密度', STAGES[1], 0.62, 0.3, 0.95, 0.01,
         '湊成新說話者的單位，與它們自己的中心的平均相似度要達到這個值，否則只是雜音。',
         '要求更緊密，更少誤增。', '更容易成立，也更容易誤增。'),
    Knob('new_cos', '判定「沒人認識」的相似度', STAGES[1], 0.45, 0.1, 0.8, 0.01,
         '一句話與每一位說話者的相似度都低於這個值，這句的單位才會被當成「誰都不像」，作為新說話者的候選。',
         '更多句被當成新的人，也更容易誤增說話者。', '更多句被歸給最像的人，聲音像的新說話者會被併進去。', star=True),
    Knob('assign_cos', '候選單位併入既有說話者的相似度', STAGES[1], 0.55, 0.2, 0.9, 0.01,
         '候選單位與某位說話者的中心相似度達到這個值，就併給他，不再當成新說話者的候選。',
         '更少併入，更多留作新說話者的候選。', '更多併入既有的人。'),
    Knob('update_cos', '更新說話者中心的相似度', STAGES[1], 0.55, 0.2, 0.9, 0.01,
         '單位與所屬說話者的中心相似度達到這個值，才用來更新那位的中心。',
         '中心更乾淨，但對聲音的變化適應較慢。', '中心可能被混進別人的聲音。'),
    Knob('merge_cos', '自動合併的相似度', STAGES[1], 0.88, 0.6, 0.99, 0.01,
         '兩位說話者的中心相似度達到這個值，就視為同一個人而自動合併（每分鐘檢查一次）。',
         '更不容易誤併，但同一個人被分成兩位時不會自動併回。', '更積極合併，聲音相近的兩個人可能被併成一位。'),
    Knob('refit_seconds', '整理週期（秒）', STAGES[1], 60.0, 20.0, 600.0, 10.0,
         '多久整理一次說話者：清理中心、自動合併、重看「不明」、更新常態音量。第一次建立說話者與背景判斷也要等到這個時間點。',
         '反應更慢，新說話者更晚出現。', '反應更快，但每次用的資料少、比較不穩。'),
    Knob('max_sentence_seconds', '一句最長（秒）', STAGES[1], 15.0, 5.0, 60.0, 1.0,
         '一個人連續說很久、沒有停頓也沒有句號時，每隔這麼久強制切成一句，標記才不會等太久。',
         '一句可能夾著兩個人，標記等得更久。', '切得更碎，每句的根據更少。'),
    Knob('orphan_keep_seconds', '候選單位保留時間（秒）', STAGES[1], 600.0, 60.0, 3600.0, 60.0,
         '「誰都不像」的聲音單位保留多久，等它們湊成新的說話者。',
         '很久才說一次話的人也有機會成為說話者。', '偶爾才發言的人不易成立。'),
    # ---- 3. labelling a sentence and "unknown" (tracker.py)
    Knob('unknown_percentile', '「不明」百分位', STAGES[2], 10.0, 0.0, 50.0, 1.0,
         '一句話與最像、第二像的說話者的差距，低於「過去各句差距的第 p 百分位」就標「不明」。',
         '「不明」更多、標錯更少。', '標出來的更多、標錯也更多；0 幾乎沒有「不明」。', star=True, top_level=True),
    Knob('abs_conf', '最低差距', STAGES[2], 0.03, 0.0, 0.3, 0.005,
         '與最像、第二像的說話者的差距低於這個值的句子，一律不標。',
         '更多句子標「不明」。', '更少句子標「不明」。'),
    Knob('min_history', '判斷「弱」所需的過去句數', STAGES[2], 8, 3, 60, 1,
         '至少累積這麼多句，才用過去的句子算上面的百分位。',
         '更晚才開始用百分位。', '用太少的句子判斷，不穩。', integer=True),
    Knob('rescore_seconds', '回頭重看的範圍（秒）', STAGES[2], 900.0, 60.0, 3600.0, 60.0,
         '說話者有變動（新增、合併）時，這麼近以內的「不明」句與背景判斷會重新看一次。',
         '更久以前的句子也會被補標，但每次整理的工作量較大。', '較舊的「不明」不再回頭補標。'),
    # ---- 4. background talk (tracker.py)
    Knob('background_rel_db', '背景：比常態小多少（dB）', STAGES[3], 6.0, 1.0, 30.0, 0.5,
         '一句話的音量要比最近的常態音量小這麼多（dB），才可能是背景。',
         '只抓非常小聲的，漏抓增加。', '更多句子被標成背景，誤標風險上升。', star=True, top_level=True),
    Knob('background_max_cos', '背景：與說話者的相似度上限', STAGES[3], 0.45, 0.1, 0.9, 0.025,
         '一句話與所有說話者的相似度（各單位最佳相似度的平均）要低於這個值，才可能是背景。',
         '抓到更多背景，但小聲的真人發言也開始被誤標。', '更保守，漏抓的背景會以「不明」進逐字稿。', star=True, top_level=True),
    Knob('background_min_units', '背景：開始判斷所需單位數', STAGES[3], 40, 10, 300, 1,
         '至少聽過這麼多個量得到音量的單位，才開始判斷背景（要先有「常態音量」可比）。',
         '更晚才開始判斷。', '用太少的資料估常態音量，不穩。', integer=True),
    Knob('background_window_units', '背景：常態音量取最近幾個單位', STAGES[3], 300, 50, 2000, 10,
         '常態音量是最近這麼多個單位的音量中位數。',
         '更穩定，但對音量的長期變化反應較慢。', '更靈敏，但容易被一小段特別大聲或小聲的話影響。', integer=True),
    Knob('quiet_first_seconds', '背景聲音成為說話者所需時間（秒）', STAGES[3], 12.0, 2.0, 60.0, 1.0,
         '主要由背景句子湊成的新說話者，至少要有這麼多秒的說話才成立（持續發言的人會被當成說話者，短暫的私語不會）。',
         '短暫的小聲說話更不容易成為說話者，但真的很小聲的人要等更久才被承認。', '背景聲音更容易長成幽靈說話者。'),
    Knob('quiet_first_sentences', '背景聲音成為說話者所需句數', STAGES[3], 4, 1, 20, 1,
         '同上，至少要來自這麼多句話。',
         '同上，更嚴。', '同上，更鬆。', integer=True),
    # ---- 5. text fill (tracker.py)
    Knob('fill_after', '文字補標：等待後文句數', STAGES[4], 10, 0, 60, 1,
         '「不明」句要等後面有這麼多句，才拿去問 LLM（後面的句子是它的上下文）。',
         '上下文更完整，但補標更晚。', '上下文少，補標較不準。', integer=True),
    Knob('fill_batch', '文字補標：累積幾句才送', STAGES[4], 6, 1, 60, 1,
         '累積這麼多待問的句子才送一次 LLM。',
         'LLM 呼叫更少，延遲更久。', '呼叫更頻繁。', integer=True),
    Knob('fill_max_age', '文字補標：最久等多久（秒）', STAGES[4], 120.0, 10.0, 900.0, 10.0,
         '最舊的待問句等了這麼久就送，不管累積了幾句。',
         '延遲更久，呼叫較少。', '更快補標，呼叫較多。'),
    Knob('fill2_batch', '二階補標：累積幾句才送', STAGES[4], 12, 1, 60, 1,
         '二階補標累積這麼多句才送一次（會議結束時的最後一輪會送出剩下的）。',
         '呼叫較少、更晚。', '更早補，呼叫較多。', integer=True),
    Knob('fill2_refits', '二階補標：與一階答案相隔幾次整理', STAGES[4], 1, 0, 10, 1,
         '一階答案回來後，至少隔這麼多次整理才問二階（等說話者與鄰句的標記更新）。',
         '更晚，鄰句的標記更新更多。', '更早，但鄰句的標記可能還沒變。', integer=True),
)

BY_KEY = {knob.key: knob for knob in KNOBS}
assert len(BY_KEY) == len(KNOBS)
DEFAULTS = {knob.key: knob.default for knob in KNOBS}
STORED = tuple(knob for knob in KNOBS if not knob.top_level)           # the knobs kept in `advanced`
TOP_LEVEL = tuple(knob.key for knob in KNOBS if knob.top_level)
UNIT_KEYS = ('target', 'pause', 'min_unit', 'max_merged', 'sent_pause')


def number(value, knob: Knob):
    """A knob's value checked against its range and kind (int or float); ValueError says which and what is allowed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'進階參數「{knob.label}」須為數字。')
    if not knob.low <= value <= knob.high:
        raise ValueError(f'進階參數「{knob.label}」須介於 {knob.low:g}–{knob.high:g}。')
    if knob.integer:
        if float(value) != int(value):
            raise ValueError(f'進階參數「{knob.label}」須為整數。')
        return int(value)
    return float(value)


def validate_advanced(raw) -> dict:
    """The stored overrides, checked and normalised: keys that are not knobs are ignored (another version wrote them), a value equal to
    the default is dropped, a bad value raises ValueError."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError('說話者標記的進階參數須為物件。')
    out = {}
    for knob in STORED:
        if knob.key not in raw:
            continue
        value = number(raw[knob.key], knob)
        if value != knob.default:
            out[knob.key] = value
    values = {**DEFAULTS, **out}
    if not values['min_unit'] <= values['target'] <= values['max_merged']:
        raise ValueError('單位長度須符合「最短單位 ≤ 單位目標長度 ≤ 合併後單位上限」。')
    return out


def effective(meeting) -> dict:
    """Every knob's value for a meeting: the default unless the settings say otherwise (`meeting` is a MeetingSettings)."""
    values = dict(DEFAULTS)
    values.update({key: value for key, value in (meeting.advanced or {}).items() if key in BY_KEY and key not in TOP_LEVEL})
    for key in TOP_LEVEL:
        values[key] = getattr(meeting, key)
    return values


def of_params(params) -> dict:
    """Every knob's value as a tracker's `Params` holds it: what a meeting really runs with (which is not always what the settings say,
    e.g. the tab's switch of the background judgement sets its threshold to 0)."""
    return {key: getattr(params.units if key in UNIT_KEYS else params, key) for key in DEFAULTS}


def differing(values: dict) -> dict:
    """The knobs among `values` that are not at their default."""
    return {key: values[key] for key in DEFAULTS if values[key] != DEFAULTS[key]}


# What moving a number was measured to do on the 15 labelled recordings (2–3 people, family and friends playing a consultation: few and
# not real meetings). "人數對" is how many of the 15 recordings came out with the right number of speakers; precision and coverage are
# of the words that carry a speaker. Only the numbers that were tried are here; the pop-up shows them under the description.
MEASURED = {
    'unknown_percentile': '實測：0→涵蓋 97.0%、5→95.5%、10（預設）→94.6%、20→89.7%、30→82.5%；精確率都在 99.7% 上下（30 時 99.9%）。',
    'new_cos': '實測：0.35→人數對 11/15 場，0.40→14/15，0.45（預設）→15/15，0.50→14/15（多分出一位），0.55→13/15。',
    'link_cos': '實測：0.40→人數對 13/15 場，0.45→14/15，0.50（預設）→15/15，0.55→11/15，0.60→5/15（多半少分出人）。',
    'assign_cos': '實測：0.45→精確率 99.1%，0.55（預設）→99.7%，0.60 起有一場多分出一位。',
    'merge_cos': '實測：0.80–0.95 沒有差別（這批錄音沒有出現相似到這種程度的兩位說話者）。',
    'first_seconds': '實測：2–4（預設 4）沒有差別，6→人數對 14/15，8→13/15。',
    'first_units': '實測：4→精確率 99.4%，6（預設）→99.7%，8→人數對 14/15，12→10/15。',
    'first_sentences': '實測：1–4 沒有差別。',
    'refit_seconds': '實測：30→涵蓋 94.9%（略高），60（預設）→94.6%，120→94.6%。',
    'update_cos': '實測：0.45→人數對 14/15，0.55（預設）→15/15，0.65→涵蓋降到 91.2%。',
    'tight_cos': '實測：0.5–0.7 沒有差別。',
    'abs_conf': '實測：0–0.06 沒有差別。',
    'target': '實測：0.5→涵蓋降到 85.6%、人數對 12/15；0.8（預設）→94.6%、15/15；1.2→涵蓋升到 96.4%，但人數對 14/15（一場有兩個人被併成一位）。',
    'pause': '實測：0.2→精確率 99.2%；0.3（預設）與 0.5→99.7%。',
    'min_unit': '實測：0.3→精確率 98.9%；0.4（預設）→99.7%；0.6→人數對 14/15。',
    'max_merged': '實測：1.0→人數對 14/15；1.3（預設）→15/15；2.0→精確率 98.8%。',
    'sent_pause': '實測：0.3→涵蓋 92.2%、精確率 99.2%；0.5（預設）→94.6%、99.7%；0.8→94.2%、99.5%。',
    'background_rel_db': '實測（標好背景的 53 句、405 句正常發言）：4 dB→抓到 91%、誤標 1.2%；5→91%、0.7%；6（預設）→85%、0.5%；8→66%、0.2%（相似度上限 0.45）。',
    'background_max_cos': '實測：6 dB 時，0.40→抓到 81%、誤標 0.2%；0.45（預設）→85%、0.5%；0.50→85%、1.0%。',
}
