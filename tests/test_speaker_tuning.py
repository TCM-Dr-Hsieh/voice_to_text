"""The tunable numbers of speaker marking: the registry agrees with the code that uses it, bad values are refused, and every number
reaches the tracker it is meant for."""
import asyncio
import json
import math
import re
from dataclasses import fields

import pytest

from voice_app.config import MeetingSettings, Settings
from voice_app.speaker import tuning
from voice_app.speaker.service import SpeakerService, build_params
from voice_app.speaker.tracker import Params
from voice_app.speaker.units import UnitParams

PARAM_DEFAULTS = {f.name: getattr(Params(), f.name) for f in fields(Params) if f.name != 'units'}
UNIT_DEFAULTS = {f.name: getattr(UnitParams(), f.name) for f in fields(UnitParams)}


def test_every_knob_is_a_number_the_tracker_has_and_its_default_is_the_trackers_default():
    for knob in tuning.KNOBS:
        table = UNIT_DEFAULTS if knob.key in tuning.UNIT_KEYS else PARAM_DEFAULTS
        assert knob.key in table, knob.key
        if knob.top_level:
            assert getattr(MeetingSettings(), knob.key) == knob.default, knob.key
        else:
            assert table[knob.key] == knob.default, knob.key
    assert set(tuning.UNIT_KEYS) == set(UNIT_DEFAULTS)


def test_the_registry_is_well_formed():
    assert len({knob.key for knob in tuning.KNOBS}) == len({knob.label for knob in tuning.KNOBS}) == len(tuning.KNOBS)
    assert {knob.stage for knob in tuning.KNOBS} == set(tuning.STAGES)
    for knob in tuning.KNOBS:
        assert knob.low <= knob.default <= knob.high and knob.step > 0, knob.key
        assert knob.what and knob.higher and knob.lower, knob.key
        if knob.integer:
            assert all(float(v) == int(v) for v in (knob.default, knob.low, knob.high, knob.step)), knob.key
    assert {knob.key for knob in tuning.KNOBS if knob.star} >= {'unknown_percentile', 'new_cos', 'background_rel_db'}


def test_the_ranges_of_the_four_boxes_of_the_tab_are_the_ranges_the_settings_enforce():
    for key in tuning.TOP_LEVEL:
        knob = tuning.BY_KEY[key]
        for value in (knob.low, knob.high):
            settings = MeetingSettings()
            setattr(settings, key, int(value) if knob.integer else value)
            settings.validate()
        for value in (knob.low - knob.step, knob.high + knob.step):
            settings = MeetingSettings()
            setattr(settings, key, int(value) if knob.integer else value)
            with pytest.raises(ValueError):
                settings.validate()


def _another_value(knob):
    value = knob.default + knob.step
    return value if value <= knob.high else knob.default - knob.step


@pytest.mark.parametrize('key', [knob.key for knob in tuning.KNOBS])
def test_every_knob_reaches_the_tracker(key):
    knob = tuning.BY_KEY[key]
    value = _another_value(knob)
    settings = MeetingSettings()
    if knob.top_level:
        setattr(settings, key, value)
    else:
        settings.advanced = {key: value}
    settings.validate()
    params = build_params(settings)
    got = getattr(params.units, key) if key in tuning.UNIT_KEYS else getattr(params, key)
    assert got == pytest.approx(value) and type(got) is (int if knob.integer else float)
    assert tuning.effective(settings)[key] == pytest.approx(value) and key in tuning.differing(tuning.effective(settings))
    for other in tuning.KNOBS:                                                    # and nothing else moved
        if other.key != key:
            table = params.units if other.key in tuning.UNIT_KEYS else params
            assert getattr(table, other.key) == other.default, other.key


def test_the_service_builds_its_tracker_from_the_settings_of_the_meeting():
    settings = MeetingSettings()
    settings.advanced = {'first_units': 9, 'target': 1.1, 'new_cos': 0.4}
    settings.unknown_percentile = 20.0
    service = SpeakerService(settings=settings, embed=lambda samples: None, llm=lambda **kw: asyncio.sleep(0), segments=lambda: [])
    params = service.tracker.params
    assert (params.first_units, params.units.target, params.new_cos, params.unknown_percentile) == (9, 1.1, 0.4, 20.0)
    settings.background_filter = False
    assert build_params(settings).background_max_cos == 0.0                       # the switch of the tab still turns the judgement off


def test_only_what_differs_from_the_default_is_kept_and_what_is_not_a_knob_is_ignored():
    raw = {'new_cos': 0.45, 'link_cos': 0.55, 'first_units': 8.0, 'removed_option': 3, 'unknown_percentile': 30}
    assert tuning.validate_advanced(raw) == {'link_cos': 0.55, 'first_units': 8}      # a default is dropped; a top-level knob has its own field
    assert tuning.validate_advanced(None) == {} and tuning.validate_advanced({}) == {}


@pytest.mark.parametrize('raw, message', [
    ({'new_cos': 0.05}, '相似度'), ({'new_cos': 0.95}, '相似度'), ({'new_cos': True}, '數字'), ({'new_cos': 'high'}, '數字'),
    ({'new_cos': float('nan')}, '數字'), ({'new_cos': float('inf')}, '數字'), ({'first_units': 6.5}, '整數'), ({'first_units': 2}, '單位數'),
    ({'target': 0.2}, '單位目標長度'), ({'target': 1.4, 'max_merged': 1.2}, '最短單位 ≤'), ({'min_unit': 0.9}, '最短單位 ≤'),
    ([1, 2], '物件'),
])
def test_bad_values_are_refused_with_the_name_of_the_number(raw, message):
    with pytest.raises(ValueError, match=message):
        tuning.validate_advanced(raw)


def test_the_settings_check_the_advanced_numbers_and_keep_them_through_a_save(tmp_path):
    settings = Settings()
    settings.meeting.advanced = {'link_cos': 0.55, 'first_units': 8, 'target': 0.9, 'new_cos': 0.45}
    settings.meeting.background_rel_db = 7.5
    settings.save(tmp_path / 'settings.json')
    saved = json.loads((tmp_path / 'settings.json').read_text(encoding='utf-8'))
    assert saved['meeting']['advanced'] == {'target': 0.9, 'first_units': 8, 'link_cos': 0.55}      # the default is not written down
    again = Settings.load(tmp_path / 'settings.json')
    assert again.meeting.advanced == saved['meeting']['advanced'] and again.meeting.background_rel_db == 7.5
    saved['meeting']['advanced']['link_cos'] = 5
    (tmp_path / 'settings.json').write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError, match='算同一位新說話者'):
        Settings.load(tmp_path / 'settings.json')


def test_settings_saved_before_the_advanced_numbers_existed_load_with_the_defaults(tmp_path):
    path = tmp_path / 'settings.json'
    Settings().save(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    del data['meeting']['advanced']
    path.write_text(json.dumps(data), encoding='utf-8')
    again = Settings.load(path)
    assert again.meeting.advanced == {} and build_params(again.meeting) == build_params(MeetingSettings())


def test_what_was_measured_belongs_to_numbers_that_exist():
    assert set(tuning.MEASURED) <= set(tuning.BY_KEY)
    assert {'unknown_percentile', 'new_cos', 'link_cos', 'background_rel_db', 'background_max_cos'} <= set(tuning.MEASURED)    # the starred ones
    for key, text in tuning.MEASURED.items():
        assert text.startswith('實測') and ('（預設）' in text or '沒有差別' in text), key            # the default is marked among the values tried, or it did not matter
        if '（預設）' in text:
            marked = float(re.search(r'([\d.]+)$', text.split('（預設）')[0]).group(1))
            assert marked == pytest.approx(tuning.BY_KEY[key].default), key                             # ... and what is marked is really the default


def test_the_meeting_file_records_every_number_the_meeting_ran_with():
    settings = MeetingSettings(advanced={'link_cos': 0.55, 'first_units': 8, 'target': 1.0}, unknown_percentile=20.0, max_speakers=6)
    service = SpeakerService(settings=settings, embed=lambda samples: None, llm=lambda **kw: asyncio.sleep(0), segments=lambda: [])
    recorded = service.map_doc()['settings']
    assert set(recorded['tuning']) == set(tuning.DEFAULTS)                          # all of them, not only the ones that were changed
    assert recorded['tuning_changed'] == {'unknown_percentile': 20.0, 'max_speakers': 6, 'link_cos': 0.55, 'first_units': 8, 'target': 1.0}
    params = service.tracker.params                                                 # and they are the ones the tracker really runs with
    for key, value in recorded['tuning'].items():
        assert (params.units if key in tuning.UNIT_KEYS else params).__dict__[key] == pytest.approx(value), key
    json.dumps(recorded)


def test_with_the_background_judgement_switched_off_the_file_says_it_ran_without_one():
    settings = MeetingSettings(background_filter=False, background_max_cos=0.4, advanced={'link_cos': 0.55})
    service = SpeakerService(settings=settings, embed=lambda samples: None, llm=lambda **kw: asyncio.sleep(0), segments=lambda: [])
    recorded = service.map_doc()['settings']
    assert service.tracker.params.background_max_cos == 0.0                           # what the tab's switch does to the tracker
    assert recorded['background_filter'] is False
    assert recorded['tuning']['background_max_cos'] == 0.0                            # not the 0.4 of the box: the number it ran with
    assert recorded['tuning_changed'] == {'link_cos': 0.55, 'background_max_cos': 0.0}
    for key, value in recorded['tuning'].items():
        params = service.tracker.params
        assert (params.units if key in tuning.UNIT_KEYS else params).__dict__[key] == pytest.approx(value), key


@pytest.mark.asyncio
async def test_the_saved_meeting_holds_those_numbers(tmp_path):
    from voice_app.meeting import Meeting
    from voice_app.session import Session
    from tests.speaker_fakes import FakeEmbedder, bases
    settings = Settings()
    settings.meeting.advanced = {'new_cos': 0.5, 'target': 1.0}
    meeting = Meeting(settings, FakeEmbedder(bases(1)), path=tmp_path / 'm.json')
    session = Session()
    session.meeting = meeting
    meeting.attach(session)
    try:
        await meeting.save()
        saved = json.loads((tmp_path / 'm.json').read_text(encoding='utf-8'))['speakers']['settings']
        assert saved['tuning']['new_cos'] == 0.5 and saved['tuning']['target'] == 1.0 and saved['tuning']['link_cos'] == tuning.DEFAULTS['link_cos']
        assert saved['tuning_changed'] == {'new_cos': 0.5, 'target': 1.0}
    finally:
        await meeting.close()
