/* Run with `node tests/replay_smoke.js`; no browser needed. A minimal DOM, a fake Audio and a fake fetch are enough for the replay window. */
const assert = require('node:assert/strict');

class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.style = {}; this.listeners = {}; this.attrs = {};
    this.className = ''; this._text = ''; this.value = ''; this.disabled = false; this.title = ''; this.id = '';
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  fire(type, event = {}) { for (const handler of this.listeners[type] || []) handler({target: this, ...event}); }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute(name, value) { this.attrs[name] = value; }
  get firstChild() { return this.children[0]; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  getBoundingClientRect() { return {left: 100, top: 100}; }
}

const players = [];
class FakeAudio {
  constructor() {
    this.handlers = {}; this.paused = true; this.ended = false; this.currentTime = 0; this.duration = NaN; this.src = ''; this.plays = 0;
    players.push(this);
  }
  addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); }
  emit(type) { for (const handler of this.handlers[type] || []) handler(); }
  play() { this.paused = false; this.plays++; this.emit('play'); return Promise.resolve(); }
  pause() { if (!this.paused) { this.paused = true; this.emit('pause'); } }
  load() {}
  removeAttribute(name) { if (name === 'src') this.src = ''; }
}

const body = new Element('body');
global.document = {createElement: tag => new Element(tag), body, addEventListener: () => {}};

const rows = [{i: 1, t: 0, label: '語音#1', text: '王主持人: 甲'}, {i: 2, t: 9, label: '語音#2', text: 'Speaker-002: 乙'},
              {i: 3, t: 12.5, label: '語音#3', text: '王主持人: 丙'}];
const playing = [];
global.window = {innerWidth: 1200, innerHeight: 800,
                 meeting: {rowAt: seconds => rows.filter(row => row.t <= seconds).at(-1) || null, setPlaying: i => playing.push(i)}};
const made = [], revoked = [];
URL.createObjectURL = blob => { made.push(blob); return 'blob:' + made.length; };
URL.revokeObjectURL = url => revoked.push(url);
global.Audio = FakeAudio;
const fetched = [], answers = [];
global.fetch = url => { fetched.push(url); return new Promise(resolve => answers.push(resolve)); };
const good = (size = 10) => ({ok: true, blob: async () => ({size}), text: async () => ''});
const refused = reason => ({ok: false, blob: async () => null, text: async () => reason});
const tick = () => new Promise(resolve => setImmediate(resolve));

require('../voice_app/replay.js');
const replay = window.meetingReplay;
const panel = () => body.children[0];
const find = (node, predicate) => predicate(node) ? node : node.children.reduce((found, child) => found || find(child, predicate), null);
const byClass = name => find(panel(), node => node.className === name);
const allByClass = name => { const out = []; (function walk(node) { if (node.className === name) out.push(node); node.children.forEach(walk); })(panel()); return out; };
const shown = () => panel().style.display === 'block';

(async () => {
  // 0. what the page and the typed start share
  assert.equal(replay.parseStart('01:20'), 80);
  assert.equal(replay.parseStart('1:02:03'), 3723);
  assert.equal(replay.parseStart(' 75.5 '), 75.5);
  assert.equal(replay.parseStart('abc'), null);
  assert.equal(replay.parseStart('1:99:'), null);
  assert.equal(replay.parseStart('75:30'), 4530);                                                  // minutes past 59 are fine without hours
  assert.equal(replay.parseStart('1:59:59'), 7199);
  assert.equal(replay.parseStart('0:59'), 59);
  assert.equal(replay.parseStart('1:99'), null);                                                   // seconds are below 60
  assert.equal(replay.parseStart('0:60'), null);
  assert.equal(replay.parseStart('1:99:99'), null);                                                // with hours, minutes and seconds are below 60
  assert.equal(replay.parseStart('1:60:00'), null);
  assert.equal(replay.parseStart('1:00:60'), null);
  assert.equal(replay.clock(121), '02:01');
  assert.equal(replay.clock(3723), '1:02:03');
  assert.equal(replay.clock(-5), '00:00');
  assert.equal(players.length, 0);                                                                // (nothing is built before the first click)
  replay.sync({token: 'any'});                                                                    // (and a sync before it is harmless)

  // 1. a click: the clip of that time is asked for with the token of the meeting, and plays when it arrives
  replay.open({start: 9, i: 2, token: 'tok/1 &', blocked: ''});
  assert.equal(panel().id, 'meeting-replay');
  assert.equal(shown(), true);
  assert.deepEqual(fetched, ['/meeting-replay.wav?start=9.00&token=tok%2F1%20%26']);              // (the token is escaped)
  assert.equal(byClass('replay-msg').textContent, '載入中…');
  assert.equal(byClass('replay-title').textContent, '回聽 · 從 00:09 起');
  assert.equal(byClass('replay-start-input').value, '00:09');
  const audio = players[0];
  answers[0](good());
  await tick();
  assert.equal(byClass('replay-msg').style.display, 'none');
  assert.equal(audio.src, 'blob:1');
  assert.equal(audio.plays, 1);

  // 2. the window follows the clip: the position, the row that is playing and its text
  const [close, toggle] = allByClass('replay-btn');                                                 // (the buttons: close is in the header)
  const bar = byClass('replay-bar');
  assert.equal(bar.disabled, true);
  audio.duration = 30;
  audio.emit('loadedmetadata');
  assert.equal(bar.disabled, false);
  assert.equal(bar.max, '30');
  audio.currentTime = 3.2;                                                                        // 9 + 3.2 = 12.2 s: still the row of 9 s
  audio.emit('timeupdate');
  assert.equal(byClass('replay-time').textContent, '0:03 / 0:30');
  assert.equal(byClass('replay-now').textContent, '語音#2　00:09');
  assert.equal(byClass('replay-text').textContent, 'Speaker-002: 乙');
  assert.equal(playing.at(-1), 2);
  audio.currentTime = 3.5;                                                                        // 12.55 s: the row of 12.5 s
  audio.emit('timeupdate');
  assert.equal(byClass('replay-now').textContent, '語音#3　00:12');
  assert.equal(playing.at(-1), 3);
  assert.equal(toggle.firstChild.textContent, 'pause');

  // 3. pause and resume, and moving the time bar
  toggle.fire('click');
  assert.equal(audio.paused, true);
  assert.equal(toggle.firstChild.textContent, 'play_arrow');
  toggle.fire('click');
  assert.equal(audio.paused, false);
  bar.value = '20';
  bar.fire('input');
  assert.equal(audio.currentTime, 20);
  assert.equal(byClass('replay-time').textContent, '0:20 / 0:30');

  // 4. a clip that cannot be had says why, and leaves nothing of the clip before it
  replay.open({start: 500, i: 9, token: 'tok', blocked: ''});
  assert.equal(audio.src, '');
  assert.equal(bar.disabled, true);
  answers[1](refused('這個時間點的聲音還沒有錄到。'));
  await tick();
  assert.equal(byClass('replay-msg').textContent, '這個時間點的聲音還沒有錄到。');
  assert.equal(byClass('replay-msg').style.display, 'block');
  toggle.fire('click');
  assert.equal(audio.plays, 2);                                                                   // (nothing is loaded: the button does nothing)
  assert.equal(toggle.firstChild.textContent, 'play_arrow');

  // 5. a second click while the first clip is still coming: only the second is played
  const before = fetched.length;
  replay.open({start: 1, i: 1, token: 'tok', blocked: ''});
  replay.open({start: 2, i: 1, token: 'tok', blocked: ''});
  assert.deepEqual(fetched.slice(before), ['/meeting-replay.wav?start=1.00&token=tok', '/meeting-replay.wav?start=2.00&token=tok']);
  answers[before + 1](good(22));
  await tick();
  assert.equal(audio.src, 'blob:' + made.length);
  const second = audio.src;
  answers[before](good(11));                                                                       // the first one arrives late
  await tick();
  assert.equal(audio.src, second);
  assert.equal(made.length, 2);                                                                    // (and was never turned into a clip)

  // 5b. a clip is being read when the next click comes in: the body that arrives afterwards is not played
  let finishReading;
  const slow = {ok: true, blob: () => new Promise(resolve => { finishReading = () => resolve({size: 5}); }), text: async () => ''};
  const reading = fetched.length;
  replay.open({start: 60, i: 1, token: 'tok', blocked: ''});
  answers[reading](slow);
  await tick();                                                                                    // the answer is in, its body is still coming
  replay.open({start: 61, i: 1, token: 'tok', blocked: ''});
  answers[reading + 1](good(7));
  await tick();
  const playedNow = audio.src, clips = made.length;
  finishReading();
  await tick();
  assert.equal(audio.src, playedNow);
  assert.equal(made.length, clips);

  // 6. the start can be typed: a time, seconds, or the field as it was (which keeps the exact second)
  const field = byClass('replay-start-input');
  const go = byClass('replay-go');
  field.value = '01:05';
  go.fire('click');
  assert.equal(fetched.at(-1), '/meeting-replay.wav?start=65.00&token=tok');
  field.value = 'soon';
  const count = fetched.length;
  go.fire('click');
  assert.equal(fetched.length, count);
  assert.equal(byClass('replay-msg').textContent, '起點格式不對，請輸入 mm:ss（例如 01:20）或秒數。');
  replay.open({start: 12.5, i: 3, token: 'tok', blocked: ''});
  assert.equal(field.value, '00:12');
  go.fire('click');                                                                                // unedited: 12.5, not 12
  assert.equal(fetched.at(-1), '/meeting-replay.wav?start=12.50&token=tok');
  field.value = '00:13';
  field.fire('keydown', {key: 'Enter'});
  assert.equal(fetched.at(-1), '/meeting-replay.wav?start=13.00&token=tok');

  // 7. audio that cannot be replayed (a source whose replay would be recorded again): nothing is asked for, the reason is shown
  const asked = fetched.length;
  replay.open({start: 30, i: 5, token: 'tok', blocked: '這個音訊來源無法回聽'});
  assert.equal(fetched.length, asked);
  assert.equal(byClass('replay-msg').textContent, '這個音訊來源無法回聽');
  assert.equal(byClass('replay-title').textContent, '回聽');
  assert.equal(audio.src, '');

  // 8. closing drops the clip, stops the sound, clears the row that was playing, and a clip still coming is not played
  replay.open({start: 40, i: 6, token: 'tok', blocked: ''});
  answers.at(-1)(good());
  await tick();
  assert.equal(audio.paused, false);
  close.fire('click');
  assert.equal(shown(), false);
  assert.equal(audio.paused, true);
  assert.equal(audio.src, '');
  assert.equal(playing.at(-1), null);
  assert.ok(revoked.length >= 1);
  replay.open({start: 41, i: 6, token: 'tok', blocked: ''});
  replay.close();
  answers.at(-1)(good());                                                                          // arrives after the close
  await tick();
  assert.equal(audio.src, '');
  assert.equal(shown(), false);

  // 9. the meeting view says which meeting it shows: the player of another one is closed
  replay.open({start: 3, i: 1, token: 'tok', blocked: ''});
  replay.sync({token: 'tok', blocked: ''});
  assert.equal(shown(), true);                                                                     // the same meeting: it stays
  replay.sync({token: 'other', blocked: ''});
  assert.equal(shown(), false);
  replay.open({start: 3, i: 1, token: 'tok', blocked: ''});
  replay.sync(undefined);                                                                          // no meeting at all
  assert.equal(shown(), false);

  console.log('replay window smoke test passed');
})().catch(error => { console.error(error); process.exit(1); });
