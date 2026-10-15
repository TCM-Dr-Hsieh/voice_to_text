/* Run with `node tests/meeting_smoke.js`; no browser needed. A minimal DOM is enough for the meeting view. */
const assert = require('node:assert/strict');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.isConnected = true;
    this.className = '';
    this.own = '';
    this.scrollTop = 0;
    this.scrollHeight = 2000;
    this.clientHeight = 500;
    this.listeners = {};
    this.title = '';
    this.tabIndex = -1;
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  click() { for (const handler of this.listeners.click || []) handler({}); }
  get classList() {
    const element = this;
    return {
      add(name) { this.toggle(name, true); },
      toggle(name, on) {
        const names = new Set(element.className.split(/\s+/).filter(Boolean));
        if (on) names.add(name); else names.delete(name);
        element.className = [...names].join(' ');
      },
      contains(name) { return element.className.split(/\s+/).includes(name); },
    };
  }
  set textContent(value) { this.own = String(value); this.children = []; }
  get textContent() { return this.own + this.children.map(child => child.textContent).join(''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  insertBefore(element, reference) {
    const at = reference ? this.children.indexOf(reference) : -1;
    if (at < 0) this.children.push(element); else this.children.splice(at, 0, element);
  }
  setAttribute() {}
}
const host = new Element();
global.document = {
  getElementById: id => (id === 'meeting-host' ? host : null),
  createElement: tag => new Element(tag),
  createTextNode: text => ({textContent: text, children: []}),
};
global.window = {};
globalThis.OpenCC = require('../voice_app/opencc.full.js');
require('../voice_app/script_converter.js');
require('../voice_app/meeting.js');

const events = [];
global.emitEvent = (name, payload) => events.push([name, payload]);          // what NiceGUI gives the page: the click goes to the server
const view = window.meeting;
const labels = {marking: true, names: {'0': '王主持人', '1': 'Speaker-002'}, unknown: '不明', voice: '語音', background: '背景',
                tnames: {'0': 'Host Wang', '1': 'Speaker-002'}, tunknown: 'Unknown', tvoice: 'Voice'};
const rows = () => view.scroller.children;
const left = i => rows()[i].children[0];
const right = i => rows()[i].children[1];
const has = (element, name) => element.className.split(/\s+/).includes(name);

// 1. rows arrive, in any order, and are kept in order; the newest are marked revisable
assert.equal(view.update({reset: true, labels, rows: [
  {i: 2, rev: false, runs: [{g: 1, t: '很高興認識你。', s: 'voice'}]},
  {i: 1, rev: false, runs: [{g: 0, t: '你好，我是王先生。', s: 'voice'}],
   tr: {st: 'ok', err: '', runs: [{g: 0, t: 'Hello, I am Mr. Wang.'}]}},
  {i: 3, rev: true, runs: [{g: null, t: '嗯。', s: ''}, {g: 0, t: '對。', s: 'text'}]},
]}), true);
assert.deepEqual(rows().map(row => row.dataset.i), ['1', '2', '3']);
assert.equal(left(0).textContent, '語音#1王主持人: 你好，我是王先生。');
assert.equal(right(0).textContent, 'Voice#1Host Wang: Hello, I am Mr. Wang.');
assert.equal(right(1).textContent, '');                                  // not translated yet: nothing on the right
assert.equal(left(1).textContent, '語音#2Speaker-002: 很高興認識你。');
assert.equal(left(2).children.length, 2);                                // two speakers in one segment: two lines, the number once
assert.equal(left(2).children[0].textContent, '語音#3不明: 嗯。');
assert.equal(left(2).children[1].textContent, '王主持人*: 對。');         // a speaker read from the text carries a star
assert.equal(has(left(2).children[0], 'm-unknown'), true);
assert.equal(has(left(2), 'm-rev'), true);
assert.equal(has(left(0), 'm-rev'), false);

// 2. a row that arrives again replaces its content; an old number is inserted at its place
view.update({rows: [{i: 3, rev: false, runs: [{g: 0, t: '好。', s: 'voice'}]}, {i: 0, rev: false, runs: [{g: 1, t: '開場。', s: 'voice'}]}]});
assert.deepEqual(rows().map(row => row.dataset.i), ['0', '1', '2', '3']);
assert.equal(left(3).textContent, '語音#3王主持人: 好。');
assert.equal(has(left(3), 'm-rev'), false);

// 3. a rename changes every label without any row being sent
view.update({labels: {...labels, names: {'0': '李主任', '1': 'Speaker-002'}, tnames: {'0': 'Director Li', '1': 'Speaker-002'}}});
assert.equal(left(3).textContent, '語音#3李主任: 好。');
assert.equal(right(1).textContent, 'Voice#1Director Li: Hello, I am Mr. Wang.');

// 4. the translation of a changed segment is marked stale, a failed one says so, an empty run is not shown
view.update({rows: [
  {i: 1, rev: false, runs: [{g: 0, t: '你好。', s: 'voice'}], tr: {st: 'stale', err: '', runs: [{g: 0, t: 'Hi.'}]}},
  {i: 2, rev: false, runs: [{g: 1, t: '嗯。', s: 'voice'}], tr: {st: 'error', err: 'server down', runs: []}},
  {i: 3, rev: false, runs: [{g: 0, t: '甲。', s: 'voice'}, {g: 1, t: '乙。', s: 'voice'}],
   tr: {st: 'ok', err: '', runs: [{g: 0, t: 'A and B.'}, {g: 1, t: ''}]}},
]});
assert.equal(has(right(1), 'm-stale'), true);
assert.match(right(2).textContent, /翻譯失敗.*server down/);
assert.equal(right(3).children.length, 1);

// 5. text is never taken for HTML
view.update({rows: [{i: 9, rev: false, runs: [{g: 0, t: '<img src=x onerror=alert(1)>', s: 'voice'}]}]});
assert.equal(left(rows().length - 1).textContent, '語音#9李主任: <img src=x onerror=alert(1)>');

// 6. without speaker marking a line is just the number and the text
view.update({labels: {...labels, marking: false}});
assert.equal(left(1).textContent, '語音#1你好。');
assert.equal(right(3).textContent, 'Voice#3A and B.');
view.update({labels});

// 7. the display can be switched between the two scripts without touching the data
view.update({rows: [{i: 10, rev: false, runs: [{g: 0, t: '頭髮與裡面', s: 'voice'}]}]});
view.setScript('simplified');
assert.equal(left(rows().length - 1).textContent, '語音#10王主持人: 头发与里面');
view.setScript('original');
assert.equal(left(rows().length - 1).textContent, '語音#10王主持人: 頭髮與裡面');

// 8. the view follows the newest row only while the reader is at the bottom
view.scroller.scrollHeight = 3000; view.scroller.scrollTop = 2500; view.scroller.clientHeight = 500;
view.update({rows: [{i: 11, rev: true, runs: [{g: 0, t: '新的。', s: 'voice'}]}]});
assert.equal(view.scroller.scrollTop, 3000);
view.scroller.scrollTop = 100;
view.update({rows: [{i: 12, rev: true, runs: [{g: 0, t: '更新的。', s: 'voice'}]}]});
assert.equal(view.scroller.scrollTop, 100);

// 9. a new meeting starts from an empty table
view.update({reset: true, labels, rows: [{i: 1, rev: true, runs: [{g: 0, t: '新會議。', s: 'voice'}]}]});
assert.deepEqual(rows().map(row => row.dataset.i), ['1']);

// 10. background talk: grey, labelled 背景, left out of the translation column, and a click on the label unmarks it; a click on an
//     unknown speaker marks it; a known speaker's label does nothing
view.update({reset: true, labels, rows: [
  {i: 1, rev: false, runs: [{g: 0, t: '各位好。', s: 'voice'}, {g: null, t: '欸你看這個。', s: 'background'}, {g: null, t: '嗯。', s: ''}],
   tr: {st: 'ok', err: '', runs: [{g: 0, t: 'Hello everyone.'}]}},
  {i: 2, rev: false, runs: [{g: null, t: '小聲說話。', s: 'background'}], tr: {st: 'ok', err: '', runs: []}},
]});
assert.equal(left(0).children.length, 3);
assert.equal(left(0).children[1].textContent, '背景: 欸你看這個。');
assert.equal(has(left(0).children[1], 'm-bg'), true);
assert.equal(has(left(0).children[0], 'm-bg'), false);
assert.equal(right(0).textContent, 'Voice#1Host Wang: Hello everyone.');   // the background run has no translation
assert.equal(right(1).textContent, '');                                    // a segment of background talk only: nothing on the right
const label = line => line.children[1];                                    // number, label, ': ', text
assert.equal(label(left(0).children[0]).listeners.click, undefined);       // a known speaker: not clickable
label(left(0).children[1]).click();                                        // background -> "not background"
label(left(0).children[2]).click();                                        // unknown -> "background"
assert.deepEqual(events, [['meeting_background', {i: 1, r: 1, t: '欸你看這個。', to: false}],
                          ['meeting_background', {i: 1, r: 2, t: '嗯。', to: true}]]);
view.update({labels: {...labels, marking: false}});                        // without marking nothing is offered
assert.equal(left(0).children[1].children.length, 2);                      // only the number and the text: no label to click
assert.equal(left(0).children[1].textContent, '欸你看這個。');
// 11. every speaker has the colour of its number, in both columns, and the colours start again from the first when they are used up;
//     unknown speakers and background talk have no colour of their own
view.update({reset: true, labels, rows: [
  {i: 1, rev: false, runs: [{g: 0, t: '甲。', s: 'voice'}, {g: 1, t: '乙。', s: 'voice'}, {g: 10, t: '丙。', s: 'voice'}, {g: 11, t: '丁。', s: 'text'},
                            {g: null, t: '戊。', s: ''}, {g: null, t: '己。', s: 'background'}],
   tr: {st: 'ok', err: '', runs: [{g: 0, t: 'A.'}, {g: 11, t: 'D.'}, {g: null, t: 'E.'}]}},
]});
const colours = element => element.children.map(line => line.className.split(/\s+/).find(name => /^m-c\d+$/.test(name)) || '');
assert.deepEqual(colours(left(0)), ['m-c0', 'm-c1', 'm-c0', 'm-c1', '', '']);
assert.deepEqual(colours(right(0)), ['m-c0', 'm-c1', '']);
view.update({labels: {...labels, marking: false}});                       // the marking stopped: the rows kept on the page lose all of it
const looks = element => element.children.map(line => line.className);
assert.deepEqual(looks(left(0)), Array(6).fill('m-line'));                // no colour, no grey of unknown or background talk
assert.deepEqual(looks(right(0)), Array(3).fill('m-line'));
view.update({labels});                                                    // (and it comes back with it)
assert.deepEqual(colours(left(0)), ['m-c0', 'm-c1', 'm-c0', 'm-c1', '', '']);
assert.ok(has(left(0).children[5], 'm-bg') && has(left(0).children[4], 'm-unknown'));
view.update({labels: {...labels, palette: 4}});                           // the server says how many colours there are
assert.deepEqual(colours(left(0)).slice(0, 4), ['m-c0', 'm-c1', 'm-c2', 'm-c3']);
view.update({labels: {...labels, palette: 4}, reset: true, rows: [{i: 1, rev: false, runs: [{g: 6, t: '甲。', s: 'voice'}]}]});
assert.deepEqual(colours(left(0)), ['m-c2']);                              // 6 % 4

// 12. a draft translation is one text for the segment with no speaker label (who speaks is not settled yet); the newest rows are
//     yellow on the right too (it may still change); the finished translation, one line per speaker, replaces it
view.update({reset: true, labels, rows: [
  {i: 1, rev: true, runs: [{g: 0, t: '甲。', s: 'voice'}, {g: 1, t: '乙。', s: 'voice'}],
   tr: {st: 'draft', err: '', runs: [{g: null, t: 'First and second.'}]}},
  {i: 2, rev: false, runs: [{g: 0, t: '丙。', s: 'voice'}], tr: {st: 'draft', err: '', runs: [{g: null, t: 'Third.'}]}},
]});
assert.equal(right(0).textContent, 'Voice#1First and second.');                // no "Unknown:" in front of it
assert.equal(right(0).children.length, 1);
assert.deepEqual(colours(right(0)), ['']);                                      // and no speaker colour
assert.equal(has(right(0), 'm-rev'), true);
assert.equal(has(right(1), 'm-rev'), false);                                    // out of the newest rows: a draft that stays, not yellow
assert.equal(right(1).textContent, 'Voice#2Third.');
view.update({rows: [{i: 1, rev: false, runs: [{g: 0, t: '甲。', s: 'voice'}, {g: 1, t: '乙。', s: 'voice'}],
                     tr: {st: 'ok', err: '', runs: [{g: 0, t: 'First.'}, {g: 1, t: 'Second.'}]}}]});
assert.equal(right(0).textContent, 'Voice#1Host Wang: First.Speaker-002: Second.');
assert.equal(has(right(0), 'm-rev'), false);
view.update({labels: {...labels, marking: false}});                              // without marking a draft is still just the text
assert.equal(right(1).textContent, 'Voice#2Third.');
view.update({labels});

// 13. the number of a segment plays what was said there again (the player is replay.js; here a stand-in that records what it is told)
const opened = [], synced = [];
window.meetingReplay = {open: info => opened.push(info), sync: replay => synced.push(replay)};
const replayLabels = {...labels, replay: {token: 'tok-1', blocked: ''}};
view.update({reset: true, labels: replayLabels, rows: [
  {i: 1, rev: false, t: 0, runs: [{g: 0, t: '甲。', s: 'voice'}, {g: 1, t: '乙。', s: 'voice'}],
   tr: {st: 'ok', err: '', runs: [{g: 0, t: 'A.'}, {g: 1, t: 'B.'}]}},
  {i: 2, rev: false, t: 9, runs: [{g: 1, t: '丙。', s: 'voice'}], tr: {st: 'draft', err: '', runs: [{g: null, t: 'C.'}]}},
  {i: 3, rev: false, t: 12.5, runs: [{g: null, t: '丁。', s: 'background'}], tr: {st: 'error', err: 'down', runs: []}},
]});
const number = line => line.children[0];
assert.equal(has(number(left(0).children[0]), 'm-play'), true);
assert.equal(has(number(left(0).children[1]), 'm-play'), false);                   // only the first line of a segment has the number
assert.equal(number(left(0).children[0]).title, '點擊回聽這一段（從這裡起 30 秒）');
number(left(1).children[0]).listeners.click[0]();                                  // the transcript side ...
number(right(1).children[0]).listeners.click[0]();                                 // ... and the translation side (a draft) play the same
number(right(2).children[0]).listeners.click[0]();                                 // (a translation that failed: the audio is still there)
number(right(0).children[0]).listeners.click[0]();
assert.deepEqual(opened, [{start: 9, i: 2, token: 'tok-1', blocked: ''}, {start: 9, i: 2, token: 'tok-1', blocked: ''},
                          {start: 12.5, i: 3, token: 'tok-1', blocked: ''}, {start: 0, i: 1, token: 'tok-1', blocked: ''}]);
number(left(1).children[0]).listeners.keydown[0]({key: 'Enter'});
number(left(1).children[0]).listeners.keydown[0]({key: 'a'});
assert.equal(opened.length, 5);                                                    // the Enter key plays it, another key does not
assert.deepEqual(view.rowAt(10), {i: 2, t: 9, label: '語音#2', text: 'Speaker-002: 丙。'});
assert.deepEqual(view.rowAt(0.05), {i: 1, t: 0, label: '語音#1', text: '王主持人: 甲。\nSpeaker-002: 乙。'});
assert.equal(view.rowAt(13).text, '背景: 丁。');
assert.equal(view.rowAt(-1), null);
view.scroller.getBoundingClientRect = () => ({top: 100, bottom: 600});
rows()[1].getBoundingClientRect = () => ({top: 900, bottom: 940});                // the row that plays is below what the view shows
view.scroller.scrollTop = 0;
view.setPlaying(2);
assert.equal(has(rows()[1], 'm-playing'), true);
assert.equal(view.scroller.scrollTop, 760);                                        // it is brought into view (900 - 100 - 40 from the top)
view.setPlaying(2);
assert.equal(view.scroller.scrollTop, 760);                                        // the same row again: nothing moves
rows()[2].getBoundingClientRect = () => ({top: 300, bottom: 340});                 // a row that is in view is not scrolled to
view.setPlaying(3);
assert.equal(view.scroller.scrollTop, 760);
view.setPlaying(2);
view.setPlaying(3);
assert.deepEqual([0, 1, 2].map(k => has(rows()[k], 'm-playing')), [false, false, true]);   // one row at a time
view.setPlaying(null);
assert.deepEqual([0, 1, 2].map(k => has(rows()[k], 'm-playing')), [false, false, false]);
view.setPlaying(99);                                                               // a row that is not there: nothing plays
assert.equal(view.playing, null);
view.update({labels: {...replayLabels, replay: {token: 'tok-2', blocked: ''}}});   // another meeting: the player is told
assert.deepEqual(synced.at(-1), {token: 'tok-2', blocked: ''});
// a meeting whose audio cannot be replayed: the number says why and still opens the player, which says it
view.update({labels: {...replayLabels, replay: {token: 'tok-2', blocked: '這個音訊來源無法回聽'}}});
assert.equal(number(left(0).children[0]).title, '這個音訊來源無法回聽');
number(left(0).children[0]).listeners.click[0]();
assert.deepEqual(opened.at(-1), {start: 0, i: 1, token: 'tok-2', blocked: '這個音訊來源無法回聽'});
// without what the server sends for it (an older server, a row without its start) the numbers are plain
view.update({reset: true, labels, rows: [{i: 1, rev: false, t: 0, runs: [{g: 0, t: '甲。', s: 'voice'}]}]});
assert.equal(has(number(left(0).children[0]), 'm-play'), false);
view.update({labels: replayLabels, rows: [{i: 1, rev: false, runs: [{g: 0, t: '甲。', s: 'voice'}]}]});
assert.equal(has(number(left(0).children[0]), 'm-play'), false);                   // (no start time: nothing to play from)

console.log('meeting view smoke test passed');
