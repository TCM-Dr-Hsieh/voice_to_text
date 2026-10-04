/* Run with `node tests/writing_smoke.js`; no browser or microphone required. */
const assert = require('node:assert/strict');

class Element {
  constructor() {
    this.children = [];
    this.value = '';
    this.dataset = {};
    this.isConnected = true;
  }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  addEventListener() {}
  querySelector() { return null; }
  setAttribute() {}
  focus() {}
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
}
const host = new Element();
global.document = {
  getElementById(id) {
    return id === 'writing-host' ? host : null;
  },
  createElement() { return new Element(); },
  createTextNode(text) { return {textContent: text}; },
  querySelector() { return null; },
  addEventListener() {},
};
global.window = {addEventListener() {}};
globalThis.OpenCC = require('../voice_app/opencc.full.js');
require('../voice_app/script_converter.js');
require('../voice_app/writing.js');

const writing = window.writing;
writing.setMode(true, 100);
writing.editor.value = '甲乙丙';
writing.editor.setSelectionRange(1, 2);
assert.deepEqual(writing.begin(100), {before: '甲', after: '丙'});
writing.render('新', '內容');
assert.equal(writing.preview.children.map(child => child.textContent).join(''), '甲新內容丙');
assert.equal(writing.editor.readOnly, true);
writing.commit('新內容');
assert.equal(writing.editor.value, '甲新內容丙');
assert.equal(writing.editor.selectionStart, 4);
assert.equal(writing.dirty(), true);

writing.editor.setSelectionRange(2, 2);
assert.deepEqual(writing.begin(100), {before: '甲新', after: '內容丙'});
writing.rollback();
assert.equal(writing.editor.value, '甲新內容丙');
assert.deepEqual([writing.editor.selectionStart, writing.editor.selectionEnd], [2, 2]);

writing.editor.setSelectionRange(1, 4);
writing.begin(100);
writing.commit('');
assert.equal(writing.editor.value, '甲新內容丙');
assert.deepEqual([writing.editor.selectionStart, writing.editor.selectionEnd], [1, 4]);
assert.equal(writing.editor.readOnly, false);
assert.equal(writing.editor.hidden, false);
assert.equal(writing.preview.hidden, true);

writing.editor.value = '小兒夜尿，以中藥治療。';
writing.editor.setSelectionRange(2, 2);
assert.deepEqual(writing.setScript('simplified'), {status: 'converted', mode: 'simplified'});
assert.equal(writing.editor.value, '小儿夜尿，以中药治疗。');
assert.equal(writing.editor.selectionStart, 2);
assert.equal(writing.dirty(), true);
assert.equal(window.scriptConverter.convert('兒童頭髮發展'), '儿童头发发展');

writing.begin(100);
assert.deepEqual(writing.setScript('traditional'), {status: 'unavailable'});
writing.render('中藥', '治療');
assert.equal(writing.preview.children.map(child => child.textContent).join(''),
  '小儿中药治疗夜尿，以中药治疗。');
writing.commit('中藥治療');
assert.equal(writing.editor.value, '小儿中药治疗夜尿，以中药治疗。');
assert.deepEqual(writing.setScript('traditional'), {status: 'converted', mode: 'traditional'});
assert.equal(writing.editor.value, '小兒中藥治療夜尿，以中藥治療。');

writing.editor.value = '頭髮 乾杯';
writing.editor.setSelectionRange(2, 2);
assert.deepEqual(writing.setScript('simplified'), {status: 'converted', mode: 'simplified'});
assert.equal(writing.editor.value, '头发 干杯');
assert.deepEqual(writing.setScript('simplified'), {status: 'restored', mode: 'original'});
assert.equal(writing.editor.value, '頭髮 乾杯');
assert.deepEqual([writing.editor.selectionStart, writing.editor.selectionEnd], [2, 2]);
assert.deepEqual(writing.setScript('simplified'), {status: 'converted', mode: 'simplified'});
assert.deepEqual(writing.setScript('traditional'), {status: 'converted', mode: 'traditional'});
assert.equal(writing.editor.value, '頭髮 乾杯');
assert.deepEqual(writing.setScript('traditional'), {status: 'restored', mode: 'original'});

assert.deepEqual(writing.setScript('simplified'), {status: 'converted', mode: 'simplified'});
writing.editor.value += '新內容';
assert.deepEqual(writing.setScript('simplified'), {status: 'edited'});
assert.equal(writing.editor.value, '头发 干杯新內容');
assert.equal(window.scriptConverter.mode, 'simplified');

(async () => {
  // Opening a traditional TXT while simplified is selected must keep the file unchanged.
  writing.savedText = writing.editor.value;
  let savedBytes;
  const handle = {
    name: '病歷.txt',
    getFile: async () => ({name: '病歷.txt', arrayBuffer: async () => Buffer.from('頭髮 乾杯', 'utf8')}),
    createWritable: async () => ({
      write: async blob => { savedBytes = Buffer.from(await blob.arrayBuffer()); },
      close: async () => {},
    }),
  };
  window.showOpenFilePicker = async () => [handle];
  await writing.open();
  assert.equal(writing.editor.value, '頭髮 乾杯');
  assert.equal(writing.savedText, '頭髮 乾杯');
  assert.equal(writing.dirty(), false);
  assert.equal(writing.articleMode, 'original');
  assert.equal(await writing.save(), true);
  assert.deepEqual(savedBytes, Buffer.from('\uFEFF頭髮 乾杯', 'utf8'));
  assert.deepEqual(writing.setScript('simplified'), {status: 'converted', mode: 'simplified'});
  assert.equal(writing.editor.value, '头发 干杯');
  assert.deepEqual(writing.setScript('simplified'), {status: 'restored', mode: 'original'});
  assert.equal(writing.editor.value, '頭髮 乾杯');
  assert.equal(writing.dirty(), false);
  writing.editor.setSelectionRange(writing.editor.value.length, writing.editor.value.length);
  writing.begin(100);
  writing.commit('語音');
  assert.equal(writing.editor.value, '頭髮 乾杯語音');
  console.log('writing insert/replace/rollback/script conversion passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
