/* Run with `node tests/remote_capture_smoke.js`; no microphone required. */
const assert = require('node:assert/strict');

global.document = {documentElement: {dataset: {}}};
global.window = {};
global.location = {protocol: 'https:', host: 'voice.example'};
const heartbeats = new Map();
let nextHeartbeat = 1;
global.setInterval = callback => {
  const id = nextHeartbeat++;
  heartbeats.set(id, callback);
  return id;
};
global.clearInterval = id => heartbeats.delete(id);

class FakeWebSocket {
  static OPEN = 1;
  static sockets = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.bufferedAmount = 0;
    this.sent = [];
    FakeWebSocket.sockets.push(this);
  }
  open() { this.readyState = FakeWebSocket.OPEN; this.onopen(); }
  send(value) { this.sent.push(value); }
  async message(value) { await this.onmessage({data: JSON.stringify(value)}); }
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    if (!this.silentClose) this.onclose();
  }
}
global.WebSocket = FakeWebSocket;
require('../voice_app/remote_capture.js');

async function main() {
  const capture = window.remoteAudio;
  capture.sessionId = 'test-session-1234';
  capture.kind = 'remote_microphone';
  capture.token = 'test-token';
  capture.node = {disconnect() {}, port: {postMessage() { capture.flushResolve?.(); }}};
  capture.context = {sampleRate: 48000, close() {}};
  capture.stream = {getTracks: () => []};
  capture.openSocket(false, {before: '', after: ''});
  const first = FakeWebSocket.sockets[0];
  first.open();
  await first.message({type: 'ready', next_sequence: 0});
  heartbeats.get(capture.heartbeatTimer)();
  assert.equal(JSON.parse(first.sent.at(-1)).type, 'ping');
  await first.message({type: 'pong'});
  capture.queueFrame(new Float32Array([.1, .2]));
  capture.queueFrame(new Float32Array([.3, .4]));
  assert.equal(capture.frames.length, 2);
  await first.message({type: 'ack', sequence: 0});
  assert.equal(capture.frames.length, 1);
  first.silentClose = true; // A half-open connection never invokes onclose.
  capture.lastServerReply = Date.now() - 15000;
  heartbeats.get(capture.heartbeatTimer)();
  assert.equal(capture.socket, null);
  capture.queueFrame(new Float32Array([.5, .6]));
  assert.equal(capture.frames.length, 2);
  await capture.stop();
  assert.equal(capture.stopRequested, true);

  clearTimeout(capture.retryTimer);
  capture.openSocket(true);
  const second = FakeWebSocket.sockets[1];
  second.open();
  assert.equal(JSON.parse(second.sent[0]).type, 'resume');
  await second.message({type: 'resumed', next_sequence: 2});
  assert.equal(JSON.parse(second.sent[1]).type, 'format');
  const resent = second.sent.filter(value => value instanceof ArrayBuffer);
  assert.equal(resent.length, 1);
  assert.equal(new DataView(resent[0]).getUint32(0, true), 2);
  assert.equal(capture.frames.length, 1);
  await second.message({type: 'ack', sequence: 2});
  assert.equal(capture.frames.length, 0);
  assert.equal(JSON.parse(second.sent.at(-1)).type, 'stop');
  await second.message({type: 'finished'});
  assert.equal(capture.sessionId, null);
  assert.equal(capture.deadlineTimer, null);

  // A connection lost before ready must retry start with the same session and context.
  capture.sessionId = 'startup-retry-1234';
  capture.kind = 'remote_microphone';
  capture.token = 'test-token';
  capture.writingContext = {before: '上文', after: '下文'};
  capture.openSocket(false);
  const third = FakeWebSocket.sockets[2];
  third.open();
  assert.equal(JSON.parse(third.sent[0]).type, 'start');
  third.silentClose = true;
  capture.lastServerReply = Date.now() - 15000;
  heartbeats.get(capture.heartbeatTimer)();
  assert.equal(capture.socket, null);
  clearTimeout(capture.retryTimer);
  capture.openSocket(true);
  const fourth = FakeWebSocket.sockets[3];
  fourth.open();
  assert.deepEqual(JSON.parse(fourth.sent[0]), {type: 'start',
    kind: 'remote_microphone', session_id: 'startup-retry-1234',
    writing_context: {before: '上文', after: '下文'}});
  await fourth.message({type: 'preparing'});
  assert.equal(capture.deadlineTimer, null);
  capture.cleanup();

  // If startup never completes, the deadline releases the writing lock.
  const nativeSetTimeout = global.setTimeout;
  let startupDeadline;
  global.setTimeout = (callback, ms) => {
    if (ms === 375000) { startupDeadline = callback; return 999999; }
    return nativeSetTimeout(callback, ms);
  };
  let rolledBack = false;
  let stopped = false;
  let alertText = '';
  window.writing = {mode: true, contextChars: 100,
    begin: () => ({before: '上文', after: '下文'}),
    rollback: () => { rolledBack = true; }};
  Object.defineProperty(global, 'navigator', {configurable: true, value: {mediaDevices: {getUserMedia: async () => ({
    getAudioTracks: () => [1],
    getTracks: () => [{addEventListener() {}, stop() { stopped = true; }}],
  })}}});
  Object.defineProperty(global, 'crypto', {configurable: true,
    value: {randomUUID: () => 'startup-deadline-1234'}});
  global.alert = message => { alertText = message; };
  await capture.start('remote_microphone', 'test-token');
  assert.equal(typeof startupDeadline, 'function', alertText);
  assert.equal(capture.sessionId, 'startup-deadline-1234');
  startupDeadline();
  assert.equal(capture.sessionId, null);
  assert.equal(rolledBack, true);
  assert.equal(stopped, true);
  assert.match(alertText, /375 秒/);
  global.setTimeout = nativeSetTimeout;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
