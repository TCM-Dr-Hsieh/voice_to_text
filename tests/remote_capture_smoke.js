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
  capture.context = {sampleRate: 48000, close() {}, createMediaStreamSource: () => ({connect() {}})};
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
  const audio = installFakeAudio();
  await capture.start('remote_microphone', 'test-token');
  assert.equal(typeof startupDeadline, 'function', alertText);
  assert.equal(capture.sessionId, 'startup-deadline-1234');
  startupDeadline();
  assert.equal(capture.sessionId, null);
  assert.equal(rolledBack, true);
  assert.equal(stopped, true);
  assert.match(alertText, /375 秒/);

  // Once the models are loaded the server asks whether this browser is still here ('prepared'): its 'go' is what starts the
  // recording. Only the socket the browser is using answers; a socket it has already given up on does not.
  capture.sessionId = 'confirm-session-1234';
  capture.kind = 'remote_microphone';
  capture.token = 'test-token';
  capture.openSocket(false, {before: '', after: ''});
  const confirming = FakeWebSocket.sockets.at(-1);
  confirming.open();
  await confirming.message({type: 'preparing'});
  assert.equal(confirming.sent.filter(value => JSON.parse(value).type === 'go').length, 0);
  await confirming.message({type: 'prepared'});
  assert.deepEqual(JSON.parse(confirming.sent.at(-1)), {type: 'go'});
  const abandoned = capture.socket;
  capture.socket = null;                                                  // the browser moved on to another socket
  const before = confirming.sent.length;
  await abandoned.message({type: 'prepared'});
  assert.equal(abandoned.sent.length, before, 'a socket that is no longer in use must not answer');
  capture.cleanup();

  // The browser's audio pipeline is built, and running, BEFORE the server is asked to start (so a pipeline that cannot start
  // never reaches the server: the last meeting stays, the article cursor is given back). The shared audio only goes in after
  // the server says ready.
  const socketsBefore = FakeWebSocket.sockets.length;
  rolledBack = false;
  audio.log.length = 0;
  global.setTimeout = nativeSetTimeout;
  await capture.start('remote_microphone', 'test-token');
  assert.equal(FakeWebSocket.sockets.length, socketsBefore + 1, alertText);
  assert.deepEqual(audio.log, ['context', 'addModule', 'node', 'resume'], 'pipeline up, nothing fed in yet');
  const started = FakeWebSocket.sockets.at(-1);
  started.open();
  assert.equal(JSON.parse(started.sent[0]).type, 'start');
  await started.message({type: 'ready', next_sequence: 0});
  assert.deepEqual(audio.log, ['context', 'addModule', 'node', 'resume', 'source', 'connect']);
  assert.equal(JSON.parse(started.sent[1]).type, 'format');
  await started.message({type: 'resumed', next_sequence: 0});             // a reconnect never feeds the audio in twice
  assert.equal(audio.log.filter(entry => entry === 'source').length, 1);
  capture.cleanup();

  // A new start closes the replay window of the meeting view first: its sound would be recorded into the run that starts (the shared
  // system audio captures it for certain). A start that is refused because one is running, and a reconnect, leave it alone.
  const replayCalls = [];
  window.meetingReplay = {close: () => replayCalls.push('close')};
  await capture.start('remote_microphone', 'test-token');
  assert.deepEqual(replayCalls, ['close'], alertText);
  await capture.start('remote_microphone', 'test-token');                  // one is running: refused
  assert.deepEqual(replayCalls, ['close']);
  capture.openSocket(true);                                                // a reconnect of the same run
  assert.deepEqual(replayCalls, ['close']);
  capture.cleanup();
  await capture.start('remote_microphone', 'test-token');
  assert.deepEqual(replayCalls, ['close', 'close']);                       // the next run closes it again
  capture.cleanup();
  delete window.meetingReplay;
  await capture.start('remote_microphone', 'test-token');                  // a page without the replay script still starts
  assert.notEqual(capture.sessionId, null);
  capture.cleanup();
  // A player that cannot be closed (its audio or its window is in a state that throws) must not stop a run from starting, nor leave
  // the start "pending" for ever, which would refuse every start after it.
  window.meetingReplay = {close: () => { throw new Error('the player is broken'); }};
  await capture.start('remote_microphone', 'test-token');
  assert.equal(capture.pending, false, alertText);
  assert.notEqual(capture.sessionId, null, alertText);
  capture.cleanup();
  await capture.start('remote_microphone', 'test-token');                  // and the next start is not refused either
  assert.equal(capture.pending, false);
  assert.notEqual(capture.sessionId, null);
  capture.cleanup();
  delete window.meetingReplay;

  for (const [failure, reason] of [['addModule', /worklet/], ['node', /AudioWorkletNode/], ['resume', /not allowed/]]) {
    const sockets = FakeWebSocket.sockets.length;
    audio.failure = failure;
    audio.log.length = 0;
    rolledBack = stopped = false;
    alertText = '';
    await capture.start('remote_microphone', 'test-token');
    assert.equal(FakeWebSocket.sockets.length, sockets, `${failure}: the server must not be contacted`);
    assert.equal(capture.sessionId, null);
    assert.equal(rolledBack, true, `${failure}: the article cursor is given back`);
    assert.equal(stopped, true, `${failure}: the shared audio is released`);
    assert.match(alertText, reason);
    assert.equal(capture.everReady, false);
  }

  // A pipeline that never comes up is given up on after a while, again without touching the server.
  audio.failure = 'hang';
  const timers = {};
  global.setTimeout = (callback, ms) => {
    if (ms === 10000 || ms === 375000) { timers[ms] = callback; return 999999; }
    return nativeSetTimeout(callback, ms);
  };
  const sockets = FakeWebSocket.sockets.length;
  rolledBack = false;
  alertText = '';
  const starting = capture.start('remote_microphone', 'test-token');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(typeof timers[10000], 'function');
  assert.equal(timers[375000], undefined);
  timers[10000]();
  await starting;
  assert.equal(FakeWebSocket.sockets.length, sockets);
  assert.equal(rolledBack, true);
  assert.match(alertText, /逾時/);

  // The attempt that timed out may still finish later, in the middle of the NEXT start (or fail much later): it must build and
  // publish nothing, so the next start keeps its own context, node and audio.
  for (const lateOutcome of ['resolve', 'reject']) {
    audio.failure = null;
    audio.late.length = audio.nodes.length = audio.contexts.length = 0;
    audio.behaviours = ['late', 'late'];
    const first = capture.start('remote_microphone', 'test-token');       // its worklet module is still loading
    await new Promise(resolve => setImmediate(resolve));
    timers[10000]();                                                      // ... and it times out
    await first;
    assert.equal(capture.sessionId, null);
    const second = capture.start('remote_microphone', 'test-token');      // the user tries again
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(audio.late.length, 2);
    audio.late[0][lateOutcome](new Error('the old module finally answered'));     // the abandoned attempt wakes up now
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(audio.nodes.length, 0, `${lateOutcome}: the abandoned attempt must not build a node`);
    assert.ok(!capture.node, `${lateOutcome}: nothing is published for the second attempt yet`);
    audio.late[1].resolve();                                              // the second attempt's own module
    await second;
    assert.equal(audio.nodes.length, 1);
    assert.equal(capture.node, audio.nodes[0]);
    assert.equal(capture.context, audio.contexts[1], `${lateOutcome}: the context is the second attempt's`);
    assert.notEqual(capture.sessionId, null);
    const next = FakeWebSocket.sockets.at(-1);
    next.open();
    await next.message({type: 'ready', next_sequence: 0});
    assert.equal(JSON.parse(next.sent[1]).type, 'format');
    assert.equal(audio.log.at(-1), 'connect');                            // the shared audio went into the second node
    capture.cleanup();
  }
  global.setTimeout = nativeSetTimeout;
}

function installFakeAudio() {
  // `behaviours` is consumed one entry per addModule call: 'late' makes that call wait until `late[n].resolve()` / `.reject()`.
  const audio = {failure: null, log: [], behaviours: [], late: [], contexts: [], nodes: []};
  global.AudioContext = class {
    constructor() {
      audio.log.push('context');
      audio.contexts.push(this);
      this.sampleRate = 48000;
      this.destination = {};
      this.audioWorklet = {addModule: async () => {
        audio.log.push('addModule');
        if (audio.behaviours.shift() === 'late') await new Promise((resolve, reject) => audio.late.push({resolve, reject}));
        if (audio.failure === 'addModule') throw new Error("Unable to load a worklet's module.");
        if (audio.failure === 'hang') await new Promise(() => {});
      }};
    }
    async resume() {
      audio.log.push('resume');
      if (audio.failure === 'resume') throw new Error('The AudioContext was not allowed to start.');
    }
    createMediaStreamSource() {
      audio.log.push('source');
      return {connect() { audio.log.push('connect'); }};
    }
    close() {}
  };
  global.AudioWorkletNode = class {
    constructor() {
      audio.log.push('node');
      if (audio.failure === 'node') throw new Error('AudioWorkletNode could not be created.');
      this.port = {postMessage() {}};
      audio.nodes.push(this);
    }
    connect() {}
    disconnect() {}
  };
  global.URL.createObjectURL = () => 'blob:fake';
  global.URL.revokeObjectURL = () => {};
  return audio;
}

main().catch(error => { console.error(error); process.exit(1); });          // (at once: a timer left by the failed step would keep the process for minutes)
