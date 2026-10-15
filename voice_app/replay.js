/* Replay: click the number of a segment in the meeting view (`語音#12`, or `Voice#12` on the translation side) to listen again to what
   was said, from where that segment starts and for at most 30 s (the clip is cut on the server, see replay.py), and read the text of
   the row that is playing meanwhile. A small window you can move by its header, with a time bar, pause / resume, a start time you can
   type, and close. It lives outside the page tree of NiceGUI and is driven here, so it keeps working while the meeting view redraws.
   The meeting view (meeting.js) gives the rows: `window.meeting.rowAt(seconds)`, `setPlaying(i)`. */
window.meetingReplay = (function () {
  const URL_PATH = '/meeting-replay.wav', ID = 'meeting-replay';
  let panel = null, audio = null, ui = {}, drag = null;
  const state = {start: 0, loaded: false, blobUrl: null, seq: 0, scrubbing: false, token: ''};

  function pad(n) { return n < 10 ? '0' + n : '' + n; }
  function clock(seconds) {                     // the same text as the times of the exported meeting text
    const s = Math.max(0, Math.floor(seconds || 0)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? h + ':' + pad(m) + ':' + pad(s % 60) : pad(m) + ':' + pad(s % 60);
  }
  function span(seconds) { const s = Math.max(0, Math.floor(seconds || 0)); return Math.floor(s / 60) + ':' + pad(s % 60); }
  function parseStart(text) {
    // mm:ss (the minutes may go past 59: 75:30), h:mm:ss, or seconds; the seconds and, with hours, the minutes are below 60
    const t = String(text).trim();
    let m = t.match(/^(\d+):(\d{1,2})$/);
    if (m) return Number(m[2]) < 60 ? Number(m[1]) * 60 + Number(m[2]) : null;
    m = t.match(/^(\d+):(\d{1,2}):(\d{1,2})$/);
    if (m) return Number(m[2]) < 60 && Number(m[3]) < 60 ? Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]) : null;
    if (/^\d+(\.\d+)?$/.test(t)) return Number(t);
    return null;
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function iconButton(icon, tip) {
    const b = el('button', 'replay-btn');
    b.type = 'button';
    b.title = tip;
    b.appendChild(el('i', 'material-icons', icon));
    return b;
  }
  function message(text) { ui.msg.textContent = text || ''; ui.msg.style.display = text ? 'block' : 'none'; }
  function view() { return window.meeting || null; }

  function build() {
    panel = el('div', 'replay-panel');
    panel.id = ID;
    panel.style.display = 'none';
    panel.setAttribute('aria-label', '回聽');
    const head = el('div', 'replay-head');
    head.title = '拖曳標題列移動';
    ui.title = el('span', 'replay-title', '回聽');
    const grow = el('div'); grow.style.flex = '1 1 auto';
    ui.close = iconButton('close', '關閉');
    head.append(ui.title, grow, ui.close);

    const body = el('div', 'replay-body');
    ui.now = el('div', 'replay-now');
    ui.text = el('div', 'replay-text');
    const controls = el('div', 'replay-controls');
    ui.toggle = iconButton('pause', '暫停／繼續');
    ui.bar = el('input', 'replay-bar');
    ui.bar.type = 'range'; ui.bar.min = '0'; ui.bar.max = '0'; ui.bar.step = '0.05'; ui.bar.value = '0';
    ui.bar.disabled = true;
    ui.time = el('span', 'replay-time', '0:00 / 0:00');
    controls.append(ui.toggle, ui.bar, ui.time);
    const startRow = el('div', 'replay-start');
    startRow.append(el('span', null, '起點'));
    ui.start = el('input', 'replay-start-input');
    ui.start.type = 'text'; ui.start.size = 8; ui.start.title = '例如 01:20、1:02:03，或秒數；按 Enter 或「從此起點播放」';
    ui.go = el('button', 'replay-go', '從此起點播放');
    ui.go.type = 'button';
    startRow.append(ui.start, ui.go);
    ui.msg = el('div', 'replay-msg');
    const note = el('div', 'replay-note', '請使用耳機收聽，以免聲音被麥克風錄進會議。');
    body.append(ui.now, ui.text, controls, startRow, ui.msg, note);
    panel.append(head, body);
    document.body.appendChild(panel);

    audio = new Audio();
    audio.preload = 'auto';
    audio.addEventListener('loadedmetadata', function () {
      ui.bar.max = String(audio.duration || 0);
      ui.bar.disabled = false;
      refresh();
    });
    ['timeupdate', 'play', 'pause', 'ended'].forEach(function (name) { audio.addEventListener(name, refresh); });
    ui.close.addEventListener('click', close);
    ui.toggle.addEventListener('click', function () {
      if (!state.loaded) return;
      if (audio.paused) play(); else audio.pause();
    });
    ui.bar.addEventListener('pointerdown', function () { state.scrubbing = true; });
    ['pointerup', 'pointercancel', 'blur'].forEach(function (name) {
      ui.bar.addEventListener(name, function () { state.scrubbing = false; });
    });
    ui.bar.addEventListener('input', function () {
      if (state.loaded) { audio.currentTime = Number(ui.bar.value); refresh(); }
    });
    function submit() {
      const typed = parseStart(ui.start.value);
      if (typed === null) { message('起點格式不對，請輸入 mm:ss（例如 01:20）或秒數。'); return; }
      // an unedited field still holds the clock text of the exact start: keep the exact (fractional) second then
      load(ui.start.value.trim() === clock(state.start) ? state.start : typed);
    }
    ui.go.addEventListener('click', submit);
    ui.start.addEventListener('keydown', function (event) { if (event.key === 'Enter') submit(); });

    head.addEventListener('pointerdown', function (event) {       // the header moves the window
      if (event.target.closest && event.target.closest('button')) return;
      const r = panel.getBoundingClientRect();
      drag = {dx: event.clientX - r.left, dy: event.clientY - r.top};
      if (event.preventDefault) event.preventDefault();
    });
    document.addEventListener('pointermove', function (event) {
      if (!drag) return;
      const x = Math.min(Math.max(event.clientX - drag.dx, 0), window.innerWidth - 80);
      const y = Math.min(Math.max(event.clientY - drag.dy, 0), window.innerHeight - 40);
      panel.style.left = x + 'px'; panel.style.top = y + 'px'; panel.style.right = 'auto'; panel.style.bottom = 'auto';
    });
    ['pointerup', 'pointercancel'].forEach(function (name) { document.addEventListener(name, function () { drag = null; }); });
  }

  function play() {
    const p = audio.play();
    if (p && p.catch) p.catch(function () { refresh(); });          // the browser refused to start by itself: the button says play
  }

  function refresh() {
    if (!panel || panel.style.display === 'none') return;
    const dur = (audio && isFinite(audio.duration)) ? audio.duration : 0, cur = audio ? audio.currentTime : 0;
    if (!state.scrubbing) ui.bar.value = String(cur);
    ui.time.textContent = span(cur) + ' / ' + span(dur);
    ui.toggle.firstChild.textContent = (state.loaded && !audio.paused && !audio.ended) ? 'pause' : 'play_arrow';
    const meeting = view();
    const row = meeting && state.loaded ? meeting.rowAt(state.start + cur + 0.05) : null;
    if (meeting) meeting.setPlaying(row ? row.i : null);
    if (row) {
      ui.now.textContent = row.label + '　' + clock(row.t);
      ui.text.textContent = row.text;
    }
  }

  function release() {
    if (state.blobUrl) { URL.revokeObjectURL(state.blobUrl); state.blobUrl = null; }
  }

  function stop() {
    state.seq++;                                                    // a clip still arriving is dropped
    audio.pause();
    audio.removeAttribute('src');                                   // the previous clip is gone: a failed load must not show its position
    audio.load();
    release();
    state.loaded = false;
    ui.bar.disabled = true; ui.bar.value = '0'; ui.bar.max = '0';
  }

  async function load(start) {
    stop();                                                         // (this also makes every earlier load out of date)
    const seq = state.seq;
    state.start = start;
    ui.start.value = clock(start);
    ui.title.textContent = '回聽 · 從 ' + clock(start) + ' 起';
    ui.now.textContent = '';
    ui.text.textContent = '';
    message('載入中…');
    refresh();
    try {
      const response = await fetch(URL_PATH + '?start=' + encodeURIComponent(start.toFixed(2)) +
                                   '&token=' + encodeURIComponent(state.token), {cache: 'no-store'});
      if (seq !== state.seq) return;                                // another click came in meanwhile
      if (!response.ok) {
        const reason = await response.text();
        if (seq !== state.seq) return;                              // the error of an earlier click arrived after a newer one took over
        message(reason || '無法取得這段聲音。');
        return;
      }
      const blob = await response.blob();
      if (seq !== state.seq) return;
      release();
      state.blobUrl = URL.createObjectURL(blob);
      audio.src = state.blobUrl;
      state.loaded = true;
      message('');
      play();
    } catch (e) {
      if (seq === state.seq) message('無法取得這段聲音：' + (e && e.message ? e.message : e));
    }
  }

  // info: {start, i, token, blocked}: `blocked` is the reason the audio of this meeting cannot be replayed ('' when it can)
  function open(info) {
    if (!panel) build();
    panel.style.display = 'block';
    state.token = String(info.token || '');
    if (info.blocked) {                                             // nothing to fetch: say why
      stop();
      ui.title.textContent = '回聽';
      ui.now.textContent = ''; ui.text.textContent = '';
      ui.start.value = clock(info.start);
      message(info.blocked);
      return;
    }
    load(Math.max(0, Number(info.start) || 0));
  }

  function close() {
    if (!panel) return;
    stop();
    panel.style.display = 'none';
    const meeting = view();
    if (meeting) meeting.setPlaying(null);
  }

  // the meeting view tells which meeting it shows now: the player of another one (or of none) is closed
  function sync(replay) {
    if (panel && panel.style.display !== 'none' && (!replay || replay.token !== state.token)) close();
  }

  return {open: open, close: close, sync: sync, parseStart: parseStart, clock: clock};
})();
