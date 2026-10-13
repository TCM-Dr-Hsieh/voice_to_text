/* The two-column meeting view: the transcript on the left, its translation on the right, one table row per segment so that a
   segment and its translation always sit side by side. The server sends only the rows that changed; everything shown is built
   with textContent (a transcript may contain anything). */
class MeetingView {
  constructor() {
    this.rows = new Map();          // segment number -> {data, el, left, right}
    this.order = [];                // the numbers present, ascending
    this.labels = {marking: true, names: {}, unknown: '不明', voice: '語音', background: '背景', palette: 10, tnames: {}, tunknown: 'Unknown',
                   tvoice: 'Voice'};
    this.script = 'original';       // display-only Traditional/Simplified conversion
    this.host = null;
    this.scroller = null;
  }

  ensure() {
    if (this.scroller?.isConnected) return true;
    const host = document.getElementById('meeting-host');
    if (!host) return false;
    host.replaceChildren();
    this.scroller = document.createElement('div');
    this.scroller.className = 'm-scroll';
    this.scroller.setAttribute('aria-label', '會議即時語音與翻譯');
    host.append(this.scroller);
    this.host = host;
    this.rows.clear();
    this.order = [];
    return true;
  }

  convert(text) {
    const converter = globalThis.window?.scriptConverter;
    return converter && this.script !== 'original' ? converter.convert(text, this.script) : text;
  }

  clear() {
    this.rows.clear();
    this.order = [];
    this.scroller?.replaceChildren();
  }

  setScript(mode) {
    this.script = mode;
    for (const entry of this.rows.values()) this.render(entry);
  }

  colour(g) {
    // a speaker's colour is its number's: the first colour for the first speaker, and again from the first when they are used up
    return g === null || g === undefined ? '' : `m-c${g % (this.labels.palette || 10)}`;
  }

  tone(g, background) {
    // how a speaker marking looks on a line: nothing at all while the marking is off (the rows kept on the page are not forgotten
    // when it stops, so they must not keep showing it)
    if (this.labels.marking === false) return '';
    return background ? 'm-bg' : (g === null || g === undefined ? 'm-unknown' : this.colour(g));
  }

  name(map, g, unknownLabel) {
    if (g === null || g === undefined) return unknownLabel;
    return map[String(g)] ?? this.labels.names[String(g)] ?? `Speaker-${String(g + 1).padStart(3, '0')}`;
  }

  line(prefix, who, mark, text, cls, act, tip) {
    const line = document.createElement('div');
    line.className = 'm-line' + (cls ? ' ' + cls : '');
    const number = document.createElement('span');
    number.className = 'm-n';
    number.textContent = prefix;
    const body = document.createElement('span');
    body.className = 'm-text';
    body.textContent = text;
    if (!who) {                                    // no speaker marking: the line is the number and the text
      line.append(number, body);
      return line;
    }
    const speaker = document.createElement('span');
    speaker.className = 'm-who';
    speaker.textContent = who + mark;
    if (act) {                                     // the label can be clicked (mark / unmark background talk)
      speaker.classList.add('m-act');
      speaker.title = tip || '';
      speaker.tabIndex = 0;
      speaker.setAttribute('role', 'button');
      speaker.addEventListener('click', act);
      speaker.addEventListener('keydown', event => { if (event.key === 'Enter') act(event); });
    }
    line.append(number, speaker, document.createTextNode(': '), body);
    return line;
  }

  mark(i, r, t, to) {
    // tell the server: sentence run r of segment i (as shown: t) is / is not background talk
    if (typeof emitEvent === 'function') emitEvent('meeting_background', {i, r, t, to});
  }

  render(entry) {
    const {data, left, right} = entry;
    left.replaceChildren();
    right.replaceChildren();
    left.classList.toggle('m-rev', !!data.rev);
    const marking = this.labels.marking !== false;
    data.runs.forEach((run, k) => {
      const background = run.s === 'background';
      const mark = run.s === 'text' || run.s === 'text2' ? '*' : '';
      const who = !marking ? '' : (background ? this.labels.background : this.name(this.labels.names, run.g, this.labels.unknown));
      // background talk can be unmarked with a click, an unknown speaker can be marked as background talk
      const to = background ? false : (run.g === null ? true : null);
      const act = marking && to !== null ? () => this.mark(data.i, k, run.t, to) : null;
      const tip = to === false ? '點一下：這不是背景，要翻譯' : '點一下：標成背景（灰色、不翻譯）';
      left.append(this.line(k === 0 ? `${this.labels.voice}#${data.i}` : '', who, mark, this.convert(run.t), this.tone(run.g, background), act, tip));
    });
    const tr = data.tr;
    right.classList.toggle('m-stale', !!tr && tr.st === 'stale');
    right.classList.toggle('m-rev', !!tr && tr.st === 'draft' && !!data.rev);       // a draft of the newest rows: it may still change
    if (!tr) return;
    if (tr.st === 'draft') {                       // one text for the segment, no speakers (they are not settled yet)
      right.append(this.line(`${this.labels.tvoice}#${data.i}`, '', '', tr.runs.map(run => run.t).join(' '), ''));
      return;
    }
    if (tr.st === 'error') {
      right.append(this.line(`${this.labels.tvoice}#${data.i}`, '', '', `（翻譯失敗）${tr.err || ''}`, 'm-error'));
      return;
    }
    tr.runs.filter(run => run.t).forEach((run, k) => {
      const who = marking ? this.name(this.labels.tnames, run.g, this.labels.tunknown) : '';
      right.append(this.line(k === 0 ? `${this.labels.tvoice}#${data.i}` : '', who, '', run.t, this.tone(run.g, false)));
    });
  }

  insert(entry, i) {
    // keep `order` sorted and the table in the same order (rows usually arrive in order; an old row may be re-sent)
    let lo = 0, hi = this.order.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.order[mid] < i) lo = mid + 1; else hi = mid;
    }
    const next = this.order[lo];
    this.order.splice(lo, 0, i);
    this.scroller.insertBefore(entry.el, next === undefined ? null : this.rows.get(next).el);
  }

  update(payload) {
    if (!this.ensure()) return false;
    if (payload.reset) this.clear();
    const bottom = this.scroller.scrollHeight - this.scroller.scrollTop - this.scroller.clientHeight < 60;
    let relabel = false;
    if (payload.labels && JSON.stringify(payload.labels) !== JSON.stringify(this.labels)) {
      this.labels = payload.labels;
      relabel = true;
    }
    for (const data of payload.rows || []) {
      let entry = this.rows.get(data.i);
      if (!entry) {
        const el = document.createElement('div');
        el.className = 'm-row';
        el.dataset.i = String(data.i);
        const left = document.createElement('div');
        left.className = 'm-left';
        const right = document.createElement('div');
        right.className = 'm-right';
        el.append(left, right);
        entry = {data, el, left, right};
        this.rows.set(data.i, entry);
        this.insert(entry, data.i);
      }
      entry.data = data;
      this.render(entry);
    }
    if (relabel) for (const entry of this.rows.values()) this.render(entry);
    if (bottom) this.scroller.scrollTop = this.scroller.scrollHeight;
    return true;
  }
}

window.meeting = new MeetingView();
