/* A note lives only in this browser page. The server receives short read-only context. */
class VoiceWriting {
  constructor() {
    this.mode = false;
    this.pending = null;
    this.fileHandle = null;
    this.fileName = '語音筆記.txt';
    this.savedText = '';
    this.articleMode = 'original';
    this.conversionSnapshot = null;
    window.addEventListener('beforeunload', event => {
      if (this.dirty() || this.pending) {
        event.preventDefault();
        event.returnValue = '';
      }
    });
    document.addEventListener('click', event => {
      const button = event.target.closest?.('#writing-host button[data-writing-action]');
      if (button) {
        this[button.dataset.writingAction]();
      }
    }, true);
    document.addEventListener('input', event => {
      if (event.target.matches?.('#writing-host .writing-editor')) {
        this.editor = event.target;
        this.refreshStatus();
      }
    });
  }

  ensure() {
    if (this.editor?.isConnected) return true;
    const host = document.getElementById('writing-host');
    if (!host) return false;
    const existing = host.querySelector('.writing-editor');
    if (existing) {
      this.editor = existing;
      this.preview = host.querySelector('.writing-preview');
      this.badge = host.querySelector('.writing-file');
      return true;
    }
    host.replaceChildren();
    const bar = document.createElement('div');
    bar.className = 'writing-toolbar';
    for (const [label, action] of [
      ['全新筆記', 'newNote'], ['開啟 TXT', 'open'],
      ['儲存 TXT', 'save'], ['另存新檔 TXT', 'saveAs'],
      ['複製全文', 'copy'],
    ]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.dataset.writingAction = action;
      bar.append(button);
    }
    this.badge = document.createElement('span');
    this.badge.className = 'writing-file';
    bar.append(this.badge);
    this.editor = document.createElement('textarea');
    this.editor.className = 'writing-editor';
    this.editor.placeholder = '在此輸入或貼上文章，將游標放到要插入語音的位置。';
    this.editor.setAttribute('aria-label', '語音書寫文章');
    this.preview = document.createElement('div');
    this.preview.className = 'writing-preview';
    this.preview.setAttribute('aria-label', '語音插入預覽');
    this.preview.hidden = true;
    host.append(bar, this.editor, this.preview);
    this.refreshStatus();
    return true;
  }

  dirty() { return !!this.editor && this.editor.value !== this.savedText; }
  refreshStatus() {
    if (this.badge) {
      const originalFile = this.articleMode === 'original' && window.scriptConverter.mode !== 'original'
        ? ' · 文章仍為原文字形，按轉換按鈕才套用' : '';
      this.badge.textContent = `${this.fileName}${this.dirty() ? ' · 尚未儲存' : ' · 已儲存'}${originalFile}`;
    }
  }
  setMode(value, contextChars = 2000) {
    this.mode = !!value;
    this.contextChars = contextChars;
    this.ensure();
    if (this.mode && !this.pending) this.editor?.focus();
  }

  setScript(mode) {
    if (!this.ensure() || this.pending) return {status: 'unavailable'};
    if (mode === this.articleMode) {
      if (this.conversionSnapshot && this.editor.value !== this.conversionSnapshot.convertedText)
        return {status: 'edited'};
      if (this.conversionSnapshot) {
        const {text, start, end} = this.conversionSnapshot;
        this.editor.value = text;
        this.editor.setSelectionRange(start, end);
      }
      this.conversionSnapshot = null;
      this.articleMode = 'original';
      window.scriptConverter.setMode('original');
      this.refreshStatus();
      return {status: 'restored', mode: 'original'};
    }
    // Reconvert the exact pre-conversion text when possible; OpenCC round trips can be lossy.
    const reusable = this.conversionSnapshot && this.editor.value === this.conversionSnapshot.convertedText;
    const text = reusable ? this.conversionSnapshot.text : this.editor.value;
    const start = reusable ? this.conversionSnapshot.start : this.editor.selectionStart;
    const end = reusable ? this.conversionSnapshot.end : this.editor.selectionEnd;
    const converted = window.scriptConverter.convert(text, mode);
    this.conversionSnapshot = {text, start, end, convertedText: converted};
    this.editor.value = converted;
    this.editor.setSelectionRange(
      Math.min(window.scriptConverter.convert(text.slice(0, start), mode).length, converted.length),
      Math.min(window.scriptConverter.convert(text.slice(0, end), mode).length, converted.length));
    this.articleMode = mode;
    window.scriptConverter.setMode(mode);
    this.refreshStatus();
    return {status: 'converted', mode};
  }

  begin(contextChars) {
    if (!this.mode || !this.ensure() || this.pending) throw new Error('書寫編輯區尚未準備好。');
    const start = this.editor.selectionStart;
    const end = this.editor.selectionEnd;
    const text = this.editor.value;
    this.pending = {text, start, end, before: text.slice(0, start), after: text.slice(end)};
    this.editor.readOnly = true;
    this.editor.hidden = true;
    this.preview.hidden = false;
    this.render('', '');
    const limit = Math.max(100, Math.min(20000, Number(contextChars) || 2000));
    return {before: this.pending.before.slice(-limit), after: this.pending.after.slice(0, limit)};
  }

  render(locked, revisable) {
    if (!this.pending || !this.ensure()) return;
    const {before, after} = this.pending;
    this.preview.replaceChildren();
    const piece = (value, className) => {
      const span = document.createElement('span');
      span.className = className;
      span.textContent = value;
      this.preview.append(span);
    };
    piece(before, '');
    piece(window.scriptConverter.convert(locked, this.articleMode), '');
    piece(window.scriptConverter.convert(revisable, this.articleMode), 'writing-revisable');
    piece(after, '');
  }

  finishView() {
    this.preview.hidden = true;
    this.editor.hidden = false;
    this.editor.readOnly = false;
    this.refreshStatus();
    this.editor.focus();
  }
  commit(corrected) {
    if (!this.pending) return;
    if (!corrected) {
      this.rollback();
      return;
    }
    const {before, after} = this.pending;
    const inserted = window.scriptConverter.convert(corrected, this.articleMode);
    this.editor.value = before + inserted + after;
    const caret = (before + inserted).length;
    this.pending = null;
    this.finishView();
    this.editor.setSelectionRange(caret, caret);
  }
  rollback() {
    if (!this.pending) return;
    const {text, start, end} = this.pending;
    this.editor.value = text;
    this.pending = null;
    this.finishView();
    this.editor.setSelectionRange(start, end);
  }

  async unsavedChoice() {
    if (!this.dirty()) return 'discard';
    const dialog = document.createElement('dialog');
    dialog.className = 'writing-confirm';
    const title = document.createElement('p');
    title.textContent = '目前文章有未儲存的修改。';
    dialog.append(title);
    for (const [label, value] of [['儲存', 'save'], ['不儲存', 'discard'], ['取消', 'cancel']]) {
      const button = document.createElement('button');
      button.textContent = label;
      button.type = 'button';
      button.addEventListener('click', () => dialog.close(value));
      dialog.append(button);
    }
    document.body.append(dialog);
    const choice = await new Promise(resolve => {
      dialog.addEventListener('close', () => resolve(dialog.returnValue || 'cancel'), {once: true});
      dialog.showModal();
    });
    dialog.remove();
    if (choice === 'save' && !(await this.save())) return 'cancel';
    return choice;
  }

  async newNote() {
    if (!this.ensure() || this.pending) return;
    if (await this.unsavedChoice() === 'cancel') return;
    this.editor.value = '';
    this.savedText = '';
    this.fileHandle = null;
    this.fileName = '語音筆記.txt';
    this.articleMode = window.scriptConverter.mode;
    this.conversionSnapshot = null;
    this.refreshStatus();
    this.editor.focus();
  }

  async open() {
    if (!this.ensure() || this.pending) return;
    if (await this.unsavedChoice() === 'cancel') return;
    try {
      let file, handle = null;
      if (window.showOpenFilePicker) {
        [handle] = await window.showOpenFilePicker({types: [{description: 'TXT', accept: {'text/plain': ['.txt']}}]});
        file = await handle.getFile();
      } else {
        file = await new Promise(resolve => {
          const input = document.createElement('input');
          input.type = 'file'; input.accept = '.txt,text/plain';
          input.style.display = 'none';
          document.body.append(input);
          const finish = value => { resolve(value); input.remove(); };
          input.addEventListener('change', () => finish(input.files[0]), {once: true});
          input.addEventListener('cancel', () => finish(null), {once: true});
          input.click();
        });
      }
      if (!file) return;
      const bytes = await file.arrayBuffer();
      const text = new TextDecoder('utf-8', {fatal: true}).decode(bytes).replace(/^\uFEFF/, '');
      this.editor.value = text;
      this.savedText = text;
      this.fileName = file.name;
      this.fileHandle = handle;
      this.articleMode = 'original';
      this.conversionSnapshot = null;
      this.refreshStatus();
      this.editor.focus();
    } catch (error) {
      if (error.name !== 'AbortError') alert(`開啟 TXT 失敗：${error.message}`);
    }
  }

  async save() { return this.write(false); }
  async saveAs() { return this.write(true); }
  async write(newFile) {
    if (!this.ensure() || this.pending) return false;
    try {
      let handle = newFile ? null : this.fileHandle;
      if (!handle && window.showSaveFilePicker) {
        handle = await window.showSaveFilePicker({suggestedName: this.fileName,
          types: [{description: 'TXT', accept: {'text/plain': ['.txt']}}]});
      }
      const bytes = new Blob(['\uFEFF', this.editor.value], {type: 'text/plain;charset=utf-8'});
      if (handle) {
        const stream = await handle.createWritable();
        await stream.write(bytes);
        await stream.close();
        this.fileHandle = handle;
        this.fileName = handle.name;
      } else {
        if (newFile) {
          const name = window.prompt('另存新檔的檔名', this.fileName);
          if (!name) return false;
          this.fileName = name.toLowerCase().endsWith('.txt') ? name : `${name}.txt`;
        }
        const url = URL.createObjectURL(bytes);
        const link = document.createElement('a');
        link.href = url; link.download = this.fileName;
        document.body.append(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        alert('此瀏覽器不支援直接覆寫 TXT；已改為下載檔案。');
      }
      this.savedText = this.editor.value;
      this.refreshStatus();
      return true;
    } catch (error) {
      if (error.name !== 'AbortError') alert(`儲存 TXT 失敗：${error.message}`);
      return false;
    }
  }

  async copy() {
    if (!this.ensure() || this.pending) return;
    try { await navigator.clipboard.writeText(this.editor.value); }
    catch (_) { alert('剪貼簿不可用，請選取文章後手動複製。'); }
  }
}

window.writing = new VoiceWriting();
