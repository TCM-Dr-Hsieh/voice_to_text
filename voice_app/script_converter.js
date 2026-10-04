/* Convert only in this browser; the server keeps the original ASR and audit JSON. */
class ScriptConverter {
  constructor() {
    this.mode = 'original';
    this.toSimplified = globalThis.OpenCC.Converter({from: 'tw', to: 'cn'});
    this.toTraditional = globalThis.OpenCC.Converter({from: 'cn', to: 'tw'});
  }

  convert(value, mode = this.mode) {
    const text = String(value ?? '');
    if (mode === 'simplified') return this.toSimplified(text);
    if (mode === 'traditional') return this.toTraditional(text);
    return text;
  }

  setMode(mode) {
    if (!['original', 'simplified', 'traditional'].includes(mode)) throw new Error('不支援的繁簡轉換方向。');
    this.mode = mode;
  }

}

window.scriptConverter = new ScriptConverter();
