/* Keep capturing in this tab during a short WebSocket outage, then replay unacknowledged PCM. */
const workletSource = `
class VoicePCM extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.length = 0;
    this.finishing = false;
    this.port.onmessage = event => {
      if (event.data === 'finish') {
        this.finishing = true;
        this.flush();
        this.port.postMessage('finished');
      }
    };
  }
  flush() {
    if (this.length) {
      const packet = this.buffer.slice(0, this.length);
      this.port.postMessage(packet, [packet.buffer]);
      this.length = 0;
    }
  }
  process(inputs) {
    if (this.finishing) return true;
    const channels = inputs[0];
    if (channels && channels.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i] / channels.length;
        this.buffer[this.length++] = sample;
        if (this.length === this.buffer.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('voice-pcm', VoicePCM);
`;

const RECONNECT_MS = 30000;
const RETRY_DELAY_MS = 700;
const HEARTBEAT_MS = 5000;
const RESPONSE_TIMEOUT_MS = 15000;
const STARTUP_TIMEOUT_MS = 375000;
const MAX_PENDING_BYTES = 32 * 1024 * 1024;
const MAX_SOCKET_BYTES = 512 * 1024;

class RemoteCapture {
  constructor() {
    this.pending = false;
    this.stopping = false;
    this.stopRequested = false;
    this.stopSent = false;
    this.stream = null;
    this.socket = null;
    this.context = null;
    this.node = null;
    this.captureError = '';
    this.writingPrepared = false;
    this.everReady = false;
    this.serverReady = false;
    this.formatSent = false;
    this.kind = null;
    this.token = null;
    this.sessionId = null;
    this.writingContext = null;
    this.frames = [];
    this.pendingBytes = 0;
    this.nextSequence = 0;
    this.lastAck = -1;
    this.lastSent = -1;
    this.retryTimer = null;
    this.deadlineTimer = null;
    this.startupTimer = null;
    this.heartbeatTimer = null;
    this.lastServerReply = 0;
    this.reconnectStarted = 0;
    this.flushResolve = null;
    this.audioStarting = null;
  }

  async start(kind, token) {
    if (this.pending || this.stream || this.socket) return;
    this.pending = true;
    this.stopping = false;
    this.captureError = '';
    try {
      const writingContext = window.writing?.mode ? window.writing.begin(window.writing.contextChars) : null;
      this.writingPrepared = !!writingContext;
      if (!navigator.mediaDevices) throw new Error('此瀏覽器需要 HTTPS 才能擷取遠端音訊。');
      // getDisplayMedia must be called directly from the user's click.
      const stream = kind === 'remote_system'
        ? await navigator.mediaDevices.getDisplayMedia({video: true, audio: true, systemAudio: 'include'})
        : await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false}});
      this.stream = stream;
      if (!stream.getAudioTracks().length) {
        throw new Error('瀏覽器沒有提供音軌；請在分享視窗勾選「分享音訊」，或改用支援的瀏覽器。');
      }
      for (const track of stream.getTracks()) {
        track.addEventListener('ended', () => {
          if (!this.stopping) this.fail('遠端音訊分享已中斷。');
        });
      }
      this.kind = kind;
      this.token = token;
      this.sessionId = crypto.randomUUID();
      this.writingContext = writingContext;
      this.startupTimer = setTimeout(() => {
        if (this.sessionId && !this.everReady) {
          this.fail('ASR 準備超過 375 秒；已停止等待並解除文章鎖定。');
        }
      }, STARTUP_TIMEOUT_MS);
      this.openSocket(false, writingContext);
    } catch (error) {
      this.cleanup();
      alert(`無法取得遠端音訊：${error.message}`);
    } finally {
      this.pending = false;
    }
  }

  openSocket(resume, writingContext = null) {
    if (this.socket || !this.sessionId) return;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/remote-audio/${this.token}`);
    this.socket = socket;
    this.lastServerReply = Date.now();
    this.startHeartbeat(socket);
    socket.onopen = () => {
      if (this.socket !== socket) return;
      const resumeActiveRecording = resume && this.everReady;
      socket.send(JSON.stringify({type: resumeActiveRecording ? 'resume' : 'start', kind: this.kind,
        session_id: this.sessionId,
        ...(resumeActiveRecording ? {} : {writing_context: this.writingContext ?? writingContext})}));
    };
    socket.onmessage = async event => {
      if (this.socket !== socket) return;
      let message;
      try { message = JSON.parse(event.data); }
      catch (_) { this.fail('遠端音訊回應格式錯誤。'); return; }
      this.lastServerReply = Date.now();
      if (message.type === 'ready' || message.type === 'resumed') {
        const next = message.next_sequence;
        if (!Number.isInteger(next) || next < 0 || next > this.nextSequence || next < this.lastAck + 1) {
          this.fail('遠端音訊重連序號不一致。');
          return;
        }
        this.serverReady = this.everReady = true;
        this.formatSent = false;
        this.reconnectStarted = 0;
        clearTimeout(this.retryTimer);
        clearTimeout(this.deadlineTimer);
        clearTimeout(this.startupTimer);
        this.retryTimer = null;
        this.deadlineTimer = null;
        this.startupTimer = null;
        this.ackThrough(next - 1);
        this.lastSent = next - 1;
        this.stopSent = false;
        this.startHeartbeat(socket);
        if (!this.node && !this.stopRequested) {
          try {
            if (!this.audioStarting) this.audioStarting = this.beginAudio();
            await this.audioStarting;
          }
          catch (error) { this.fail(`無法開始遠端音訊：${error.message}`); return; }
          finally { this.audioStarting = null; }
        }
        if (this.socket !== socket || !this.serverReady) return;
        if (this.node) this.sendFormat();
        this.sendPending();
        this.maybeFinish();
      } else if (message.type === 'ack') {
        if (!Number.isInteger(message.sequence) || message.sequence >= this.nextSequence) {
          this.fail('遠端音訊確認序號不合法。');
          return;
        }
        this.ackThrough(message.sequence);
        this.sendPending();
        this.maybeFinish();
      } else if (message.type === 'stop') {
        await this.stop();
      } else if (message.type === 'pong') {
        // The server is responsive even if there are no audio ACKs right now.
      } else if (message.type === 'preparing') {
        this.reconnectStarted = 0;
        clearTimeout(this.deadlineTimer);
        this.deadlineTimer = null;
      } else if (message.type === 'finished' || message.type === 'cancel') {
        this.cleanup();
      } else if (message.type === 'retry') {
        socket.close();
      } else if (message.type === 'error') {
        this.fail(message.message || '遠端音訊服務發生錯誤。');
      }
    };
    socket.onclose = () => this.abandonSocket(socket);
    socket.onerror = () => this.abandonSocket(socket);
  }

  startHeartbeat(socket) {
    clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = setInterval(() => {
      if (this.socket !== socket) return;
      if (Date.now() - this.lastServerReply >= RESPONSE_TIMEOUT_MS) {
        this.abandonSocket(socket);
        return;
      }
      if (!this.serverReady) return;
      if (socket.readyState === WebSocket.OPEN) {
        try { socket.send(JSON.stringify({type: 'ping'})); }
        catch (_) { this.abandonSocket(socket); }
      } else {
        this.abandonSocket(socket);
      }
    }, HEARTBEAT_MS);
  }

  abandonSocket(socket) {
    if (this.socket !== socket) return;
    this.socket = null;
    this.serverReady = false;
    this.formatSent = false;
    this.stopSent = false;
    clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
    try { socket.close(); } catch (_) { /* Replace the stalled socket anyway. */ }
    this.scheduleReconnect();
  }

  scheduleReconnect() {
    if (!this.sessionId) return;
    if (!this.reconnectStarted) {
      this.reconnectStarted = Date.now();
      this.deadlineTimer = setTimeout(() => {
        this.fail('遠端音訊超過 30 秒無法重連；已收到的聲音會標示為未完成。');
      }, RECONNECT_MS);
    }
    if (Date.now() - this.reconnectStarted >= RECONNECT_MS) {
      this.fail('遠端音訊超過 30 秒無法重連；已收到的聲音會標示為未完成。');
      return;
    }
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.openSocket(true);
    }, RETRY_DELAY_MS);
  }

  queueFrame(samples) {
    if (!this.sessionId) return;
    if (this.nextSequence >= 0xffffffff) {
      this.fail('遠端音訊錄音時間過長，無法繼續傳送。');
      return;
    }
    const packet = new ArrayBuffer(4 + samples.byteLength);
    new DataView(packet).setUint32(0, this.nextSequence, true);
    new Uint8Array(packet, 4).set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
    this.frames.push({sequence: this.nextSequence++, packet});
    this.pendingBytes += packet.byteLength;
    if (this.pendingBytes > MAX_PENDING_BYTES) {
      this.fail('遠端音訊暫存區已滿；已收到的聲音會標示為未完成。');
      return;
    }
    this.sendPending();
  }

  ackThrough(sequence) {
    if (sequence < this.lastAck) return;
    this.lastAck = sequence;
    while (this.frames.length && this.frames[0].sequence <= sequence) {
      this.pendingBytes -= this.frames.shift().packet.byteLength;
    }
  }

  sendPending() {
    const socket = this.socket;
    if (!this.serverReady || !this.formatSent || socket?.readyState !== WebSocket.OPEN) return;
    for (const frame of this.frames) {
      if (frame.sequence <= this.lastSent) continue;
      if (socket.bufferedAmount >= MAX_SOCKET_BYTES) break;
      try { socket.send(frame.packet); }
      catch (_) { socket.close(); return; }
      this.lastSent = frame.sequence;
    }
  }

  async beginAudio() {
    const context = new AudioContext();
    this.context = context;
    const url = URL.createObjectURL(new Blob([workletSource], {type: 'text/javascript'}));
    try { await context.audioWorklet.addModule(url); }
    finally { URL.revokeObjectURL(url); }
    if (this.stopping) return;
    const node = new AudioWorkletNode(context, 'voice-pcm');
    this.node = node;
    node.port.onmessage = event => {
      if (event.data instanceof Float32Array) this.queueFrame(event.data);
      else if (event.data === 'finished' && this.flushResolve) {
        this.flushResolve();
        this.flushResolve = null;
      }
    };
    context.createMediaStreamSource(this.stream).connect(node);
    node.connect(context.destination); // The worklet outputs silence, so there is no echo.
    await context.resume();
  }

  sendFormat() {
    if (this.serverReady && this.socket?.readyState === WebSocket.OPEN && this.context) {
      try {
        this.socket.send(JSON.stringify({type: 'format', sample_rate: this.context.sampleRate}));
        this.formatSent = true;
      }
      catch (_) { this.socket.close(); }
    }
  }

  async stop() {
    if (!this.sessionId) return;
    if (this.stopRequested) return;
    this.stopRequested = this.stopping = true;
    let finished = true;
    if (this.node) {
      finished = await new Promise(resolve => {
        const timer = setTimeout(() => {
          this.flushResolve = null;
          resolve(false);
        }, 5000);
        this.flushResolve = () => {
          clearTimeout(timer);
          resolve(true);
        };
        try { this.node.port.postMessage('finish'); }
        catch (_) {
          clearTimeout(timer);
          this.flushResolve = null;
          resolve(false);
        }
      });
    }
    if (!finished) this.captureError = '瀏覽器無法確認音訊尾段已送出。';
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    this.sendPending();
    this.maybeFinish();
  }

  maybeFinish() {
    if (!this.stopRequested || this.stopSent || this.frames.length ||
        !this.serverReady || this.socket?.readyState !== WebSocket.OPEN) return;
    try {
      this.socket.send(JSON.stringify(this.captureError
        ? {type: 'error', message: this.captureError}
        : {type: 'stop'}));
      this.stopSent = true;
    } catch (_) {
      this.socket.close();
    }
  }

  fail(message) {
    if (!this.sessionId) return;
    if (this.socket?.readyState === WebSocket.OPEN) {
      try { this.socket.send(JSON.stringify({type: 'error', message})); }
      catch (_) { /* The server's reconnect deadline will close the partial recording. */ }
    }
    this.cleanup();
    alert(message);
  }

  cleanup() {
    if (this.writingPrepared && !this.everReady) window.writing?.rollback();
    clearTimeout(this.retryTimer);
    clearTimeout(this.deadlineTimer);
    clearTimeout(this.startupTimer);
    clearInterval(this.heartbeatTimer);
    this.retryTimer = null;
    this.deadlineTimer = null;
    this.startupTimer = null;
    this.heartbeatTimer = null;
    this.writingPrepared = false;
    this.everReady = this.serverReady = false;
    this.formatSent = false;
    this.stopping = true;
    if (this.node) this.node.disconnect();
    if (this.context) this.context.close();
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    const socket = this.socket;
    this.socket = null;
    if (socket) socket.close();
    this.node = this.context = this.stream = null;
    this.flushResolve = null;
    this.audioStarting = null;
    this.captureError = '';
    this.kind = this.token = this.sessionId = null;
    this.writingContext = null;
    this.frames = [];
    this.pendingBytes = 0;
    this.nextSequence = 0;
    this.lastAck = this.lastSent = -1;
    this.reconnectStarted = 0;
    this.lastServerReply = 0;
    this.stopRequested = this.stopSent = false;
    this.pending = false;
  }
}

window.remoteAudio = new RemoteCapture();
document.documentElement.dataset.remoteAudioReady = 'true';
