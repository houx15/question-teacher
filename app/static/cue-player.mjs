class BrowserPausableTimeline {
  constructor({
    now = () => performance.now(),
    setTimeoutImpl = globalThis.setTimeout,
    clearTimeoutImpl = globalThis.clearTimeout,
  } = {}) {
    this.now = now;
    // Injected timer functions follow the browser API's unbound-call contract.
    this.setTimeoutImpl = setTimeoutImpl;
    this.clearTimeoutImpl = clearTimeoutImpl;
    this.timer = null;
    this.resolve = null;
    this.remaining = 0;
    this.startedAt = 0;
    this.paused = false;
  }

  wait(duration) {
    this.remaining = Math.max(0, Number(duration) || 0);
    return new Promise((resolve) => {
      this.resolve = resolve;
      this.start();
    });
  }

  start() {
    if (!this.resolve || this.paused) return;
    this.startedAt = this.now();
    const setTimeoutImpl = this.setTimeoutImpl;
    this.timer = setTimeoutImpl(() => this.finish(), this.remaining);
  }

  finish() {
    if (!this.resolve) return;
    const resolve = this.resolve;
    this.resolve = null;
    this.timer = null;
    this.remaining = 0;
    resolve();
  }

  pause() {
    if (this.paused || !this.resolve) return;
    this.paused = true;
    const clearTimeoutImpl = this.clearTimeoutImpl;
    if (this.timer !== null) clearTimeoutImpl(this.timer);
    this.timer = null;
    this.remaining = Math.max(
      0,
      this.remaining - (this.now() - this.startedAt),
    );
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this.start();
  }

  cancel() {
    const clearTimeoutImpl = this.clearTimeoutImpl;
    if (this.timer !== null) clearTimeoutImpl(this.timer);
    this.timer = null;
    const resolve = this.resolve;
    this.resolve = null;
    resolve?.();
  }
}


function browserAudio(url) {
  const audio = new Audio();
  audio.preload = "auto";
  audio.src = url;
  return audio;
}


function readableFallbackDuration(cue) {
  const characters = String(cue?.spoken_text || "").replace(/\s/g, "").length;
  return Math.min(9000, Math.max(2200, characters * 115));
}


function readablePlaybackTimeout(cue) {
  // Allow three readable-duration windows, bounded so missing media events
  // cannot stall a lesson indefinitely.
  return Math.min(30000, Math.max(6000, readableFallbackDuration(cue) * 3));
}


function actionList(value) {
  return Array.isArray(value) ? value : [];
}


export class CuePlayer {
  constructor({
    leadMs = 200,
    createAudio = browserAudio,
    createTimeline = () => new BrowserPausableTimeline(),
    applyActions = () => {},
    fallbackDuration = readableFallbackDuration,
    onsetTimeoutMs = 8000,
    playbackTimeout = readablePlaybackTimeout,
    onCueText = () => {},
    onBeatComplete = () => {},
    onAudioUnavailable = () => {},
    restoreSnapshot = () => {},
  } = {}) {
    this.leadMs = leadMs;
    this.createAudio = createAudio;
    this.createTimeline = createTimeline;
    this.applyActions = applyActions;
    this.fallbackDuration = fallbackDuration;
    this.onsetTimeoutMs = onsetTimeoutMs;
    this.playbackTimeout = playbackTimeout;
    this.onCueText = onCueText;
    this.onBeatComplete = onBeatComplete;
    this.onAudioUnavailable = onAudioUnavailable;
    this.restoreSnapshot = restoreSnapshot;
    this.token = 0;
    this.beat = null;
    this.snapshot = undefined;
    this.cues = [];
    this.currentIndex = -1;
    this.currentState = null;
    this.currentTimeline = null;
    this.currentAudio = null;
    this.preloaded = null;
    this.paused = false;
    this.beatCompleted = false;
    this.playbackMode = "beat";
    this.sequenceToken = null;
    this.sequenceResolve = null;
  }

  playBeat(beat, { snapshot } = {}) {
    this.stop();
    this.playbackMode = "beat";
    this.beat = beat;
    this.snapshot = snapshot;
    this.cues = Array.isArray(beat?.sync_cues)
      ? beat.sync_cues.filter((cue) => cue && typeof cue === "object")
      : [];
    this.currentIndex = -1;
    this.paused = false;
    this.beatCompleted = false;
    const token = this.token;
    if (this.cues.length === 0) {
      this.completeBeat(token);
      return;
    }
    this.launchCue(0, token);
  }

  playCueSequence(cues, token) {
    this.stop();
    this.playbackMode = "sequence";
    this.beat = null;
    this.snapshot = undefined;
    this.sequenceToken = token;
    this.cues = Array.isArray(cues)
      ? cues.filter((cue) => cue && typeof cue === "object")
      : [];
    this.currentIndex = -1;
    this.paused = false;
    this.beatCompleted = false;
    const playbackToken = this.token;
    const completion = new Promise((resolve) => {
      this.sequenceResolve = resolve;
    });
    if (this.cues.length === 0) {
      this.completeBeat(playbackToken);
      return completion;
    }
    this.launchCue(0, playbackToken);
    return completion;
  }

  launchCue(index, token) {
    this.playCue(index, token).catch(() => {
      if (token === this.token) this.completeBeat(token);
    });
  }

  async playCue(index, token) {
    if (token !== this.token) return;
    const cue = this.cues[index];
    if (!cue) {
      this.completeBeat(token);
      return;
    }
    this.currentIndex = index;
    const state = {
      token,
      index,
      cue,
      audio: null,
      listeners: null,
      startApplied: false,
      endApplied: false,
      fallbackStarted: false,
      unavailableNotified: false,
      settled: false,
      playPending: false,
      pausedDuringPlay: false,
      failureHandling: false,
      endedHandling: false,
      waitingForResume: false,
      resumePromise: null,
      resumeResolve: null,
      cancelled: false,
      cancelPromise: null,
      cancelResolve: null,
      failurePromise: null,
      failureResolve: null,
      onsetDeadline: null,
      onsetTask: null,
      resumeRequested: false,
      watchdogDeadline: null,
    };
    state.cancelPromise = new Promise((resolve) => {
      state.cancelResolve = resolve;
    });
    state.failurePromise = new Promise((resolve) => {
      state.failureResolve = resolve;
    });
    this.currentState = state;
    this.safeCall(
      this.onCueText,
      cue.display_text || cue.spoken_text || "",
      cue,
      index,
    );
    this.safeApply(actionList(cue.lead_actions));
    const waited = await this.wait(this.leadMs, token, state, "lead");
    if (!waited) return;
    if (!cue.audio_url) {
      await this.beginFallback(state);
      return;
    }
    await this.beginAudio(state);
  }

  async wait(duration, token, state, phase) {
    const timeline = this.createTimeline(phase);
    this.currentTimeline = timeline;
    this.phase = phase;
    try {
      const waiting = timeline.wait(duration);
      if (this.paused) timeline.pause?.();
      await waiting;
    } catch {
      return false;
    } finally {
      if (this.currentTimeline === timeline) this.currentTimeline = null;
    }
    return this.isCurrent(state, token);
  }

  async beginAudio(state) {
    if (!this.isCurrent(state, state.token)) return;
    const audio = this.takePreloaded(state.index)
      || this.createAudio(state.cue.audio_url);
    state.audio = audio;
    this.currentAudio = audio;
    this.phase = "audio";
    const onEnded = () => {
      this.settleAudio(state, "ended").catch(() => {});
    };
    const onError = () => {
      if (!this.isCurrent(state, state.token)) return;
      this.settleAudio(state, "failure").catch(() => {});
    };
    state.listeners = { onEnded, onError };
    audio.addEventListener?.("ended", onEnded);
    audio.addEventListener?.("error", onError);

    await this.startTrackedOnset(state);
  }

  startTrackedOnset(state) {
    if (!this.isCurrent(state, state.token)) return Promise.resolve();
    if (state.onsetTask) return state.onsetTask;
    const task = this.playAudioToOnset(state);
    state.onsetTask = task;
    task.finally(() => {
      if (state.onsetTask !== task) return;
      state.onsetTask = null;
      if (
        state.resumeRequested
        && !this.paused
        && this.phase === "audio"
        && this.isCurrent(state, state.token)
        && !state.failureHandling
        && !state.fallbackStarted
      ) {
        state.resumeRequested = false;
        this.startTrackedOnset(state);
      }
    }).catch(() => {});
    return task;
  }

  async playAudioToOnset(state) {
    const deadline = this.startOnsetDeadline(state);
    while (this.isCurrent(state, state.token) && !state.fallbackStarted) {
      if (this.paused) await this.waitForResume(state);
      if (!this.isCurrent(state, state.token)) return;

      state.playPending = true;
      state.pausedDuringPlay = false;
      let playResult;
      try {
        playResult = Promise.resolve(state.audio.play()).then(
          () => ({ kind: "play-resolved" }),
          (error) => ({ kind: "play-rejected", error }),
        );
      } catch (error) {
        playResult = Promise.resolve({ kind: "play-rejected", error });
      }

      const result = await Promise.race([
        playResult,
        deadline.promise,
        state.cancelPromise,
        state.failurePromise,
      ]);
      state.playPending = false;

      if (
        result.kind === "cancelled"
        || result.kind === "failure-claimed"
        || !this.isCurrent(state, state.token)
      ) return;
      if (result.kind === "onset-timeout") {
        await this.settleAudio(state, "failure");
        return;
      }

      const interruptedByPause = state.pausedDuringPlay;
      if (result.kind === "play-rejected") {
        const pauseAbort = (
          result.error?.name === "AbortError"
          && interruptedByPause
        );
        if (pauseAbort) {
          if (this.paused) await this.waitForResume(state);
          if (!this.isCurrent(state, state.token)) return;
          continue;
        }
        await this.settleAudio(state, "failure");
        return;
      }
      if (state.failureHandling || state.fallbackStarted) return;
      if (this.paused) await this.waitForResume(state);
      if (
        !this.isCurrent(state, state.token)
        || state.failureHandling
        || state.fallbackStarted
      ) return;
      if (interruptedByPause) continue;

      this.cancelOnsetDeadline(state);
      const firstOnset = !state.startApplied;
      this.applyStart(state);
      if (firstOnset) {
        this.preloadNext(state.index, state.token);
        this.startPlaybackWatchdog(state);
      }
      return;
    }
  }

  async settleAudio(state, outcome) {
    if (!this.isCurrent(state, state.token) || state.fallbackStarted) return;
    if (outcome === "failure") {
      if (state.failureHandling) return;
      state.failureHandling = true;
      state.failureResolve?.({ kind: "failure-claimed" });
    } else {
      if (state.endedHandling || state.failureHandling) return;
      state.endedHandling = true;
    }
    this.cancelOnsetDeadline(state);
    this.cancelPlaybackWatchdog(state);
    if (this.paused) await this.waitForResume(state);
    if (
      !this.isCurrent(state, state.token)
      || (outcome === "ended" && state.failureHandling)
    ) return;
    if (outcome === "failure") {
      await this.beginFallback(state);
    } else {
      this.applyStart(state);
      this.finishCue(state);
    }
  }

  startOnsetDeadline(state) {
    if (state.onsetDeadline) return state.onsetDeadline;
    const timeline = this.createTimeline("onset");
    const deadline = {
      timeline,
      cancelled: false,
      promise: null,
    };
    const duration = Math.max(0, Number(this.onsetTimeoutMs) || 0);
    deadline.promise = Promise.resolve(timeline.wait(duration)).then(
      () => ({
        kind: deadline.cancelled ? "cancelled" : "onset-timeout",
      }),
      () => ({
        kind: deadline.cancelled ? "cancelled" : "onset-timeout",
      }),
    );
    state.onsetDeadline = deadline;
    if (this.paused) timeline.pause?.();
    return deadline;
  }

  cancelOnsetDeadline(state) {
    const deadline = state?.onsetDeadline;
    if (!deadline) return;
    deadline.cancelled = true;
    deadline.timeline.cancel?.();
    state.onsetDeadline = null;
  }

  startPlaybackWatchdog(state) {
    if (
      state.watchdogDeadline
      || !this.isCurrent(state, state.token)
      || state.failureHandling
      || state.fallbackStarted
    ) return;
    let configured;
    try {
      configured = typeof this.playbackTimeout === "function"
        ? this.playbackTimeout(state.cue)
        : this.playbackTimeout;
    } catch {
      configured = readablePlaybackTimeout(state.cue);
    }
    const numeric = Number(configured);
    const duration = Number.isFinite(numeric)
      ? Math.max(0, numeric)
      : readablePlaybackTimeout(state.cue);
    const timeline = this.createTimeline("watchdog");
    const deadline = {
      timeline,
      cancelled: false,
    };
    state.watchdogDeadline = deadline;
    const waiting = Promise.resolve(timeline.wait(duration)).then(
      () => !deadline.cancelled,
      () => !deadline.cancelled,
    );
    if (this.paused) timeline.pause?.();
    waiting.then((expired) => {
      if (
        !expired
        || state.watchdogDeadline !== deadline
        || !this.isCurrent(state, state.token)
      ) return;
      state.watchdogDeadline = null;
      this.settleAudio(state, "failure").catch(() => {});
    }).catch(() => {});
  }

  cancelPlaybackWatchdog(state) {
    const deadline = state?.watchdogDeadline;
    if (!deadline) return;
    deadline.cancelled = true;
    deadline.timeline.cancel?.();
    state.watchdogDeadline = null;
  }

  async waitForResume(state) {
    if (!this.paused || !this.isCurrent(state, state.token)) return;
    if (!state.resumePromise) {
      state.resumePromise = new Promise((resolve) => {
        state.resumeResolve = resolve;
      });
    }
    state.waitingForResume = true;
    await state.resumePromise;
    state.waitingForResume = false;
    state.resumePromise = null;
    state.resumeResolve = null;
  }

  releaseResumeGate(state) {
    if (!state) return;
    const resolve = state?.resumeResolve;
    state.resumeResolve = null;
    resolve?.();
  }

  async beginFallback(state) {
    if (
      !this.isCurrent(state, state.token)
      || state.fallbackStarted
      || state.settled
    ) return;
    state.fallbackStarted = true;
    this.cancelOnsetDeadline(state);
    this.cancelPlaybackWatchdog(state);
    if (state.audio) {
      this.releaseAudio(state.audio, state.listeners, true);
      if (this.currentAudio === state.audio) this.currentAudio = null;
      state.audio = null;
      state.listeners = null;
    }
    if (!state.unavailableNotified) {
      state.unavailableNotified = true;
      this.safeCall(this.onAudioUnavailable, state.cue, state.index);
    }
    this.applyStart(state);
    const duration = typeof this.fallbackDuration === "function"
      ? this.fallbackDuration(state.cue)
      : this.fallbackDuration;
    const waited = await this.wait(
      duration,
      state.token,
      state,
      "fallback",
    );
    if (waited) this.finishCue(state);
  }

  applyStart(state) {
    if (state.startApplied || !this.isCurrent(state, state.token)) return;
    state.startApplied = true;
    this.safeApply(actionList(state.cue.start_actions));
  }

  finishCue(state) {
    if (!this.isCurrent(state, state.token) || state.settled) return;
    state.settled = true;
    this.cancelOnsetDeadline(state);
    this.cancelPlaybackWatchdog(state);
    if (state.audio) {
      this.releaseAudio(state.audio, state.listeners, false);
      if (this.currentAudio === state.audio) this.currentAudio = null;
    }
    if (!state.endApplied) {
      state.endApplied = true;
      this.safeApply(actionList(state.cue.end_actions));
    }
    const nextIndex = state.index + 1;
    if (nextIndex < this.cues.length) {
      this.launchCue(nextIndex, state.token);
    } else {
      this.completeBeat(state.token);
    }
  }

  completeBeat(token) {
    if (token !== this.token || this.beatCompleted) return;
    this.beatCompleted = true;
    this.phase = "complete";
    if (this.playbackMode === "sequence") {
      this.settleSequence(true);
      return;
    }
    this.safeCall(this.onBeatComplete, this.beat);
  }

  settleSequence(completed) {
    const resolve = this.sequenceResolve;
    if (!resolve) return;
    const token = this.sequenceToken;
    this.sequenceResolve = null;
    this.sequenceToken = null;
    resolve({ completed, token });
  }

  preloadNext(index, token) {
    const nextIndex = index + 1;
    const nextCue = this.cues[nextIndex];
    if (
      token !== this.token
      || !nextCue?.audio_url
      || this.preloaded
    ) return;
    try {
      const audio = this.createAudio(nextCue.audio_url);
      audio.preload = "auto";
      this.preloaded = { index: nextIndex, audio };
    } catch {
      // The next cue will use its normal readable fallback.
    }
  }

  takePreloaded(index) {
    if (this.preloaded?.index !== index) return null;
    const { audio } = this.preloaded;
    this.preloaded = null;
    return audio;
  }

  pause() {
    if (this.paused) return;
    this.paused = true;
    if (this.phase === "audio") {
      if (this.currentState?.playPending) {
        this.currentState.pausedDuringPlay = true;
      }
      this.currentAudio?.pause?.();
    }
    else this.currentTimeline?.pause?.();
    this.currentState?.onsetDeadline?.timeline.pause?.();
    this.currentState?.watchdogDeadline?.timeline.pause?.();
  }

  resume() {
    if (!this.paused) return;
    this.paused = false;
    this.currentTimeline?.resume?.();
    const state = this.currentState;
    state?.onsetDeadline?.timeline.resume?.();
    state?.watchdogDeadline?.timeline.resume?.();
    if (this.phase === "audio" && this.currentAudio) {
      if (state?.waitingForResume) {
        this.releaseResumeGate(state);
        return;
      }
      if (state?.playPending) return;
      if (state?.onsetTask) {
        if (state.startApplied) state.resumeRequested = true;
        return;
      }
      this.startTrackedOnset(state);
    }
  }

  stop() {
    this.token += 1;
    this.cancelState(this.currentState);
    this.currentTimeline?.cancel?.();
    this.currentTimeline = null;
    if (this.currentAudio) {
      this.releaseAudio(
        this.currentAudio,
        this.currentState?.listeners,
        true,
      );
    }
    this.currentAudio = null;
    if (this.preloaded?.audio) {
      this.releaseAudio(this.preloaded.audio, null, true);
    }
    this.preloaded = null;
    this.currentState = null;
    this.currentIndex = -1;
    this.phase = "stopped";
    this.paused = false;
    this.settleSequence(false);
  }

  cancelState(state) {
    if (!state || state.cancelled) return;
    state.cancelled = true;
    state.cancelResolve?.({ kind: "cancelled" });
    state.failureResolve?.({ kind: "cancelled" });
    this.releaseResumeGate(state);
    this.cancelOnsetDeadline(state);
    this.cancelPlaybackWatchdog(state);
  }

  replay() {
    const beat = this.beat;
    const snapshot = this.snapshot;
    if (!beat) return;
    this.stop();
    this.playbackMode = "beat";
    this.safeCall(this.restoreSnapshot, snapshot);
    this.beat = beat;
    this.snapshot = snapshot;
    this.cues = Array.isArray(beat.sync_cues)
      ? beat.sync_cues.filter((cue) => cue && typeof cue === "object")
      : [];
    this.beatCompleted = false;
    const token = this.token;
    if (this.cues.length === 0) {
      this.completeBeat(token);
      return;
    }
    this.launchCue(0, token);
  }

  isCurrent(state, token) {
    return (
      token === this.token
      && state === this.currentState
      && !state?.settled
      && !state?.cancelled
    );
  }

  safeApply(actions) {
    if (actions.length === 0) return;
    this.safeCall(this.applyActions, actions);
  }

  safeCall(callback, ...args) {
    try {
      callback(...args);
    } catch {
      // Host rendering and announcements must not break media settlement.
    }
  }

  releaseAudio(audio, listeners, pause) {
    if (listeners) {
      audio.removeEventListener?.("ended", listeners.onEnded);
      audio.removeEventListener?.("error", listeners.onError);
    }
    if (pause) audio.pause?.();
    audio.removeAttribute?.("src");
    audio.load?.();
  }
}


export { BrowserPausableTimeline };
