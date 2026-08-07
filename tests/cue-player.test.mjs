import test from "node:test";
import assert from "node:assert/strict";

import {
  BrowserPausableTimeline,
  CuePlayer,
} from "../app/static/cue-player.mjs";


const flush = async () => {
  for (let turn = 0; turn < 8; turn += 1) {
    await Promise.resolve();
  }
};


test("browser timeline calls injected setTimeout without a receiver", async () => {
  let scheduled = null;
  function strictSetTimeout(callback, delay) {
    if (this !== undefined) throw new TypeError("Illegal invocation");
    scheduled = { callback, delay };
    return 41;
  }
  const timeline = new BrowserPausableTimeline({
    now: () => 0,
    setTimeoutImpl: strictSetTimeout,
    clearTimeoutImpl: () => {},
  });

  const waiting = timeline.wait(200);
  if (!scheduled) await waiting;
  assert.equal(scheduled.delay, 200);
  scheduled.callback();
  await waiting;

  assert.equal(timeline.timer, null);
});


test("browser timeline pauses resumes and cancels with unbound clearTimeout", async () => {
  let now = 0;
  let nextTimer = 1;
  const scheduled = new Map();
  const setCalls = [];
  const clearCalls = [];
  const setTimeoutImpl = (callback, delay) => {
    const timer = nextTimer;
    nextTimer += 1;
    setCalls.push({ timer, delay });
    scheduled.set(timer, callback);
    return timer;
  };
  function strictClearTimeout(timer) {
    if (this !== undefined) throw new TypeError("Illegal invocation");
    clearCalls.push(timer);
    scheduled.delete(timer);
  }
  const timeline = new BrowserPausableTimeline({
    now: () => now,
    setTimeoutImpl,
    clearTimeoutImpl: strictClearTimeout,
  });

  const waiting = timeline.wait(200);
  now = 80;
  timeline.pause();

  assert.deepEqual(setCalls, [{ timer: 1, delay: 200 }]);
  assert.deepEqual(clearCalls, [1]);
  assert.equal(timeline.remaining, 120);

  timeline.resume();
  assert.deepEqual(setCalls, [
    { timer: 1, delay: 200 },
    { timer: 2, delay: 120 },
  ]);

  timeline.cancel();
  await waiting;

  assert.deepEqual(clearCalls, [1, 2]);
  assert.equal(scheduled.size, 0);
  assert.equal(timeline.timer, null);
});


class FakeTimeline {
  constructor(events, recordWait = true) {
    this.events = events;
    this.recordWait = recordWait;
    this.remaining = 0;
    this.paused = false;
    this.cancelled = false;
    this.resolve = null;
  }

  wait(duration) {
    this.remaining = duration;
    if (this.recordWait) this.events.push(`wait:${duration}`);
    return new Promise((resolve) => {
      this.resolve = resolve;
    });
  }

  elapse(duration) {
    if (this.paused || this.cancelled || !this.resolve) return;
    this.remaining = Math.max(0, this.remaining - duration);
    if (this.remaining === 0) {
      const resolve = this.resolve;
      this.resolve = null;
      resolve();
    }
  }

  complete() {
    this.elapse(this.remaining);
  }

  pause() {
    this.paused = true;
  }

  resume() {
    this.paused = false;
  }

  cancel() {
    this.cancelled = true;
    const resolve = this.resolve;
    this.resolve = null;
    resolve?.();
  }
}


class FakeAudio {
  constructor(url, events, playMode = "resolve") {
    this.url = url;
    this.events = events;
    this.playMode = playMode;
    this.listeners = new Map();
    this.playCalls = 0;
    this.pauseCalls = 0;
    this.preload = "";
    this.src = url;
    this.deferredPlay = null;
    this.pendingPlay = null;
  }

  addEventListener(type, callback) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(callback);
  }

  removeEventListener(type, callback) {
    this.listeners.get(type)?.delete(callback);
  }

  play() {
    this.playCalls += 1;
    this.events.push(`audio:play:${this.url}`);
    const modes = Array.isArray(this.playMode)
      ? this.playMode
      : [this.playMode];
    const mode = modes[Math.min(this.playCalls - 1, modes.length - 1)];
    if (["reject", "not-allowed", "network"].includes(mode)) {
      const error = new Error("blocked");
      if (mode === "not-allowed") error.name = "NotAllowedError";
      if (mode === "network") error.name = "NetworkError";
      return Promise.reject(error);
    }
    if (["defer", "pause-abort", "never"].includes(mode)) {
      if (!this.deferredPlay || Array.isArray(this.playMode)) {
        const deferred = {};
        deferred.mode = mode;
        deferred.promise = new Promise((resolve, reject) => {
          deferred.resolve = resolve;
          deferred.reject = reject;
        });
        this.deferredPlay = deferred;
        this.pendingPlay = deferred;
      }
      return this.deferredPlay.promise;
    }
    return Promise.resolve();
  }

  pause() {
    this.pauseCalls += 1;
    this.events.push(`audio:pause:${this.url}`);
    if (this.pendingPlay?.mode === "pause-abort") {
      const error = new Error("pause interrupted play");
      error.name = "AbortError";
      const pending = this.pendingPlay;
      this.pendingPlay = null;
      pending.reject(error);
    }
  }

  removeAttribute(name) {
    if (name === "src") this.src = "";
  }

  load() {}

  emit(type) {
    this.events.push(`audio:${type}:${this.url}`);
    for (const callback of [...(this.listeners.get(type) || [])]) {
      callback();
    }
  }
}


function cue(id, {
  audioUrl = `${id}.mp3`,
  lead = [],
  start = [],
  end = [],
} = {}) {
  return {
    cue_id: id,
    spoken_text: `spoken ${id}`,
    audio_url: audioUrl,
    lead_actions: lead,
    start_actions: start,
    end_actions: end,
  };
}


function createHarness({ playModes = [] } = {}) {
  const events = [];
  const timelines = [];
  const onsetTimelines = [];
  const watchdogTimelines = [];
  const audios = [];
  const unavailable = [];
  const completed = [];
  const restored = [];
  const player = new CuePlayer({
    createAudio: (url) => {
      const audio = new FakeAudio(
        url,
        events,
        playModes[audios.length] || "resolve",
      );
      audios.push(audio);
      return audio;
    },
    createTimeline: (phase) => {
      const bounded = phase === "onset" || phase === "watchdog";
      const timeline = new FakeTimeline(events, !bounded);
      if (phase === "onset") onsetTimelines.push(timeline);
      else if (phase === "watchdog") watchdogTimelines.push(timeline);
      else timelines.push(timeline);
      return timeline;
    },
    applyActions: (actions) => {
      for (const action of actions) events.push(action.event);
    },
    fallbackDuration: () => 400,
    onsetTimeoutMs: 500,
    playbackTimeout: () => 600,
    onCueText: () => {},
    onBeatComplete: () => {
      events.push("beat:complete");
      completed.push(true);
    },
    onAudioUnavailable: (activeCue) => {
      unavailable.push(activeCue.cue_id);
    },
    restoreSnapshot: (snapshot) => {
      restored.push(snapshot);
      events.push("snapshot:restore");
    },
  });
  return {
    player,
    events,
    timelines,
    onsetTimelines,
    watchdogTimelines,
    audios,
    unavailable,
    completed,
    restored,
  };
}


test("two cues play semantic action boundaries in exact order", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        lead: [{ event: "lead:problem-root" }],
        start: [{ event: "start:write-substitution" }],
        end: [{ event: "end:weaken-root" }],
      }),
      cue("cue-002", {
        lead: [{ event: "lead:problem-equation" }],
        start: [{ event: "start:write-result" }],
        end: [{ event: "end:finish-result" }],
      }),
    ],
  });

  harness.timelines[0].complete();
  await flush();
  harness.audios[0].emit("ended");
  await flush();
  harness.timelines[1].complete();
  await flush();
  harness.audios[1].emit("ended");
  await flush();

  assert.deepEqual(harness.events, [
    "lead:problem-root",
    "wait:200",
    "audio:play:cue-001.mp3",
    "start:write-substitution",
    "audio:ended:cue-001.mp3",
    "end:weaken-root",
    "lead:problem-equation",
    "wait:200",
    "audio:play:cue-002.mp3",
    "start:write-result",
    "audio:ended:cue-002.mp3",
    "end:finish-result",
    "beat:complete",
  ]);
});


test("cue without visual actions still plays and completes", async () => {
  const harness = createHarness();
  harness.player.playBeat({ sync_cues: [cue("cue-001")] });
  harness.timelines[0].complete();
  await flush();
  harness.audios[0].emit("ended");
  await flush();

  assert.equal(harness.audios[0].playCalls, 1);
  assert.equal(harness.completed.length, 1);
});


test("pause freezes the lead delay and resume uses the exact remainder", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:write-result" }],
      }),
    ],
  });
  const delay = harness.timelines[0];
  delay.elapse(80);
  harness.player.pause();
  delay.elapse(120);
  assert.equal(delay.remaining, 120);
  assert.equal(harness.audios.length, 0);
  assert.equal(harness.events.includes("start:write-result"), false);

  harness.player.resume();
  delay.elapse(119);
  assert.equal(harness.audios.length, 0);
  delay.elapse(1);
  await flush();
  assert.equal(harness.audios[0].playCalls, 1);
  assert.equal(harness.events.includes("start:write-result"), true);
});


test("pause during audio resumes the same cue without restarting actions", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:once" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  harness.player.resume();
  await flush();

  assert.equal(harness.audios.length, 1);
  assert.equal(audio.pauseCalls, 1);
  assert.equal(audio.playCalls, 2);
  assert.equal(
    harness.events.filter((event) => event === "start:once").length,
    1,
  );
});


test("pending play resolution waits for resume before onset and preload", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:pending-success" }],
      }),
      cue("cue-002"),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.deferredPlay.resolve();
  await flush();

  assert.equal(audio.pauseCalls, 1);
  assert.equal(audio.playCalls, 1);
  assert.equal(harness.audios.length, 1);
  assert.equal(harness.events.includes("start:pending-success"), false);
  assert.equal(harness.completed.length, 0);

  harness.player.resume();
  await flush();
  assert.equal(harness.audios[0], audio);
  assert.equal(audio.playCalls, 2);
  assert.equal(harness.audios.length, 2);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:pending-success"
    ).length,
    1,
  );
});


test("pause AbortError retries the same initial audio after resume", async () => {
  const harness = createHarness({
    playModes: [["pause-abort", "resolve"]],
  });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:abort-initial" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  await flush();

  assert.equal(audio.playCalls, 1);
  assert.equal(harness.onsetTimelines.length, 1);
  assert.deepEqual(harness.unavailable, []);
  assert.equal(harness.events.includes("start:abort-initial"), false);

  harness.player.resume();
  await flush();
  assert.equal(harness.audios.length, 1);
  assert.equal(harness.audios[0], audio);
  assert.equal(audio.playCalls, 2);
  assert.equal(harness.onsetTimelines.length, 1);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:abort-initial"
    ).length,
    1,
  );
});


test("rapid pause AbortError during resume never duplicates onset", async () => {
  const harness = createHarness({
    playModes: [["resolve", "pause-abort", "resolve"]],
  });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:abort-resume" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  harness.player.resume();
  await flush();
  assert.equal(audio.playCalls, 2);
  assert.equal(harness.onsetTimelines.length, 2);
  harness.player.pause();
  await flush();

  assert.deepEqual(harness.unavailable, []);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:abort-resume"
    ).length,
    1,
  );

  harness.player.resume();
  await flush();
  assert.equal(harness.audios.length, 1);
  assert.equal(audio.playCalls, 3);
  assert.equal(harness.onsetTimelines.length, 2);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:abort-resume"
    ).length,
    1,
  );
});


test("genuine play rejections still enter one fallback", async () => {
  for (const mode of ["not-allowed", "network"]) {
    const harness = createHarness({ playModes: [mode] });
    harness.player.playBeat({
      sync_cues: [
        cue("cue-001", {
          start: [{ event: "start:genuine-reject" }],
        }),
      ],
    });
    harness.timelines[0].complete();
    await flush();

    assert.deepEqual(harness.unavailable, ["cue-001"]);
    assert.equal(harness.timelines.length, 2);
    assert.equal(
      harness.events.filter(
        (event) => event === "start:genuine-reject"
      ).length,
      1,
    );
  }
});


test("pending play rejection waits for resume before one fallback", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:pending-reject" }],
        end: [{ event: "end:pending-reject" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.deferredPlay.reject(new Error("blocked"));
  await flush();

  assert.equal(audio.pauseCalls, 1);
  assert.equal(audio.playCalls, 1);
  assert.equal(harness.timelines.length, 1);
  assert.equal(harness.events.includes("start:pending-reject"), false);
  assert.deepEqual(harness.unavailable, []);

  harness.player.resume();
  await flush();
  assert.equal(harness.audios.length, 1);
  assert.equal(audio.playCalls, 1);
  assert.equal(harness.timelines.length, 2);
  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:pending-reject"
    ).length,
    1,
  );
  harness.timelines[1].complete();
  await flush();
  assert.equal(
    harness.events.filter(
      (event) => event === "end:pending-reject"
    ).length,
    1,
  );
  assert.equal(harness.completed.length, 1);
});


test("pending audio error waits for resume and ignores late play outcome", async () => {
  for (const lateOutcome of ["resolve", "reject"]) {
    const harness = createHarness({ playModes: ["defer"] });
    harness.player.playBeat({
      sync_cues: [
        cue("cue-001", {
          start: [{ event: "start:pending-error" }],
          end: [{ event: "end:pending-error" }],
        }),
      ],
    });
    harness.timelines[0].complete();
    await flush();
    const audio = harness.audios[0];
    harness.player.pause();
    audio.emit("error");
    await flush();

    assert.deepEqual(harness.unavailable, []);
    assert.equal(harness.timelines.length, 1);
    assert.equal(harness.events.includes("start:pending-error"), false);

    harness.player.resume();
    await flush();
    assert.deepEqual(harness.unavailable, ["cue-001"]);
    assert.equal(harness.timelines.length, 2);
    assert.equal(
      harness.events.filter(
        (event) => event === "start:pending-error"
      ).length,
      1,
    );
    audio.deferredPlay[lateOutcome](
      lateOutcome === "reject" ? new Error("late rejection") : undefined
    );
    await flush();
    assert.deepEqual(harness.unavailable, ["cue-001"]);
    assert.equal(harness.timelines.length, 2);
    assert.equal(
      harness.events.filter(
        (event) => event === "start:pending-error"
      ).length,
      1,
    );

    harness.timelines[1].complete();
    await flush();
    assert.equal(
      harness.events.filter(
        (event) => event === "end:pending-error"
      ).length,
      1,
    );
    assert.equal(harness.completed.length, 1);
  }
});


test("started audio error while paused delays fallback until resume", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:started-error" }],
        end: [{ event: "end:started-error" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  assert.equal(harness.events.includes("start:started-error"), true);
  harness.player.pause();
  audio.emit("error");
  await flush();

  assert.deepEqual(harness.unavailable, []);
  assert.equal(harness.timelines.length, 1);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:started-error"
    ).length,
    1,
  );

  harness.player.resume();
  await flush();
  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(harness.timelines.length, 2);
  harness.timelines[1].complete();
  await flush();
  assert.equal(
    harness.events.filter(
      (event) => event === "start:started-error"
    ).length,
    1,
  );
  assert.equal(
    harness.events.filter(
      (event) => event === "end:started-error"
    ).length,
    1,
  );
  assert.equal(harness.completed.length, 1);
});


test("ended cannot overtake a paused pending-audio failure", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:error-before-ended" }],
        end: [{ event: "end:error-before-ended" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.emit("error");
  audio.emit("ended");
  await flush();

  assert.deepEqual(harness.unavailable, []);
  assert.equal(harness.events.includes("start:error-before-ended"), false);
  assert.equal(harness.events.includes("end:error-before-ended"), false);
  assert.equal(harness.completed.length, 0);

  harness.player.resume();
  await flush();
  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(harness.timelines.length, 2);
  audio.deferredPlay.resolve();
  await flush();
  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:error-before-ended"
    ).length,
    1,
  );
  harness.timelines[1].complete();
  await flush();
  assert.equal(
    harness.events.filter(
      (event) => event === "end:error-before-ended"
    ).length,
    1,
  );
  assert.equal(harness.completed.length, 1);
});


test("paused ended waits before ending and starting the next cue", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:ended-gate" }],
        end: [{ event: "end:ended-gate" }],
      }),
      cue("cue-002", {
        lead: [{ event: "lead:next-after-ended" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.emit("ended");
  await flush();

  assert.equal(harness.events.includes("end:ended-gate"), false);
  assert.equal(harness.events.includes("lead:next-after-ended"), false);
  assert.equal(harness.completed.length, 0);

  harness.player.resume();
  await flush();
  audio.emit("ended");
  await flush();
  assert.equal(
    harness.events.filter(
      (event) => event === "start:ended-gate"
    ).length,
    1,
  );
  assert.equal(
    harness.events.filter(
      (event) => event === "end:ended-gate"
    ).length,
    1,
  );
  assert.equal(
    harness.events.filter(
      (event) => event === "lead:next-after-ended"
    ).length,
    1,
  );
  assert.equal(harness.timelines.length, 2);
});


test("failure owns paused ended and stop invalidates queued ended", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:ended-before-error" }],
        end: [{ event: "end:ended-before-error" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.emit("ended");
  audio.emit("error");
  await flush();
  harness.player.resume();
  await flush();

  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(harness.timelines.length, 2);
  assert.equal(
    harness.events.filter(
      (event) => event === "start:ended-before-error"
    ).length,
    1,
  );
  assert.equal(harness.events.includes("end:ended-before-error"), false);
  harness.timelines[1].complete();
  await flush();
  assert.equal(
    harness.events.filter(
      (event) => event === "end:ended-before-error"
    ).length,
    1,
  );
  assert.equal(harness.completed.length, 1);

  const stopped = createHarness();
  stopped.player.playBeat({
    sync_cues: [
      cue("cue-stop", {
        start: [{ event: "start:stopped-ended" }],
        end: [{ event: "end:stopped-ended" }],
      }),
    ],
  });
  stopped.timelines[0].complete();
  await flush();
  stopped.player.pause();
  stopped.audios[0].emit("ended");
  await flush();
  stopped.player.stop();
  await flush();
  assert.equal(stopped.events.includes("end:stopped-ended"), false);
  assert.equal(stopped.completed.length, 0);
});


test("stop invalidates an audio error waiting behind resume", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:error-gate" }],
        end: [{ event: "end:error-gate" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.emit("error");
  await flush();
  harness.player.stop();
  audio.deferredPlay.resolve();
  await flush();

  assert.deepEqual(harness.unavailable, []);
  assert.equal(harness.timelines.length, 1);
  assert.equal(harness.events.includes("start:error-gate"), false);
  assert.equal(harness.events.includes("end:error-gate"), false);
  assert.equal(harness.completed.length, 0);
});


test("stop releases a paused play outcome without stale work", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:resume-gate" }],
        end: [{ event: "end:resume-gate" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.pause();
  audio.deferredPlay.resolve();
  await flush();
  harness.player.stop();
  await flush();

  assert.equal(harness.events.includes("start:resume-gate"), false);
  assert.equal(harness.events.includes("end:resume-gate"), false);
  assert.equal(harness.completed.length, 0);
});


test("stop invalidates stale timer, audio, error, and play callbacks", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:stale" }],
        end: [{ event: "end:stale" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  const audio = harness.audios[0];
  harness.player.stop();
  audio.emit("ended");
  audio.emit("error");
  audio.deferredPlay.resolve();
  await flush();

  assert.equal(audio.pauseCalls, 1);
  assert.equal(harness.events.includes("start:stale"), false);
  assert.equal(harness.events.includes("end:stale"), false);
  assert.equal(harness.completed.length, 0);
});


test("never-settling play times out or cancels without stale work", async () => {
  const timedOut = createHarness({ playModes: ["never"] });
  timedOut.player.playBeat({
    sync_cues: [
      cue("cue-timeout", {
        start: [{ event: "start:onset-timeout" }],
      }),
    ],
  });
  timedOut.timelines[0].complete();
  await flush();
  assert.equal(timedOut.onsetTimelines.length, 1);
  const onset = timedOut.onsetTimelines[0];
  onset.elapse(200);
  timedOut.player.pause();
  onset.elapse(300);
  assert.equal(onset.remaining, 300);
  assert.deepEqual(timedOut.unavailable, []);
  timedOut.player.resume();
  onset.elapse(299);
  await flush();
  assert.deepEqual(timedOut.unavailable, []);
  onset.elapse(1);
  await flush();
  assert.deepEqual(timedOut.unavailable, ["cue-timeout"]);
  assert.equal(timedOut.events.includes("start:onset-timeout"), true);
  assert.equal(timedOut.timelines.length, 2);

  const stopped = createHarness({ playModes: ["never"] });
  stopped.player.playBeat({
    sync_cues: [
      cue("cue-stop", {
        start: [{ event: "start:cancelled-onset" }],
      }),
    ],
  });
  stopped.timelines[0].complete();
  await flush();
  assert.equal(stopped.onsetTimelines.length, 1);
  stopped.player.stop();
  stopped.onsetTimelines[0].complete();
  await flush();
  assert.deepEqual(stopped.unavailable, []);
  assert.equal(stopped.events.includes("start:cancelled-onset"), false);
  assert.equal(stopped.completed.length, 0);
  assert.equal(stopped.onsetTimelines[0].cancelled, true);
});


test("playback watchdog falls back once when audio never ends", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        start: [{ event: "start:watchdog" }],
        end: [{ event: "end:watchdog" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  assert.equal(harness.watchdogTimelines.length, 1);
  harness.watchdogTimelines[0].complete();
  await flush();

  assert.deepEqual(harness.unavailable, ["cue-001"]);
  assert.equal(
    harness.events.filter((event) => event === "start:watchdog").length,
    1,
  );
  assert.equal(harness.timelines.length, 2);
  harness.timelines[1].complete();
  await flush();
  assert.equal(
    harness.events.filter((event) => event === "end:watchdog").length,
    1,
  );
  assert.equal(harness.completed.length, 1);
});


test("pause freezes playback watchdog until its remaining active time", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [cue("cue-001")],
  });
  harness.timelines[0].complete();
  await flush();
  const watchdog = harness.watchdogTimelines[0];
  watchdog.elapse(200);
  harness.player.pause();
  watchdog.elapse(400);
  assert.equal(watchdog.remaining, 400);
  assert.deepEqual(harness.unavailable, []);

  harness.player.resume();
  await flush();
  watchdog.elapse(399);
  assert.deepEqual(harness.unavailable, []);
  watchdog.elapse(1);
  await flush();
  assert.deepEqual(harness.unavailable, ["cue-001"]);
});


test("normal ended cancels playback watchdog permanently", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [cue("cue-001")],
  });
  harness.timelines[0].complete();
  await flush();
  const watchdog = harness.watchdogTimelines[0];
  harness.audios[0].emit("ended");
  await flush();
  assert.equal(watchdog.cancelled, true);
  watchdog.complete();
  await flush();
  assert.deepEqual(harness.unavailable, []);
  assert.equal(harness.completed.length, 1);
});


test("replay restores one snapshot and restarts cue zero", async () => {
  const harness = createHarness();
  const snapshot = { board: "entry" };
  const beat = {
    sync_cues: [
      cue("cue-001", {
        lead: [{ event: "lead:cue-zero" }],
        start: [{ event: "start:cue-zero" }],
        end: [{ event: "end:cue-zero" }],
      }),
    ],
  };
  harness.player.playBeat(beat, { snapshot });
  harness.timelines[0].complete();
  await flush();
  const staleAudio = harness.audios[0];

  harness.player.replay();
  assert.deepEqual(harness.restored, [snapshot]);
  assert.equal(
    harness.events.filter((event) => event === "lead:cue-zero").length,
    2,
  );
  staleAudio.emit("ended");
  await flush();
  assert.equal(harness.events.includes("end:cue-zero"), false);

  harness.timelines[1].complete();
  await flush();
  assert.notEqual(harness.audios[1], staleAudio);
  assert.equal(
    harness.events.filter((event) => event === "start:cue-zero").length,
    2,
  );
});


test("missing audio uses a pausable fallback without changing cue order", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [
      cue("cue-001", {
        audioUrl: null,
        lead: [{ event: "lead:fallback" }],
        start: [{ event: "start:fallback" }],
        end: [{ event: "end:fallback" }],
      }),
    ],
  });
  harness.timelines[0].complete();
  await flush();
  assert.deepEqual(harness.events, [
    "lead:fallback",
    "wait:200",
    "start:fallback",
    "wait:400",
  ]);
  assert.deepEqual(harness.unavailable, ["cue-001"]);

  harness.player.pause();
  harness.timelines[1].elapse(100);
  assert.equal(harness.timelines[1].remaining, 400);
  harness.player.resume();
  harness.timelines[1].complete();
  await flush();
  assert.deepEqual(harness.events.slice(-2), [
    "end:fallback",
    "beat:complete",
  ]);
});


test("play rejection and runtime error fall back exactly once", async () => {
  const rejected = createHarness({ playModes: ["reject"] });
  rejected.player.playBeat({
    sync_cues: [
      cue("cue-reject", {
        lead: [{ event: "lead:reject" }],
        start: [{ event: "start:reject" }],
        end: [{ event: "end:reject" }],
      }),
    ],
  });
  rejected.timelines[0].complete();
  await flush();
  rejected.timelines[1].complete();
  await flush();
  assert.equal(
    rejected.events.filter((event) => event === "lead:reject").length,
    1,
  );
  assert.equal(
    rejected.events.filter((event) => event === "start:reject").length,
    1,
  );
  assert.equal(
    rejected.events.filter((event) => event === "end:reject").length,
    1,
  );

  const errored = createHarness();
  errored.player.playBeat({
    sync_cues: [
      cue("cue-error", {
        lead: [{ event: "lead:error" }],
        start: [{ event: "start:error" }],
        end: [{ event: "end:error" }],
      }),
    ],
  });
  errored.timelines[0].complete();
  await flush();
  errored.audios[0].emit("error");
  await flush();
  errored.audios[0].emit("ended");
  errored.timelines[1].complete();
  await flush();
  for (const boundary of ["lead:error", "start:error", "end:error"]) {
    assert.equal(
      errored.events.filter((event) => event === boundary).length,
      1,
    );
  }
  assert.equal(errored.completed.length, 1);
});


test("beat completes exactly once after the final cue end", async () => {
  const harness = createHarness();
  harness.player.playBeat({
    sync_cues: [cue("cue-001"), cue("cue-002")],
  });
  harness.timelines[0].complete();
  await flush();
  harness.audios[0].emit("ended");
  await flush();
  assert.equal(harness.completed.length, 0);
  harness.timelines[1].complete();
  await flush();
  harness.audios[1].emit("ended");
  harness.audios[1].emit("ended");
  await flush();
  assert.equal(harness.completed.length, 1);
});


test("next cue preloads only after current audio starts and never plays", async () => {
  const harness = createHarness({ playModes: ["defer"] });
  harness.player.playBeat({
    sync_cues: [cue("cue-001"), cue("cue-002")],
  });
  assert.equal(harness.audios.length, 0);
  harness.timelines[0].complete();
  await flush();
  assert.equal(harness.audios.length, 1);
  assert.equal(harness.audios[0].playCalls, 1);
  harness.audios[0].deferredPlay.resolve();
  await flush();

  assert.equal(harness.audios.length, 2);
  assert.equal(harness.audios[1].playCalls, 0);
  assert.equal(harness.timelines.length, 1);
  harness.audios[0].emit("ended");
  await flush();
  harness.timelines[1].complete();
  await flush();
  assert.equal(harness.audios.length, 2);
  assert.equal(harness.audios[1].playCalls, 1);
});


test("empty and malformed cue lists complete safely without actions", async () => {
  for (const syncCues of [null, [], [null]]) {
    const harness = createHarness();
    harness.player.playBeat({ sync_cues: syncCues });
    await flush();
    assert.deepEqual(harness.events, ["beat:complete"]);
    assert.equal(harness.audios.length, 0);
    assert.equal(harness.timelines.length, 0);
  }
});
