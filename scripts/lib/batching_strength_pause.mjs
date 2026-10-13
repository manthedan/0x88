// A missing snapshot is NOT proof of idleness. Throw so the driver's owned
// browser is closed before benchlock can be released. Dependencies are injected
// for failure-path tests; this helper never launches a browser on its own.
export async function pauseStrengthPage({ requestPause, readSnapshot, sleep, now = Date.now, timeoutMs = 180_000 }) {
  await requestPause();
  const deadline = now() + timeoutMs;
  while (now() < deadline) {
    const snapshot = await readSnapshot();
    if (!snapshot) throw new Error('pause confirmation disappeared; close the owned browser before releasing benchlock');
    if (snapshot.phase === 'paused' || snapshot.phase === 'done') return snapshot;
    if (snapshot.phase === 'error') throw new Error('page failed before pause confirmation');
    await sleep(1000);
  }
  throw new Error('pause grace expired; close the owned browser before releasing benchlock');
}

// runAgent rejects CLI failures/timeouts, but killing a CLI isn't killing its
// browser. Keep the caller (and hence benchlock) alive until the owned session
// acknowledges closure. A broken control channel requires manual recovery;
// silently returning would let GPU inference overlap the next benchmark.
export async function closeStrengthBrowser({ close, sleep, onRetry = () => {} }) {
  for (;;) {
    try {
      const result = await close();
      if (result?.closed !== true) throw new Error('browser close did not acknowledge closed:true');
      return;
    } catch (error) {
      onRetry(error);
      await sleep(1000);
    }
  }
}
