/**
 * Polling helper for job status.
 *
 * Polls `fetcher` every `intervalMs` while the predicate says the job is still
 * pending (queued/running); stops as soon as it reaches a terminal state or the
 * owner disposes. Used by ItemPage (ingest) and DescriptionsPage (generation).
 */
import { createSignal, onCleanup, type Accessor } from 'solid-js';

export const POLL_INTERVAL_MS = 2000;

export interface PollController<T> {
  /** Latest polled value (undefined until the first tick). */
  value: Accessor<T | undefined>;
  /** True while a tick is in flight. */
  pending: Accessor<boolean>;
  stop: () => void;
}

/**
 * Start polling `fetcher` until `isTerminal(latest)` is true.
 * Every tick replaces `value`; the first fetch happens immediately.
 */
export function pollUntil<T>(
  fetcher: () => Promise<T>,
  isTerminal: (value: T) => boolean,
  intervalMs: number = POLL_INTERVAL_MS,
): PollController<T> {
  const [value, setValue] = createSignal<T>();
  const [pending, setPending] = createSignal(false);
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let inFlight = false;

  const tick = async () => {
    if (stopped || inFlight) return;
    inFlight = true;
    setPending(true);
    try {
      const next = await fetcher();
      if (stopped) return;
      setValue(() => next);
      if (isTerminal(next)) {
        stopped = true;
        return;
      }
    } catch {
      // Transient fetch errors: keep polling; UI shows the last good value.
    } finally {
      inFlight = false;
      setPending(false);
    }
    if (!stopped) timer = setTimeout(tick, intervalMs);
  };

  const stop = () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = undefined;
    onCleanup(stop);
  };

  void tick();
  return { value, pending, stop };
}

/** Job status helpers shared by ingest + description jobs. */
export const INGEST_TERMINAL = new Set(['succeeded', 'failed']);
export const DESC_TERMINAL = new Set([
  'draft_ready',
  'approved',
  'rejected',
  'published',
  'failed',
]);

export function isJobPending(status: string, terminal: Set<string>): boolean {
  return !terminal.has(status);
}
