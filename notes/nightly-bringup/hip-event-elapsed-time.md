# Test plan: hipEventElapsedTime GPU-reach timing

**Covers hrx-system branch:** `users/zjgarvey/fix/hip-event-elapsed-time`
**Gap:** `hipEventElapsedTime` measured host *enqueue* spacing, so back-to-back
records around a kernel reported ~0 ms. MIOpen's auto-tuner rejects non-positive
elapsed times, breaking conv tuning. The fix samples a monotonic timestamp inside
a stream-ordered host callback (when the GPU *reaches* the event), so elapsed
reflects real GPU-reach duration. It also rejects a negative delta (events on
independent streams) rather than returning a bogus negative value.

> Scope note: this is coarse host-observed GPU-reach timing, not a device
> timestamp; cross-stream and concurrent re-record are intentionally out of
> scope (a device-timestamp implementation is a tracked follow-up). Sufficient
> for the MIOpen auto-tuner unblock.

## Assertions (GPU required — these exercise the queue/host-call path)

1. **Positive, plausible duration (the MIOpen regression guard).** Record
   `start`, enqueue a sized D2D `hipMemcpyAsync` (64–256 MiB) or a spin kernel of
   known minimum duration, record `stop`, `hipEventSynchronize(stop)`, then
   `hipEventElapsedTime(&ms, start, stop)`. Assert `hipSuccess`, **`ms > 0.0f`
   strictly**, and `ms` within a generous band (`0 < ms < 1000`; for the spin
   kernel `ms >= expected_min * 0.5`). Loop ≥100 iterations — assert no iteration
   returns a negative/zero `ms` or a spurious error.
2. **Not-ready before synchronize → `hipErrorNotReady`.** Record `start`/`stop`
   around a long-enough op and call `hipEventElapsedTime` *without* synchronizing.
   Assert `hipErrorNotReady` — never `hipErrorInvalidValue`.
3. **Disable-timing contract.** Both events created with `hipEventDisableTiming`
   → `hipEventElapsedTime` returns `hipErrorInvalidHandle`. Separately assert a
   disable-timing event still works for record + synchronize + query (the
   skipped-callback path must not break sync).
4. **Never-recorded.** Two events, record neither (or only one) →
   `hipEventElapsedTime` returns a clean error, not a crash or garbage `ms`.
5. **Cross-stream / negative guard.** Record `start` on stream A and `stop` on
   stream B, synchronize both, call `hipEventElapsedTime`. The branch now rejects
   a negative delta with a defined error (not a bogus negative `ms`) — assert the
   result is a defined error or a non-negative `ms`. A negative value here would
   reproduce the original MIOpen complaint.

## Status
Review confirmed the ns→ms conversion, the completion/no-race ordering, and the
not-ready/disable-timing error codes. Open follow-ups documented on the branch:
device-side timestamps (to remove the per-record flush + host round-trip) and the
re-record error-code refinement. Implement as a `tests/event/` Catch2 case via
`hip().hipEventRecord/hipEventElapsedTime` on a real device.
