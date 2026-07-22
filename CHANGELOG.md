# Changelog

> These changelogs are AI-written and human-reviewed, because no one (least of all my wife
> and kids) wants me wasting my precious time meticulously documenting this shit, useful
> though it may be.

All notable user-facing changes to expenvelope are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`integrate_interval`, `get_upper_integration_bound`, and range `max_level`/`min_level`
  no longer copy the tail of the segment list on every call.** They sliced
  `self.segments[start_index:]` before iterating, making short queries near the beginning
  of a long envelope cost time proportional to the envelope's total segment count
  (~270x slowdown on a 66k-segment envelope). They now iterate by index; results are
  numerically identical.

## [0.8.0] - 2026-07-12

### Added

- `snap_float_to_nice_decimal`, now part of the public API. Envelope durations are derived
  as differences between segment boundary times, so an envelope built on a regular grid
  accumulates floating-point dust; this snaps a value to a nearby nice decimal.
  (It previously lived in clockblocks, which now imports it from here.)
- `Envelope.get_durations(rounded=False)`. Pass `rounded=True` to get the intended segment
  lengths rather than their floating-point residue — used for repr and serialization,
  without disturbing the exact boundary times the curve math relies on.

### Changed

- Envelope durations are rounded in `repr` and in JSON serialization, so a curve built from
  clean durations reads back as clean durations.
- `Envelope.insert_interpolated`: the `min_difference` parameter becomes `relative_tolerance`,
  and it now means something different. It was an *absolute* distance a new point had to keep
  from existing control points; it is now a *fraction of the width of the segment the point
  falls in*, so the tolerance scales with the segment instead of being a fixed number that is
  too coarse for narrow segments and too fine for wide ones. The default is still `1e-7`, but
  it is now `1e-7 * segment_width`. **Breaking** if you passed it by keyword, and a behavior
  change even if you didn't.
- `get_upper_integration_bound` is solved per segment, according to the segment's shape:
  linear and constant segments get a closed-form solution (the positive root of a quadratic,
  written so it stays exact as the slope approaches zero), while exponential segments use a
  bracketed Newton method — Newton steps kept inside a shrinking bracket, so it converges
  fast but cannot escape. Its default `max_error` tightens from `1e-10` to `1e-14`.

### Fixed

- `Envelope.__eq__` considered a longer envelope equal to a shorter one that matched it as
  a prefix.
- `split_at` could manufacture near-zero-width sliver segments. It bypassed
  `insert_interpolated`'s coincident-point guard in order to split exactly where asked, so a
  split landing a floating-point hair away from an existing control point — as happens when
  the split coordinate is reconstructed by subtraction — inserted a degenerate segment and
  read the curve slightly off. It now honors the tolerance.

## Earlier versions

For changes prior to 0.8.0, see the commit history.
