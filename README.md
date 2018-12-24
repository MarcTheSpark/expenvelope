# expenvelope

_expenvelope_ is a python library for managing piecewise exponential curves, original intended as a tool for algorithmic music composition. Curves are simple to make, expressive, and useful for controlling dynamics, tempo, and other higher-level parameters. 

The central `Envelope` class bears some relation to SuperCollider's [_Env_](http://doc.sccode.org/Classes/Env.html) object, and is represented behind the scenes as a contiguous set of `EnvelopeSegments`. There are a number of different class methods available for constructing envelopes, including:

```python
Envelope.from_levels
Envelope.from_levels_and_durations
Envelope.from_points
Envelope.release
Envelope.ar
Envelope.asr
Envelope.adsr
Envelope.from_function

```

In addition to the central `value_at` function, utilities have been included to append and insert new points, insert a new interpolated control point without changing the curve, integrate over intervals, find the maximum slope reached, and find the average value, among other things. Envelopes (and EnvelopeSegments) can be added, subtracted, multiplied and divided, with these operations yielding new Envelopes that are close approximations to the resulting function using piecewise exponential curves.

_expenvelope_ is a key dependency of [clockblocks](https://github.com/MarcTheSpark/clockblocks), a package for for controlling the flow of musical time, and [scamp](https://github.com/MarcTheSpark/scamp/), a Suite for Composing Algorithmic Music in Python.
