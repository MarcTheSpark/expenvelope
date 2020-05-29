"""
Expenvelope is a library for expressing piecewise exponential curves, with musical applications in mind.
(This is modeled in some ways on the `Env` class in SuperCollider.) Contents of this package include the `envelope`
module, which defines the central :class:`~expenvelope.envelope.Envelope` class, and the `envelope_segment` module,
which defines the :class:`~expenvelope.envelope_segment.EnvelopeSegment` class, out of which an
:class:`~expenvelope.envelope.Envelope` is built.
"""

from .envelope import Envelope
from .envelope_segment import EnvelopeSegment
