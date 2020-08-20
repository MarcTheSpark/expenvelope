"""
Expenvelope is a library for expressing piecewise exponential curves, with musical applications in mind.
(This is modeled in some ways on the `Env` class in SuperCollider.) Contents of this package include the `envelope`
module, which defines the central :class:`~expenvelope.envelope.Envelope` class, and the `envelope_segment` module,
which defines the :class:`~expenvelope.envelope_segment.EnvelopeSegment` class, out of which an
:class:`~expenvelope.envelope.Envelope` is built.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  This file is part of SCAMP (Suite for Computer-Assisted Music in Python)                      #
#  Copyright © 2020 Marc Evanstein <marc@marcevanstein.com>.                                     #
#                                                                                                #
#  This program is free software: you can redistribute it and/or modify it under the terms of    #
#  the GNU General Public License as published by the Free Software Foundation, either version   #
#  3 of the License, or (at your option) any later version.                                      #
#                                                                                                #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#  See the GNU General Public License for more details.                                          #
#                                                                                                #
#  You should have received a copy of the GNU General Public License along with this program.    #
#  If not, see <http://www.gnu.org/licenses/>.                                                   #
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #

from .envelope import Envelope
from .envelope_segment import EnvelopeSegment
