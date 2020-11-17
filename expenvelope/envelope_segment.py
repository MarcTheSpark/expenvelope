r"""
Module containing the :class:`EnvelopeSegment` class. Every :class:`envelope.Envelope` object, under the hood, is made
up of a list of :class:`EnvelopeSegment`\ s. :class:`EnvelopeSegment`\ s support arithmetic operations like addition,
subtraction, multiplication, and division. They also support horizontal scaling and shifting.

Note also that this is defining a function whose domain is a portion of the real number line, but whose range can
actually be nearly anything, including, e.g., numpy arrays. All of the functionality, even integration, works for
mappings onto other kinds of ranges.
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

from ._utilities import _make_envelope_segments_from_function, _curve_shape_from_start_mid_and_end_levels, \
    _get_curvature_from_filled_amount
import numbers
import math
from typing import Union, Tuple, TypeVar


T = TypeVar('T', bound='EnvelopeSegment')


class EnvelopeSegment:

    """
    A segment of an envelope, with the ability to perform interpolation and integration.

    :param start_time: the start time of the segment (where it is in the parent :class:`Envelope`)
    :param end_time: the end time of the segment
    :param start_level: the level of the envelope segment at the beginning. Note that a type has not been specified;
        the only requirements are that the type should be consistent and respond to addition and multiplication.
        This means, for instance, that numpy arrays could be used.
    :param end_level: the level of the envelope segment at the end.
    :param curve_shape: 0 is linear, > 0 changes late, < 0 changes early. Also, string expressions involving "exp"
        can be given, where "exp" stands for the shape that will produce constant proportional change per unit time.
    :ivar start_time: the start time of the segment
    :ivar end_time: the end time of the segment
    """

    def __init__(self, start_time: float, end_time: float, start_level, end_level, curve_shape: Union[float, str]):
        # note that start_level, end_level, and curvature are properties, since we want
        # to recalculate the constants that we use internally if they are changed.
        self.start_time = start_time
        self.end_time = end_time
        self._start_level = start_level
        self._end_level = end_level
        if isinstance(curve_shape, str):
            assert end_level / start_level > 0, \
                "Exponential interpolation is impossible between {} and {}".format(start_level, end_level)
            exp_shape = math.log(end_level / start_level)
            # noinspection PyBroadException
            try:
                self._curve_shape = eval(curve_shape, {"exp": exp_shape})
            except Exception:
                raise ValueError("Expression for curve shape not understood")
        else:
            self._curve_shape = curve_shape
        # we avoid calculating the constants until necessary for the calculations so that this
        # class is lightweight and can freely be created and discarded by an Envelope object

        self._A = self._B = None

    @classmethod
    def from_endpoints_and_halfway_level(cls, start_time: float, end_time: float,
                                         start_level, end_level, halfway_level) -> T:
        """
        Construct an EnvelopeSegment with the given start/end times/levels, specifying curve shape indirectly through
        the desired halfway level.

        :param start_time: the start time of the segment (where it is in the parent :class:`Envelope`)
        :param end_time: the end time of the segment
        :param start_level: the level of the envelope segment at the beginning. (See documentation for
            :class:`EnvelopeSegment`)
        :param end_level: the level of the envelope segment at the end.
        :param halfway_level: The level we want to reach halfway through the segment.
        """
        curve_shape = _curve_shape_from_start_mid_and_end_levels(start_level, halfway_level, end_level)
        return cls(start_time, end_time, start_level, end_level, curve_shape)

    def _calculate_coefficients(self):
        # A and _B are constants used in integration, and it's more efficient to just calculate them once.
        if abs(self._curve_shape) < 0.000001:
            # the curve shape is essentially zero, so set the constants to none as a flag to use linear interpolation
            self._A = self._B = None
            return
        else:
            self._A = (self._start_level - (self._end_level - self._start_level) / (math.exp(self._curve_shape) - 1))
            self._B = (self._end_level - self._start_level) / (self._curve_shape * (math.exp(self._curve_shape) - 1))

    @property
    def start_level(self):
        """
        The start level of this segment. (Can take a wide variety of types; see documentation for
        :class:`EnvelopeSegment`)
        """
        return self._start_level

    @start_level.setter
    def start_level(self, start_level):
        self._start_level = start_level
        self._calculate_coefficients()

    @property
    def end_level(self):
        """
        The end level of this segment. (Can take a wide variety of types; see documentation for
        :class:`EnvelopeSegment`)
        """
        return self._end_level

    @end_level.setter
    def end_level(self, end_level):
        self._end_level = end_level
        self._calculate_coefficients()

    @property
    def curve_shape(self) -> float:
        """
        The curve shape of this segment. (See documentation for :class:`EnvelopeSegment`)
        """
        return self._curve_shape

    @curve_shape.setter
    def curve_shape(self, curve_shape):
        self._curve_shape = curve_shape
        self._calculate_coefficients()

    @property
    def duration(self) -> float:
        """
        Duration of this segment.
        """
        return self.end_time - self.start_time

    def max_level(self):
        """
        Get the maximum level achieved in this segment
        """
        return max(self.start_level, self.end_level)

    def average_level(self):
        """
        Get the average level achieved in this segment
        """
        return self.integrate_segment(self.start_time, self.end_time) / self.duration

    def max_absolute_slope(self):
        """
        Get the max absolute value of the slope of this segment over the interval.
        (Since the slope of e^x is e^x, the max slope of e^x in the interval of [0, S] is e^S. If S is negative, the
        curve has exactly the same slopes, but in reverse (still need to think about why), so that's why the max slope
        term ends up being e^abs(S). We then have to scale that by the average slope over our interval divided by
        the average slope of e^x over [0, S] to get the true, scaled average slope. Hence the other scaling terms.)
        """
        if self.duration == 0:
            # a duration of zero means we have an immediate change of value. Since this function is used primarily
            # to figure out the temporal resolution needed for smoothness, that doesn't matter; it's supposed to be
            # a discontinuity. So we just return zero as a throwaway.
            return 0
        if abs(self._curve_shape) < 0.000001:
            # it's essentially linear, so just return the average slope
            return abs(self._end_level - self._start_level) / self.duration
        return math.exp(abs(self._curve_shape)) * abs(self._end_level - self._start_level) / self.duration * \
               abs(self._curve_shape) / (math.exp(abs(self._curve_shape)) - 1)

    def value_at(self, t: float, clip_at_boundary: bool = True):
        """
        Get interpolated value of the curve at time t.
        The equation here is y(t) = y1 + (y2 - y1) / (e^S - 1) * (e^(S*t) - 1)
        (y1=starting rate, y2=final rate, t=progress along the curve 0 to 1, S=curve_shape)
        Essentially it's an appropriately scaled and stretched segment of e^x with x in the range [0, S]
        as S approaches zero, we get a linear segment, and S of ln(y2/y1) represents normal exponential interpolation
        large values of S correspond to last-minute change, and negative values of S represent early change.

        :param t: time at which to evaluate the level (relative to the time zero, not to the start time of this segment)
        :param clip_at_boundary: if True, any t outside the boundary gets evaluated based on the start or end level
            (whichever is applicable.
        """
        if self._A is None:
            self._calculate_coefficients()

        if clip_at_boundary and t >= self.end_time:
            return self._end_level
        elif clip_at_boundary and t <= self.start_time:
            return self._start_level
        else:
            norm_t = (t - self.start_time) / (self.end_time - self.start_time)
        if abs(self._curve_shape) < 0.000001:
            # S is or is essentially zero, so this segment is linear. That limiting case breaks
            # our standard formula, but is easy to simply interpolate
            return self._start_level + norm_t * (self._end_level - self._start_level)

        return self._start_level + (self._end_level - self._start_level) / \
               (math.exp(self._curve_shape) - 1) * (math.exp(self._curve_shape * norm_t) - 1)

    def _segment_antiderivative(self, normalized_t):
        # the antiderivative of the interpolation curve y(t) = y1 + (y2 - y1) / (e^S - 1) * (e^(S*t) - 1)
        return self._A * normalized_t + self._B * math.exp(self._curve_shape * normalized_t)

    def integrate_segment(self, t1, t2):
        """
        Integrate part of this segment.

        :param t1: start time (relative to the time zero, not to the start time of this segment)
        :param t2: end time (ditto)
        """
        assert self.start_time <= t1 <= self.end_time and self.start_time <= t2 <= self.end_time, \
            "Integration bounds must be within curve segment bounds."
        if t1 == t2:
            return 0
        if self._A is None:
            self._calculate_coefficients()

        norm_t1 = (t1 - self.start_time) / (self.end_time - self.start_time)
        norm_t2 = (t2 - self.start_time) / (self.end_time - self.start_time)

        if abs(self._curve_shape) < 0.000001:
            # S is or is essentially zero, so this segment is linear. That limiting case breaks
            # our standard formula, but is easy to simple calculate based on average level
            start_level = (1 - norm_t1) * self.start_level + norm_t1 * self.end_level
            end_level = (1 - norm_t2) * self.start_level + norm_t2 * self.end_level
            return (t2 - t1) * (start_level + end_level) / 2

        segment_length = self.end_time - self.start_time

        return segment_length * (self._segment_antiderivative(norm_t2) - self._segment_antiderivative(norm_t1))

    def get_integral_range(self):
        """
        Returns the range of possible values for the integral of this segment available by tweaking curvature

        :return a tuple of (low, high)
        """
        return self.duration * min(self.start_level, self.end_level), \
               self.duration * max(self.start_level, self.end_level)

    def set_curvature_to_desired_integral(self, desired_integral) -> None:
        """
        Changes the curvature of this segment so as to hit a desired target for the integral of the segment

        :param desired_integral: target value of the segment integral
        """
        low, high = self.get_integral_range()
        if not low < desired_integral < high:
            raise ValueError("Desired integral out of adjustable range.")
        if self.end_level > self.start_level:
            self._curve_shape = _get_curvature_from_filled_amount((desired_integral - low) / (high - low))
        else:
            self._curve_shape = _get_curvature_from_filled_amount(1 - (desired_integral - low) / (high - low))
        self._calculate_coefficients()

    def split_at(self, t: float) -> Tuple[T, T]:
        """
        Split this segment into two EnvelopeSegment's without altering the curve shape and return them.
        This segment is altered in the process.

        :param t: where to split it (t is absolute time)
        :return: a tuple of this segment modified to be only the first part, and a new segment for the second part
        """
        assert self.start_time < t < self.end_time
        middle_level = self.value_at(t)
        # since the curve shape represents how much of the curve e^x we go through, you simply split proportionally
        curve_shape_1 = (t - self.start_time) / (self.end_time - self.start_time) * self.curve_shape
        curve_shape_2 = self.curve_shape - curve_shape_1
        new_segment = EnvelopeSegment(t, self.end_time, middle_level, self.end_level, curve_shape_2)

        self.end_time = t
        self._end_level = middle_level
        self._curve_shape = curve_shape_1
        self._calculate_coefficients()
        return self, new_segment

    def clone(self) -> T:
        """
        Make a duplicate of this segment.
        """
        return EnvelopeSegment(self.start_time, self.end_time, self.start_level, self.end_level, self.curve_shape)

    def shift_vertical(self, amount) -> T:
        """
        Shifts the output of this segment by the specified amount.

        :param amount: the amount to shift up and down by
        :return: self, for chaining purposes
        """
        self._start_level += amount
        self._end_level += amount
        self._calculate_coefficients()
        return self

    def scale_vertical(self, amount) -> T:
        """
        Scales the output of this segment by the specified amount.

        :param amount: amount to scale output by
        :return: self, for chaining purposes
        """
        self._start_level *= amount
        self._end_level *= amount
        self._calculate_coefficients()
        return self

    def shift_horizontal(self, amount: float) -> T:
        """
        Shifts the domain of this segment by the specified amount.

        :param amount: the amount to shift the domain by
        :return: self, for chaining purposes
        """
        assert isinstance(amount, numbers.Number)
        self.start_time += amount
        self.end_time += amount
        return self

    def scale_horizontal(self, amount: float) -> T:
        """
        Scales the domain of this segment by the specified amount.

        :param amount: amount to scale domain by
        :return: self, for chaining purposes
        """
        self.start_time *= amount
        self.end_time *= amount
        return self

    def is_shifted_version_of(self, other: T, tolerance: float = 1e-10) -> bool:
        """
        Determines if this segment is simply a shifted version of another segment

        :param other: another EnvelopeSegment
        :param tolerance: how close it needs to be to count as the same
        """
        return abs(self.start_time - other.start_time) < tolerance and \
               abs(self.end_time - other.end_time) < tolerance and \
               ((self._start_level - other._start_level) - (self._end_level - other._end_level)) < tolerance and \
               (self._curve_shape - other._curve_shape) < tolerance

    def _get_graphable_point_pairs(self, resolution: int = 25, endpoint: bool = True):
        x_values = [self.start_time + x / resolution * self.duration
                    for x in range(resolution + 1 if endpoint else resolution)]
        y_values = [self.value_at(x) for x in x_values]
        return x_values, y_values

    def show_plot(self, title: str = None, resolution: int = 25) -> None:
        """
        Uses matplotlib to display a graph of this EnvelopeSegment.

        :param title: (optional) the title to give the graph
        :param resolution: how many points to use in creating the graph
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Could not find matplotlib, which is needed for plotting.")
        fig, ax = plt.subplots()
        ax.plot(*self._get_graphable_point_pairs(resolution))
        ax.set_title('Graph of Envelope Segment' if title is None else title)
        plt.show()

    def _reciprocal(self):
        assert self.start_level * self.end_level > 0, "Cannot divide by EnvelopeSegment that crosses zero"
        return self.from_endpoints_and_halfway_level(self.start_time, self.end_time,
                                                     1 / self.start_level, 1 / self.end_level,
                                                     1 / self.value_at((self.start_time + self.end_time) / 2))

    def __eq__(self, other):
        return self.start_time == other.start_time and self.end_time == other.end_time \
               and self._start_level == other._start_level and self._end_level == other._end_level \
               and self._curve_shape == other._curve_shape

    def __neg__(self):
        return EnvelopeSegment(self.start_time, self.end_time,
                               -self.start_level, -self.end_level, self.curve_shape)

    def __add__(self, other):
        from .envelope import Envelope
        if isinstance(other, numbers.Number):
            return EnvelopeSegment(self.start_time, self.end_time, self._start_level + other,
                                   self._end_level + other, self._curve_shape)
        elif isinstance(other, EnvelopeSegment):
            if self.start_time == other.start_time and self.end_time == other.end_time:
                segments = _make_envelope_segments_from_function(lambda t: self.value_at(t) + other.value_at(t),
                                                                 self.start_time, self.end_time)
                if len(segments) == 1:
                    return segments[0]
                else:
                    return Envelope.from_segments(segments)
            else:
                raise ValueError("EnvelopeSegments can only be added if they have the same time range.")
        else:
            raise TypeError("Can only add EnvelopeSegment to a constant or another EnvelopeSegment")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__radd__(-other)

    def __mul__(self, other):
        from .envelope import Envelope
        if isinstance(other, numbers.Number):
            out = self.clone()
            out.scale_vertical(other)
            return out
        elif isinstance(other, EnvelopeSegment):
            if self.start_time == other.start_time and self.end_time == other.end_time:
                segments = _make_envelope_segments_from_function(lambda t: self.value_at(t) * other.value_at(t),
                                                                 self.start_time, self.end_time)
                if len(segments) == 1:
                    return segments[0]
                else:
                    return Envelope.from_segments(segments)
            else:
                raise ValueError("EnvelopeSegments can only be added if they have the same time range.")
        else:
            raise TypeError("Can only multiply EnvelopeSegment with a constant or another EnvelopeSegment")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return self._reciprocal() * other

    def __contains__(self, t):
        # checks if the given time is contained within this envelope segment
        # maybe this is silly, but it seemed a little convenient
        return self.start_time <= t < self.end_time \
               or t == self.start_time  # in case start_time == end_time, we still want it to count

    def __repr__(self):
        return "EnvelopeSegment({}, {}, {}, {}, {})".format(self.start_time, self.end_time, self.start_level,
                                                            self.end_level, self.curve_shape)
