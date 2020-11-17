r"""
Module containing the central :class:`Envelope` class, which represents a piece-wise exponential function.
:class:`Envelope`\ s support arithmetic operations like addition, subtraction, multiplication, and division. They
also support horizontal scaling and shifting.

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
from itertools import zip_longest

from ._utilities import _make_envelope_segments_from_function, _curve_shape_from_start_mid_and_end_levels
from .json_serializer import SavesToJSON
from .envelope_segment import EnvelopeSegment
import numbers
from typing import Sequence, Union, Callable, Tuple, TypeVar


T = TypeVar('T', bound='Envelope')


class Envelope(SavesToJSON):

    r"""
    Class representing a piece-wise exponential function. This class was designed with musical applications in mind,
    with the intention being to represent a continuously changing musical parameter over time. For instance, an
    Envelope might be used to represent the pitch curve of a glissando, or the volume curve of a forte-piano.

    :param levels: the levels of the curve. These can be anything that acts like a number. For instance, one could
        even use numpy arrays as levels.
    :param durations: the durations of the curve segments. (Should have length one less than levels.)
    :param curve_shapes: the curve shape values (optional, should have length one less than levels). Generally these
        will be floats, with the default of 0 representing linear change, > 0 representing late change, and < 0
        representing early change. It is also possible to use the string "exp" to produce constant proportional
        change per unit time, so long as the segment does not touch zero. Finally, strings containing "exp", such
        as "exp ** 2 / 5" will be evaluated with the curve shape required for exponential change being plugged
        in for the variable "exp".
    :param offset: starts curve from somewhere other than zero
    :ivar segments: list of :class:`~expenvelope.envelope_segment.EnvelopeSegment`\ s representing the pieces of this
        envelope.
    """

    def __init__(self, levels: Sequence = (0,), durations: Sequence[float] = (),
                 curve_shapes: Sequence[Union[float, str]] = None, offset: float = 0):
        if not hasattr(levels, "__len__"):
            levels = (levels,)
        try:
            assert hasattr(levels, "__len__") and hasattr(durations, "__len__") \
                   and (curve_shapes is None or hasattr(curve_shapes, "__len__"))
            assert len(levels) > 0 and len(durations) == len(levels) - 1
            if len(levels) == 1:
                levels = levels + levels
                durations = (0, )
            if curve_shapes is None:
                curve_shapes = [0] * (len(levels) - 1)
        except AssertionError:
            raise ValueError("Bad arguments for envelope construction; there must be one fewer durations (and "
                             "curve shapes, if not None) than levels.")
        self.segments = Envelope._construct_segments_list(levels, durations, curve_shapes, offset)

    @staticmethod
    def _construct_segments_list(levels: Sequence = (0, 0), durations: Sequence[float] = (0,),
                                 curve_shapes: Sequence[Union[float, str]] = None, offset: float = 0):
        segments = []
        t = offset
        for i in range(len(levels) - 1):
            segments.append(EnvelopeSegment(t, t + durations[i], levels[i], levels[i + 1], curve_shapes[i]))
            t += durations[i]
        return segments

    # ---------------------------- Class methods --------------------------------

    @classmethod
    def from_segments(cls, segments: Sequence[EnvelopeSegment]) -> T:
        """
        Create a new envelope from a list of :class:`~expenvelope.envelope_segment.EnvelopeSegment`\ s.

        :param segments: list of segments
        """
        out = cls()
        assert all(isinstance(x, EnvelopeSegment) for x in segments)
        out.segments = segments
        return out

    @classmethod
    def from_levels_and_durations(cls, levels: Sequence, durations: Sequence[float],
                                  curve_shapes: Sequence[Union[float, str]] = None, offset: float = 0) -> T:
        """
        Construct an Envelope from levels, durations, and optionally curve shapes.

        :param levels: the levels of the curve. These can be anything that acts like a number. For instance, one could
            even use numpy arrays as levels.
        :param durations: the durations of the curve segments. (Should have length one less than levels.)
        :param curve_shapes: the curve shape values (optional, should have length one less than levels). Generally these
            will be floats, with the default of 0 representing linear change, > 0 representing late change, and < 0
            representing early change. It is also possible to use the string "exp" to produce constant proportional
            change per unit time, so long as the segment does not touch zero. Finally, strings containing "exp", such
            as "exp ** 2 / 5" will be evaluated with the curve shape required for exponential change being plugged
            in for the variable "exp".
        :param offset: starts curve from somewhere other than zero
        :return: an Envelope constructed accordingly
        """
        return cls(levels, durations, curve_shapes, offset)

    @classmethod
    def from_levels(cls, levels: Sequence, length: float = 1.0, offset: float = 0) -> T:
        """
        Construct an envelope from levels alone, normalized to the given length.
        
        :param levels: the levels of the curve. These can be anything that acts like a number. For instance, one could
            even use numpy arrays as levels.
        :param length: the total length of the curve, divided evenly amongst the levels
        :param offset: starts curve from somewhere other than zero
        :return: an Envelope constructed accordingly
        """
        return cls.from_levels_and_durations(
            *Envelope._levels_and_length_to_levels_durations_and_curves(levels, length), offset
        )

    @staticmethod
    def _levels_and_length_to_levels_durations_and_curves(levels: Sequence, length: float):
        if not len(levels) > 0:
            raise ValueError("At least one level is needed to construct an envelope.")
        if len(levels) == 1:
            levels = list(levels) * 2
        # just given levels, so we linearly interpolate segments of equal length
        durations = [length / (len(levels) - 1)] * (len(levels) - 1)
        curves = [0.0] * (len(levels) - 1)
        return levels, durations, curves

    @classmethod
    def from_list(cls, constructor_list: Sequence) -> T:
        """
        Construct an envelope from a list that can take a number of formats
        
        :param constructor_list: Either a flat list that just contains levels, or a list of lists either of the form
            [levels_list, total_duration], [levels_list, durations_list] or [levels_list, durations_list,
            curve_shape_list] for example:
            - an input of [1, 0.5, 0.3] is interpreted as evenly spaced levels with a total duration of 1
            - an input of [[1, 0.5, 0.3], 3.0] is interpreted as levels and durations with a total duration of e.g. 3.0
            - an input of [[1, 0.5, 0.3], [0.2, 0.8]] is interpreted as levels and durations
            - an input of [[1, 0.5, 0.3], [0.2, 0.8], [2, 0.5]] is interpreted as levels, durations, and curvatures
        :return: an Envelope constructed accordingly
        """
        assert hasattr(constructor_list, "__len__")
        if hasattr(constructor_list[0], "__len__"):
            # we were given levels and durations, and possibly curvature values
            if len(constructor_list) == 2:
                if hasattr(constructor_list[1], "__len__"):
                    # given levels and durations
                    return cls.from_levels_and_durations(constructor_list[0], constructor_list[1])
                else:
                    # given levels and the total length
                    return cls.from_levels(constructor_list[0], length=constructor_list[1])

            elif len(constructor_list) >= 3:
                # given levels, durations, and curvature values
                return cls.from_levels_and_durations(constructor_list[0], constructor_list[1], constructor_list[2])
        else:
            # just given levels
            return cls.from_levels(constructor_list)

    @classmethod
    def from_points(cls, *points: Sequence) -> T:
        """
        Construct an envelope from points
        
        :param points: list of points, each of which is of the form (time, value) or (time, value, curve_shape)
        :return: an Envelope constructed accordingly
        """
        return cls(*Envelope._unwrap_points(*points))

    @staticmethod
    def _unwrap_points(*points: Sequence):
        assert all(len(point) >= 2 for point in points)
        points = tuple(sorted(points, key=lambda point: point[0]))
        times, levels, *extra = tuple(zip_longest(*points, fillvalue=0))
        offset = times[0]
        durations = tuple(b - a for a, b in zip(times[:-1], times[1:]))
        if len(extra) > 0:
            curve_shapes = extra[0][:-1] if len(extra[0]) == len(times) else extra[0]
        else:
            curve_shapes = None
        return levels, durations, curve_shapes, offset

    @classmethod
    def release(cls, duration: float, start_level=1, curve_shape: Union[float, str] = None) -> T:
        """
        Construct an simple decaying envelope
        
        :param duration: total decay length
        :param start_level: level decayed from
        :param curve_shape: shape of the curve (see documentation for :func:`Envelope.from_levels_and_durations`)
        :return: an Envelope constructed accordingly
        """
        curve_shapes = (curve_shape,) if curve_shape is not None else None
        return cls.from_levels_and_durations((start_level, 0), (duration,), curve_shapes=curve_shapes)

    @classmethod
    def ar(cls, attack_length: float, release_length: float, peak_level=1,
            attack_shape: Union[float, str] = None, release_shape: Union[float, str] = None) -> T:
        """
        Construct an attack/release envelope

        :param attack_length: rise time
        :param release_length: release time
        :param peak_level: level reached after attack and before release (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :param attack_shape: sets curve shape for attack portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :param release_shape: sets curve shape for release portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :return: an Envelope constructed accordingly
       """
        curve_shapes = None if attack_shape is release_shape is None else \
            (0 if attack_shape is None else attack_shape, 0 if release_shape is None else release_shape)
        return cls.from_levels_and_durations((0, peak_level, 0), (attack_length, release_length),
                                             curve_shapes=curve_shapes)

    @classmethod
    def asr(cls, attack_length: float, sustain_level, sustain_length: float, release_length: float,
            attack_shape: Union[float, str] = None, release_shape: Union[float, str] = None) -> T:
        """
        Construct an attack/sustain/release envelope

        :param attack_length: rise time
        :param sustain_level: sustain level reached after attack and before release
        :param sustain_length: length of sustain portion of curve
        :param release_length: release time
        :param attack_shape: sets curve shape for attack portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :param release_shape: sets curve shape for release portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :return: an Envelope constructed accordingly
       """
        curve_shapes = None if attack_shape is release_shape is None else \
            (0 if attack_shape is None else attack_shape, 0, 0 if release_shape is None else release_shape)
        return cls.from_levels_and_durations((0, sustain_level, sustain_level, 0),
                                             (attack_length, sustain_length, release_length),
                                             curve_shapes=curve_shapes)

    @classmethod
    def adsr(cls, attack_length: float, attack_level, decay_length: float, sustain_level, sustain_length: float,
             release_length: float, attack_shape: Union[float, str] = None, decay_shape: Union[float, str] = None,
             release_shape: Union[float, str] = None) -> T:
        """
        Construct a standard attack/decay/sustain/release envelope

        :param attack_length: rise time
        :param attack_level: level reached after attack before decay
        :param decay_length: length of decay portion of the curve
        :param sustain_level: sustain level reached after decay and before release
        :param sustain_length: length of sustain portion of curve
        :param release_length: release time
        :param attack_shape: sets curve shape for attack portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :param decay_shape: sets curve shape for decay portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :param release_shape: sets curve shape for release portion of the curve (see documentation for
            :func:`Envelope.from_levels_and_durations`)
        :return: an Envelope constructed accordingly
       """
        curve_shapes = None if attack_shape is decay_shape is release_shape is None else \
            (0 if attack_shape is None else attack_shape, 0 if decay_shape is None else decay_shape,
             0, 0 if release_shape is None else release_shape)
        return cls.from_levels_and_durations((0, attack_level, sustain_level, sustain_level, 0),
                                             (attack_length, decay_length, sustain_length, release_length),
                                             curve_shapes=curve_shapes)

    @classmethod
    def from_function(cls, function: Callable[[float], float], domain_start: float = 0, domain_end: float = 1,
                      resolution_multiple: int = 2, key_point_precision: int = 2000,
                      key_point_iterations: int = 5) -> T:
        """
        Constructs an Envelope that approximates an arbitrary function. By default, the function is split at local
        extrema and inflection points found through a pretty unsophisticated numerical process.

        :param function: a function from time to level (often a lambda function)
        :param domain_start: the beginning of the function domain range to capture
        :param domain_end: the end of the function domain range to capture
        :param resolution_multiple: factor by which we add extra key points between the extrema and inflection points
            to improve the curve fit..
        :param key_point_precision: precision with which we break up the domain of the function in searching for key
            points (how many discrete differences we use).
        :param key_point_iterations: every time a prospective key point is found, we run a more narrow search on that
            segment of the domain, using smaller step sizes, which gives us a more precise location for the key point.
            We then repeat this narrowing in process, up to this many iterations
        :return: an Envelope constructed accordingly
        """
        return cls.from_segments(_make_envelope_segments_from_function(
            function, domain_start, domain_end, resolution_multiple, key_point_precision, key_point_iterations))

    # ---------------------------- Various Properties --------------------------------

    def length(self) -> float:
        """
        The length of the domain on which this Envelope is defined (end time minus start time).
        """
        if len(self.segments) == 0:
            return 0
        return self.segments[-1].end_time - self.segments[0].start_time

    def start_time(self) -> float:
        """
        Beginning of the domain on which this Envelope is defined.
        """
        return self.offset

    def end_time(self) -> float:
        """
        End of the domain on which this Envelope is defined.
        """
        return self.segments[-1].end_time

    def start_level(self):
        """
        Beginning value of the Envelope
        """
        return self.segments[0].start_level

    def end_level(self):
        """
        Ending value of the Envelope
        """
        return self.segments[-1].end_level

    def max_level(self, t_range: Tuple[float, float] = None):
        """
        Returns the highest value that the Envelope takes over the given range.

        :param t_range: tuple defining the start and end time of the interval to check. If None, return the max level
            reached over the entire Envelope.
        """
        if t_range is None:
            # checking over the entire range, so that's easy
            return max(segment.max_level() for segment in self.segments)
        else:
            # checking over the range (t1, t2), so look at the values at those endpoints and any anchor points between
            assert hasattr(t_range, "__len__") and len(t_range) == 2 and t_range[0] < t_range[1]
            t1, t2 = t_range
            points_to_check = [self.value_at(t1), self.value_at(t2)]

            for segment in self.segments[self._get_index_of_segment_at(t1, left_most=True):]:
                if t1 <= segment.start_time <= t2:
                    points_to_check.append(segment.start_level)
                if t1 <= segment.end_time <= t2:
                    points_to_check.append(segment.end_level)
                if segment.end_time > t2:
                    break
            return max(points_to_check)

    def average_level(self, t_range: Tuple[float, float] = None):
        """
        Returns the average value that the Envelope takes over the given range.

        :param t_range: tuple defining the start and end time of the interval to check. If None, return the average
            level reached over the entire Envelope.
        """
        if t_range is None:
            t_range = self.start_time(), self.start_time() + self.length()
            interval_length = self.length()
        else:
            interval_length = t_range[1] - t_range[0]
        return self.integrate_interval(*t_range) / interval_length

    def max_absolute_slope(self):
        """
        Returns the maximum absolute value of the slope over the entire Envelope.
        """
        return max(segment.max_absolute_slope() for segment in self.segments)

    @property
    def levels(self) -> Sequence:
        """
        Tuple of levels at all segment boundary points.
        """
        return tuple(segment.start_level for segment in self.segments) + (self.end_level(),)

    @property
    def durations(self) -> Sequence[float]:
        """
        Tuple of all the segment lengths.
        """
        return tuple(segment.duration for segment in self.segments)

    @property
    def times(self) -> Sequence[float]:
        """
        Tuple of all the segment start times.
        """
        return tuple(segment.start_time for segment in self.segments) + (self.end_time(),)

    @property
    def curve_shapes(self) -> Sequence[Union[float, str]]:
        """
        Tuple of all the segment curve shapes.
        """
        return tuple(segment.curve_shape for segment in self.segments)

    @property
    def offset(self) -> float:
        """
        Alias for :func:`Envelope.start_time`.
        """
        return self.segments[0].start_time

    # ----------------------- Insertion of new control points --------------------------
    
    def _get_index_of_segment_at(self, t, left_most=False, right_most=False):
        # first take care of the case that t is outside the envelope by just returning the first or last segment index
        # (also, might as well include the cases where we're inside those segments too)
        if t > self.segments[-1].start_time:
            return len(self.segments) - 1
        elif t < self.segments[0].end_time:
            return 0
        # if there are a lot of segments, we bisect the list repeatedly until we get close t
        lo_index = 0
        hi_index = len(self.segments)
        while True:
            test_index = (lo_index + hi_index) // 2
            this_segment = self.segments[test_index]
            if t in this_segment:
                # found it! except there's a wrinkle; since there can be zero-length segments, there might be more
                # than one that contains t. So left_most and right_most let us specify if we want the left or right one
                if left_most:
                    while test_index > 0 and self.segments[test_index - 1].end_time >= t:
                        test_index -= 1
                elif right_most:
                    while test_index < len(self.segments) - 1 and self.segments[test_index + 1].start_time <= t:
                        test_index += 1
                return test_index
            else:
                if lo_index == hi_index - 1:
                    # this segment doesn't work, but it's the only remaining option
                    # This shouldn't happen
                    raise IndexError("Can't find segment index; Envelope must be malformed.")
                if this_segment.start_time > t:
                    # test_index is too high, so don't look any higher
                    hi_index = test_index
                else:
                    # test index is too low, so don't look any lower
                    lo_index = test_index

    def insert(self, t, level, curve_shape_in=0, curve_shape_out=0) -> None:
        """
        Insert a curve point at time t, and set the shape of the curve into and out of it. This essentially divides
        the segment at that point in two.

        :param t: The time at which to add a point
        :param level: The level of the new point we are adding
        :param curve_shape_in: the curve shape of the new segment going into the point we are adding
        :param curve_shape_out: the curve shape of the new segment going out of the point we are adding
        """
        if t < self.start_time():
            self.prepend_segment(level, self.start_time() - t, curve_shape_out)
        if t > self.end_time():
            # adding a point after the curve
            self.append_segment(level, t - self.end_time(), curve_shape_in)
            return
        else:
            for i, segment in enumerate(self.segments):
                if segment.start_time < t < segment.end_time:
                    # we are inside an existing segment, so we break it in two
                    # save the old segment end time and level, since these will be the end of the second half
                    end_time = segment.end_time
                    end_level = segment.end_level
                    # change the first half to end at t and have the given shape
                    segment.end_time = t
                    segment.curve_shape = curve_shape_in
                    segment.end_level = level
                    new_segment = EnvelopeSegment(t, end_time, level, end_level, curve_shape_out)
                    self.segments.insert(i + 1, new_segment)
                    break
                else:
                    if t == segment.start_time:
                        # we are right on the dot of an existing segment, so we replace it
                        segment.start_level = level
                        segment.curve_shape = curve_shape_out
                    if t == segment.end_time:
                        segment.end_level = level
                        segment.curve_shape = curve_shape_in

    def insert_interpolated(self, t: float, min_difference: float = 1e-7) -> float:
        """
        Insert another curve point at the given time, without changing the shape of the curve. A point only gets added
        if it's at least min_difference from all existing control points.

        :param t: the point at which to insert the point
        :param min_difference: the minimum difference that this point has to be from an existing point on the curve
            in order for a new point to be added.
        :return: the t value at which we interpolated. If we try to insert within min_difference of an existing control
            point, then no new point is added, and we return the t of the nearest control point.
        """
        if t < self.start_time():
            # we set tolerance to -1 here to ensure that the initial segement doesn't simply get extended
            # we actually want an extra control point, redundant or not
            self.prepend_segment(self.start_level(), self.start_time() - t, tolerance=-1)
            return t
        if t > self.end_time():
            # tolerance set to -1 for same reason as above
            self.append_segment(self.end_level(), t - self.end_time(), tolerance=-1)
            return t
        if abs(t - self.start_time()) <= min_difference:
            return self.start_time()
        if abs(t - self.end_time()) <= min_difference:
            return self.end_time()
        for i, segment in enumerate(self.segments):
            if t in segment:
                # this is the case that matters; t is within one of the segments
                # make sure that we're further than min_difference from either endpoint
                if abs(t - segment.start_time) <= min_difference:
                    return segment.start_time
                if abs(t - segment.end_time) <= min_difference:
                    return segment.end_time
                # if not, then we split at this point
                part1, part2 = segment.split_at(t)
                self.segments.insert(i + 1, part2)
                return t

    # ----------------------- Appending / removing segments --------------------------

    def append_segment(self, level, duration: float, curve_shape: float = None, tolerance: float = 0,
                       halfway_level=None) -> None:
        """
        Append a segment to the end of the curve ending at level and lasting for duration.
        If we're adding a linear segment to a linear segment, then we extend the last linear segment
        instead of adding a new one if the level is within tolerance of where the last one was headed
        
        :param level: the level we're going to
        :param duration: the duration of the new segment
        :param curve_shape: defaults to 0 (linear)
        :param tolerance: tolerance for extending a linear segment rather than adding a new one
        :param halfway_level: alternate way of defining the curve shape. If this is set and the curve shape is
            not then we use this to determine the curve shape.
        """
        curve_shape = curve_shape if curve_shape is not None \
            else _curve_shape_from_start_mid_and_end_levels(self.end_level(), halfway_level, level) \
            if halfway_level is not None else 0
        if self.segments[-1].duration == 0:
            # the previous segment has no length. Are we also adding a segment with no length?
            if duration == 0:
                # If so, replace the end level of the existing zero-length segment
                self.segments[-1].end_level = level
            else:
                # okay, we're adding a segment with length
                # did the previous segment actually change the level?
                if self.segments[-1].end_level != self.segments[-1].start_level:
                    # If so we keep it and add a new one
                    self.segments.append(EnvelopeSegment(self.end_time(), self.end_time() + duration,
                                                         self.end_level(), level, curve_shape))
                else:
                    # if not, just modify the previous segment into what we want
                    self.segments[-1].end_level = level
                    self.segments[-1].end_time = self.end_time() + duration
                    self.segments[-1].curve_shape = curve_shape
        elif self.segments[-1].curve_shape == 0 and curve_shape == 0 and \
                abs(self.segments[-1].value_at(self.end_time() + duration,
                                               clip_at_boundary=False) - level) <= tolerance:
            # we're adding a point that would be a perfect continuation of the previous linear segment
            # (could do this for non-linear, but it's probably not worth the effort)
            self.segments[-1].end_time = self.length() + duration
            self.segments[-1].end_level = level
        else:
            self.segments.append(EnvelopeSegment(self.end_time(), self.end_time() + duration,
                                                 self.end_level(), level, curve_shape))

    def prepend_segment(self, level, duration: float, curve_shape: float = None, tolerance: float = 0,
                        halfway_level=None) -> None:
        """
        Prepend a segment to the beginning of the curve, starting at level and lasting for duration.
        If we're adding a linear segment to a linear segment, then we extend the last linear segment
        instead of adding a new one if the level is within tolerance of where the last one was headed

        :param level: the level that the prepended segment starts at
        :param duration: the duration of the new segment
        :param curve_shape: defaults to 0 (linear)
        :param tolerance: tolerance for extending a linear segment rather than adding a new one
        :param halfway_level: alternate way of defining the curve shape. If this is set and the curve shape is
            not then we use this to determine the curve shape.
        """
        curve_shape = curve_shape if curve_shape is not None \
            else _curve_shape_from_start_mid_and_end_levels(self.end_level(), halfway_level, level) \
            if halfway_level is not None else 0
        if self.segments[0].duration == 0:
            # the first segment has no length. Are we also prepending a segment with no length?
            if duration == 0:
                # If so, replace the start level of the existing zero-length segment
                self.segments[0].start_level = level
            else:
                # okay, we're adding a segment with length
                # does the first segment actually change the level?
                if self.segments[-1].end_level != self.segments[-1].start_level:
                    # If so we keep it and add a new one before it
                    self.segments.insert(0, EnvelopeSegment(self.start_time() - duration, self.start_time(),
                                                            level, self.start_level(), curve_shape))
                else:
                    # if not, just modify the previous segment into what we want
                    self.segments[0].start_level = level
                    self.segments[0].start_time = self.start_time() - duration
                    self.segments[0].curve_shape = curve_shape
        elif self.segments[0].curve_shape == 0 and curve_shape == 0 and \
                abs(self.segments[0].value_at(self.start_time() - duration,
                                              clip_at_boundary=False) - level) <= tolerance:
            # we're adding a point that would be a perfect extrapolation of the initial linear segment
            # (could do this for non-linear, but it's probably not worth the effort)
            self.segments[0].start_time = self.start_time() - duration
            self.segments[0].start_level = level
        else:
            self.segments.insert(0, EnvelopeSegment(self.start_time() - duration, self.start_time(),
                                                    level, self.start_level(), curve_shape))

    def pop_segment(self) -> Union[EnvelopeSegment, None]:
        """
        Remove and return the last segment of this Envelope.
        If there is only one segment, reduce it to length zero and return None.
        """
        if len(self.segments) == 1:
            if self.segments[0].end_time != self.segments[0].start_time or \
                    self.segments[0].end_level != self.segments[0].start_level:
                self.segments[0].end_time = self.segments[0].start_time
                self.segments[0].end_level = self.segments[0].start_level
                return
            else:
                raise IndexError("Cannot pop from empty Envelope")
        return self.segments.pop()

    def pop_segment_from_start(self) -> Union[EnvelopeSegment, None]:
        """
        Remove and return the first segment of this Envelope.
        If there is only one segment, reduce it to length zero and return None.
        """
        if len(self.segments) == 1:
            if self.segments[0].end_time != self.segments[0].start_time or \
                    self.segments[0].end_level != self.segments[0].start_level:
                self.segments[0].start_time = self.segments[0].end_time
                self.segments[0].start_level = self.segments[0].end_level
                return
            else:
                raise IndexError("Cannot pop from empty Envelope")
        return self.segments.pop(0)

    def remove_segments_after(self, t: float) -> None:
        """
        Removes all segments after the given time (including a partial segment if t lands in the middle of a segment).

        :param t: the point at which this Envelope is to be truncated.
        """
        if t < self.start_time():
            while True:
                try:
                    self.pop_segment()
                except IndexError:
                    break
        for segment in self.segments:
            if t == segment.start_time:
                while self.end_time() > t:
                    self.pop_segment()
                return
            elif segment.start_time < t < segment.end_time:
                self.insert_interpolated(t)
                while self.end_time() > t:
                    self.pop_segment()
                return

    def remove_segments_before(self, t: float) -> None:
        """
        Removes all segments before the given time (including a partial segment if t lands in the middle of a segment).

        :param t: the point at which this Envelope is to be truncated.
        """
        if t > self.end_time():
            while True:
                try:
                    self.pop_segment_from_start()
                except IndexError:
                    break
        for segment in reversed(self.segments):
            if t == segment.end_time:
                while self.start_time() < t:
                    self.pop_segment_from_start()
                return
            elif segment.start_time < t < segment.end_time:
                self.insert_interpolated(t)
                while self.start_time() < t:
                    self.pop_segment_from_start()
                return

    def append_envelope(self, envelope_to_append: T) -> __qualname__:
        """
        Extends this envelope by another one (shifted to start at the end of this one).
        """
        if self.end_level() != envelope_to_append.start_level():
            self.append_segment(envelope_to_append.start_level(), 0)
        for segment in envelope_to_append.segments:
            self.append_segment(segment.end_level, segment.duration, segment.curve_shape)
        return self

    def prepend_envelope(self, envelope_to_prepend: T) -> T:
        """
        Extends this envelope backwards by another one (shifted to end at the start of this one).
        """
        if self.start_level() != envelope_to_prepend.end_level():
            self.prepend_segment(envelope_to_prepend.end_level(), 0)
        for segment in reversed(envelope_to_prepend.segments):
            self.prepend_segment(segment.start_level, segment.duration, segment.curve_shape)
        return self

    # ------------------------ Interpolation, Integration --------------------------

    def value_at(self, t: float, from_left: bool = False):
        """
        Get the value of this Envelope at the given time.

        :param t: the time
        :param from_left: if true, get the limit as we approach t from the left. In the case of a zero-length segment,
            which suddenly changes the value, this tells us what the value was right before the jump instead of right
            after the jump.
        """
        if t < self.start_time():
            return self.start_level()

        containing_segment_index = self._get_index_of_segment_at(t, left_most=from_left, right_most=not from_left)

        return self.segments[containing_segment_index].value_at(t)

    def integrate_interval(self, t1: float, t2: float):
        """
        Get the definite integral under this Envelope from t1 to t2

        :param t1: lower bound of integration
        :param t2: upper bound of integration
        """
        if t1 == t2:
            return 0
        if t2 < t1:
            return -self.integrate_interval(t2, t1)
        if t1 < self.start_time():
            return (self.start_time() - t1) * self.segments[0].start_level + \
                   self.integrate_interval(self.start_time(), t2)
        if t2 > self.end_time():
            return (t2 - self.end_time()) * self.end_level() + self.integrate_interval(t1, self.end_time())
        # now that the edge conditions are covered, we just add up the segment integrals
        integral = 0

        for segment in self.segments[self._get_index_of_segment_at(t1):]:
            if t1 < segment.start_time:
                if t2 > segment.start_time:
                    if t2 <= segment.end_time:
                        # this segment contains the end of our integration interval, so we're done after this
                        integral += segment.integrate_segment(segment.start_time, t2)
                        break
                    else:
                        # this segment is fully within out integration interval, so add its full area
                        integral += segment.integrate_segment(segment.start_time, segment.end_time)
            elif t1 in segment:
                # since we know that t2 > t1, there's two possibilities
                if t2 in segment or t2 == segment.end_time:
                    # this segment contains our whole integration interval
                    integral += segment.integrate_segment(t1, t2)
                    break
                else:
                    # this is the first segment in our integration interval
                    integral += segment.integrate_segment(t1, segment.end_time)
        return integral

    def get_upper_integration_bound(self, t1: float, desired_area: float, max_error: float = 0.001) -> float:
        """
        Given a lower integration bound, find the upper bound that will result in the desired integral

        :param t1: lower bound of integration
        :param desired_area: desired value of the integral.
        :param max_error: the upper bound is found through a process of successive approximation; once we get within
            this error, the approximation is considered good enough.
        """
        if desired_area < max_error:
            return t1
        t1_level = self.value_at(t1)
        t2_guess = desired_area / t1_level + t1
        area = self.integrate_interval(t1, t2_guess)
        if area <= desired_area:
            if desired_area - area < max_error:
                # we hit it almost perfectly and didn't go over
                return t2_guess
            else:
                # we undershot, so start from where we left off.
                # Eventually we will get close enough that we're below the max_error
                return self.get_upper_integration_bound(t2_guess, desired_area - area, max_error=max_error)
        else:
            # we overshot, so back up to a point that we know must be below the upper integration bound
            conservative_guess = t1_level / self.max_level((t1, t2_guess)) * (t2_guess - t1) + t1
            return self.get_upper_integration_bound(
                conservative_guess, desired_area - self.integrate_interval(t1, conservative_guess), max_error=max_error
            )

    # -------------------------------- Utilities --------------------------------

    def normalize_to_duration(self, desired_duration: float, in_place: bool = True) -> T:
        """
        Stretch or squeeze the segments of this Envelope so that it has the desired total duration.

        :param desired_duration: the desired new duration of the Envelope
        :param in_place: if True, modifies this Envelope in place; if False, makes a copy first
        """
        out = self if in_place else self.duplicate()
        if self.length() != desired_duration:
            ratio = desired_duration / self.length()
            for segment in out.segments:
                segment.start_time = (segment.start_time - self.start_time()) * ratio + self.start_time()
                segment.end_time = (segment.end_time - self.start_time()) * ratio + self.start_time()
        return out

    def local_extrema(self, include_saddle_points: bool = False) -> Sequence[float]:
        """
        Returns a list of the times where the curve changes direction.

        :param include_saddle_points: if True, also include points where the curve starts to plateau
        """
        local_extrema = []
        last_direction = 0
        for segment in self.segments:
            if segment.end_level > segment.start_level:
                direction = 1
            elif segment.end_level < segment.start_level:
                direction = -1
            else:
                # if this segment was static, then keep the direction we had going in
                direction = last_direction
                # if we want to include saddle points, then check that the last segment was not also flat
                # (if there's a series of flat segments in a row, only count the first one as a saddle point)
                if include_saddle_points and last_direction != 0 and segment.start_time not in local_extrema:
                    local_extrema.append(segment.start_time)
            if last_direction * direction < 0 and segment.start_time not in local_extrema:
                # we changed sign, since
                local_extrema.append(segment.start_time)
            last_direction = direction
        return local_extrema

    def split_at(self, t: Union[float, Sequence[float]], change_original: bool = False) -> Sequence[T]:
        """
        Splits the Envelope at one or several points and returns a tuple of the pieces

        :param t: either the time t or a tuple/list of times t at which to split the curve
        :param change_original: if true, the original Envelope gets turned into the first of the returned tuple
        :return: tuple of Envelopes representing the pieces this has been split into
        """
        cls = type(self)
        to_split = self if change_original else cls.from_segments([x.clone() for x in self.segments])

        # if t is a tuple or list, we split at all of those times and return len(t) + 1 segments
        # This is implemented recursively. If len(t) is 1, t is replaced by t[0]
        # If len(t) > 1, then we sort and set aside t[1:] as remaining splits to do on the second half
        # and set t to t[0]. Note that we subtract t[0] from each of t[1:] to shift it to start from 0
        remaining_splits = None
        if hasattr(t, "__len__"):
            # ignore all split points that are outside this Envelope's range
            t = [x for x in t if to_split.start_time() <= x <= to_split.end_time()]
            if len(t) == 0:
                # if no usable points are left we're done (note we always return a tuple for consistency)
                return to_split,

            if len(t) > 1:
                t = list(t)
                t.sort()
                remaining_splits = [x - t[0] for x in t[1:]]
            t = t[0]

        # cover the case of trying to split outside of the Envelope's range
        # (note we always return a tuple for consistency)
        if not to_split.start_time() < t < to_split.end_time():
            return to_split,

        # Okay, now we go ahead with a single split at time t
        to_split.insert_interpolated(t, 0)
        for i, segment in enumerate(to_split.segments):
            if segment.start_time == t:
                second_half = cls.from_segments(to_split.segments[i:])
                to_split.segments = to_split.segments[:i]
                for second_half_segment in second_half.segments:
                    second_half_segment.start_time -= t
                    second_half_segment.end_time -= t
                break

        if remaining_splits is None:
            return to_split, second_half
        else:
            return to_split, second_half.split_at(remaining_splits, change_original=True)

    def _to_dict(self):
        json_dict = {'levels': self.levels}

        if all(x == self.durations[0] for x in self.durations):
            json_dict['length'] = self.length()
        else:
            json_dict['durations'] = self.durations

        if any(x != 0 for x in self.curve_shapes):
            json_dict['curve_shapes'] = self.curve_shapes

        if self.offset != 0:
            json_dict['offset'] = self.offset

        return json_dict

    @classmethod
    def _from_dict(cls, json_dict):
        curve_shapes = None if 'curve_shapes' not in json_dict else json_dict['curve_shapes']
        offset = 0 if 'offset' not in json_dict else json_dict['offset']
        if 'length' in json_dict:
            return cls.from_levels(json_dict['levels'], json_dict['length'], offset)
        else:
            return cls.from_levels_and_durations(json_dict['levels'], json_dict['durations'],
                                                 curve_shapes, offset)

    def is_shifted_version_of(self, other: T, tolerance: float = 1e-10) -> bool:
        """
        Determines if this segment is simply a shifted version of another segment

        :param other: another EnvelopeSegment
        :param tolerance: how close it needs to be to count as the same
        """
        assert isinstance(other, Envelope)
        return all(x.is_shifted_version_of(y, tolerance) for x, y in zip(self.segments, other.segments))

    def shift_vertical(self, amount) -> T:
        """
        Shifts the levels of this Envelope the specified amount.

        :param amount: the amount to shift up and down by
        :return: self, for chaining purposes
        """
        for segment in self.segments:
            segment.shift_vertical(amount)
        return self

    def scale_vertical(self, amount) -> T:
        """
        Scales the levels of this segment by the specified amount.

        :param amount: amount to scale output by
        :return: self, for chaining purposes
        """
        for segment in self.segments:
            segment.scale_vertical(amount)
        return self

    def shift_horizontal(self, amount: float) -> T:
        """
        Shifts the domain of this Envelope by the specified amount.

        :param amount: the amount to shift the domain by
        :return: self, for chaining purposes
        """
        for segment in self.segments:
            segment.shift_horizontal(amount)
        return self

    def scale_horizontal(self, amount: float) -> T:
        """
        Scales the domain of this Envelope by the specified amount.

        :param amount: amount to scale domain by
        :return: self, for chaining purposes
        """
        for segment in self.segments:
            segment.scale_horizontal(amount)
        return self

    def _get_graphable_point_pairs(self, resolution=25):
        x_values = []
        y_values = []
        for i, segment in enumerate(self.segments):
            # only include the endpoint on the very last segment, since otherwise there would be repeats
            segment_x_values, segment_y_values = segment._get_graphable_point_pairs(
                resolution=resolution, endpoint=(i == len(self.segments) - 1)
            )
            x_values.extend(segment_x_values)
            y_values.extend(segment_y_values)
        return x_values, y_values

    def show_plot(self, title: str = None, resolution: int = 25, show_segment_divisions: bool = True,
                  x_range: Tuple[float, float] = None, y_range: Tuple[float, float] = None) -> None:
        """
        Shows a plot of this Envelope using matplotlib.

        :param title: A title to give the plot.
        :param resolution: number of points to use per envelope segment
        :param show_segment_divisions: Whether or not to place dots at the division points between envelope segments
        :param x_range: min and max value shown on the x-axis
        :param y_range: min and max value shown on the y-axis
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Could not find matplotlib, which is needed for plotting.")
        fig, ax = plt.subplots()
        ax.plot(*self._get_graphable_point_pairs(resolution))
        if show_segment_divisions:
            ax.plot(self.times, self.levels, 'o')
        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        ax.set_title('Graph of Envelope' if title is None else title)
        plt.show()

    @staticmethod
    def _apply_binary_operation_to_pair(envelope1, envelope2, binary_function):
        envelope1 = envelope1.duplicate()
        envelope2 = envelope2.duplicate()
        for t in set(envelope1.times + envelope2.times):
            envelope1.insert_interpolated(t)
            envelope2.insert_interpolated(t)
        result_segments = []
        for s1, s2 in zip(envelope1.segments, envelope2.segments):
            this_segment_result = binary_function(s1, s2)
            # when we add or multiply two EnvelopeSegments, we might get an EnvelopeSegment if it's simple
            # or we might get an Envelope if the result is best represented by multiple segments
            if isinstance(this_segment_result, Envelope):
                # if it's an envelope, append all of it's segments
                result_segments.extend(this_segment_result.segments)
            else:
                # otherwise, it should just be a segment
                assert isinstance(this_segment_result, EnvelopeSegment)
                result_segments.append(this_segment_result)
        return Envelope.from_segments(result_segments)

    def _reciprocal(self):
        assert all(x > 0 for x in self.levels) or all(x < 0 for x in self.levels), \
            "Cannot divide by Envelope that crosses zero"
        return Envelope.from_segments([segment._reciprocal() for segment in self.segments])

    def __eq__(self, other):
        if not isinstance(other, Envelope):
            return False
        return all(this_segment == other_segment for this_segment, other_segment in zip(self.segments, other.segments))

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return Envelope.from_segments([segment + other for segment in self.segments])
        elif isinstance(other, Envelope):
            return Envelope._apply_binary_operation_to_pair(self, other, lambda a, b: a + b)
        else:
            raise ValueError("Envelope can only be added to a constant or another envelope")

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Envelope.from_segments([-segment for segment in self.segments])

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__radd__(-other)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return Envelope.from_segments([segment * other for segment in self.segments])
        elif isinstance(other, Envelope):
            return Envelope._apply_binary_operation_to_pair(self, other, lambda a, b: a * b)
        else:
            raise ValueError("Envelope can only be multiplied by a constant or another envelope")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return self._reciprocal() * other

    def __repr__(self):
        return "Envelope({}, {}, {}, {})".format(self.levels, self.durations, self.curve_shapes, self.offset)
