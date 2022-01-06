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

import math
import bisect
from abc import ABC, abstractmethod
from typing import TypeVar, Type
import json


def _make_envelope_segments_from_function(function, domain_start, domain_end, scanning_step_size=0.05,
                                          keypoint_resolution_multiple=1, slope_change_threshold=0.1,
                                          iterations=5, min_key_point_distance=1e-7, check_for_discontinuities=True):
    """
    Makes a list of EnvelopeSegments from the given function

    :param function: a function of one variable
    :param domain_start: where to start in the domain
    :param domain_end: where to end in the domain
    :param scanning_step_size: step size for the initial scans for key points (discontinuities, undifferentiable points,
        extrema and inflection points)
    :param keypoint_resolution_multiple: how many extra key points to place in between each pair of key points to
        fit the curve more faithfully.
    :param slope_change_threshold: the degree of sudden slope change we look for in scanning for knees and
        discontinuities. Only sudden changes of slope larger than this will be detected, so if there is a very subtle
        knee in the function, this value may need to be reduced.
    :param iterations: how many times to zoom in to find key points more precisely
    :param min_key_point_distance: key points must be this far apart or they are merged
    :param check_for_discontinuities: if True, look for discontinuities in the function before searching for extrema
        and inflection points.
    """
    from .envelope_segment import EnvelopeSegment

    segments = []

    discontinuities = _get_discontinuities_and_undifferentiable_points(
        function, domain_start, domain_end, scanning_step_size,
        slope_change_threshold=slope_change_threshold, iterations=iterations
    ) if check_for_discontinuities else ()

    if len(discontinuities) > 0:
        # discontinuities is always even in length, containing a point right before and right after each jump/knee
        discontinuities = [domain_start] + discontinuities + [domain_end]
        for i in range(0, len(discontinuities), 2):
            section_start, section_end = discontinuities[i], discontinuities[i + 1]
            segments.extend(_make_envelope_segments_from_function(
                function, section_start, section_end, scanning_step_size=scanning_step_size,
                keypoint_resolution_multiple=keypoint_resolution_multiple,
                slope_change_threshold=slope_change_threshold, iterations=iterations,
                min_key_point_distance=min_key_point_distance, check_for_discontinuities=False
            ))
            if section_end != domain_end:
                segments.append(EnvelopeSegment(section_end, discontinuities[i + 2], function(section_end),
                                                function(discontinuities[i + 2]), 0))
        return segments
    else:
        key_points = _get_extrema_and_inflection_points(function, domain_start, domain_end,
                                                        scanning_step_size=scanning_step_size, iterations=iterations)
        if keypoint_resolution_multiple > 1:
            key_points = [left + k * (right - left) / keypoint_resolution_multiple
                          for left, right in zip(key_points[:-1], key_points[1:])
                          for k in range(keypoint_resolution_multiple)] + [key_points[-1]]
        segments = []
        i = 0
        while i < len(key_points) - 1:
            segment_start = key_points[i]
            segment_end = key_points[i + 1]
            halfway_point = (segment_start + segment_end) / 2
            segment_start_value = function(segment_start)
            segment_end_value = function(segment_end)
            segment_halfway_value = function(halfway_point)

            # we're trying to split at the min / max locations to get monotonic segments
            # in case we get a segment that is neither strictly monotonic not constant,
            # we can just split it straight down the middle
            is_strictly_monotonic = min(segment_start_value, segment_end_value) < segment_halfway_value < \
                                    max(segment_start_value, segment_end_value)
            is_constant = segment_start_value == segment_halfway_value == segment_end_value
            if not is_strictly_monotonic and not is_constant and segment_end - segment_start > min_key_point_distance:
                # if it's not monotonic, and it's not constant, and the key points aren't already super close,
                # add a key point halfway and try again without incrementing i
                key_points.insert(i + 1, halfway_point)
                continue

            if is_strictly_monotonic:
                segments.append(EnvelopeSegment.from_endpoints_and_halfway_level(
                    segment_start, segment_end,
                    segment_start_value, segment_end_value, segment_halfway_value
                ))
            else:
                # not monotonic (constant segment or key points got too close together)
                segments.append(EnvelopeSegment(
                    segment_start, segment_end,
                    segment_start_value, segment_end_value, 0
                ))
            i += 1

        return segments


def _get_discontinuities_and_undifferentiable_points(function, domain_start, domain_end, scanning_step_size,
                                                     iterations, slope_change_threshold):
    """
    Finds jump discontinuities and points where the function is not differentiable, and returns a list of points right
    before and right after each of these discontinuities.

    :param function: a function of one variable
    :param domain_start: where to start in the domain
    :param domain_end: where to end in the domain
    :param scanning_step_size: the step size to use when scanning the function
    :param iterations: How many iterations of zooming in and bumping the slope_change_threshold to do.
    :param slope_change_threshold: a discontinuity or undifferentiable point is found by looking for a sudden change in
        slope between small scanning steps. If the slope changes by more than this value between successive steps,
        we repeatedly zoom in to that step, cut the scanning_step_size by a factor of 10, and see if this jump remains.
        If it's differentiable, it should become linear and the change of slope should fall below the threshold with
        enough iterations. If it's non-differentiable, the change of slope should remain the same. If it's not even
        continuous, the change of slope should approach infinity, which will definitely be over the threshold.
    """
    x = domain_start
    last_f_x = function(domain_start)
    last_slope = None
    discontinuities = []
    while x < domain_end:
        x += scanning_step_size
        f_x = function(x)
        slope = (f_x - last_f_x) / scanning_step_size
        if last_slope is not None and abs(slope - last_slope) > slope_change_threshold:
            # a slope that is suddenly greater or less than the previous slope
            if iterations <= 0:
                discontinuities.extend([x - scanning_step_size, x])
            else:
                new_bounds = max(domain_start, x - scanning_step_size * 1.1), \
                             min(domain_end, x + scanning_step_size * 0.1)
                discontinuities.extend(_get_discontinuities_and_undifferentiable_points(
                    function, *new_bounds, scanning_step_size/10, iterations - 1, slope_change_threshold
                ))
            # if we find a discontinuity, we need to reset last_slope to 0, otherwise it will register both
            # going into the discontinuity and leaving it
            last_slope = None
        else:
            last_slope = slope
        last_f_x = f_x
    discontinuities.sort()
    return discontinuities


def _get_extrema_and_inflection_points(function, domain_start, domain_end, scanning_step_size=0.01, iterations=5,
                                       include_endpoints=True, return_on_first_point=False):
    key_points = []
    last_f_x = function(domain_start)
    first_difference = None
    second_difference = None
    x = domain_start
    while x < domain_end:
        x += scanning_step_size
        f_x = function(x)

        # some rounding is necessary to avoid floating point inaccuracies from creating false sign changes
        this_difference = f_x - last_f_x
        if abs(this_difference) < 1e-13:
            this_difference = 0

        if first_difference is not None:
            # if we've gone through the loop before and have a value from last time for first_difference
            # check if first difference changes sign in the first derivative
            if math.copysign(1, this_difference) * math.copysign(1, first_difference) < 0:
                # there's been a change of sign, so there's a local min or max somewhere between the last
                # step and this one. If we are iterating further, search for the precise location
                if iterations > 1:
                    extremum = _get_extrema_and_inflection_points(
                        function, x - 2 * scanning_step_size, x, scanning_step_size/10, iterations - 1,
                        include_endpoints=False, return_on_first_point=True
                    )
                else:
                    # otherwise, simply use the average
                    extremum = x - scanning_step_size / 2
                # if we return_on_first_point (as we do in iteration), just return
                if return_on_first_point:
                    return extremum
                else:
                    # otherwise, check that it's not redundant and add it to the list
                    if extremum not in key_points:
                        key_points.append(extremum)

            this_second_difference = this_difference - first_difference
            if abs(this_second_difference) < 1e-13:
                this_second_difference = 0

            if second_difference is not None:
                # check if second difference changes sign
                if math.copysign(1, this_second_difference) * math.copysign(1, second_difference) < 0:
                    # there's been a change of sign, so there's an inflection point somewhere between the last
                    # step and this one. If we are iterating further, search for the precise location
                    if iterations > 1:
                        inflection_point = _get_extrema_and_inflection_points(
                            function, x - 3 * scanning_step_size, x, scanning_step_size / 10, iterations - 1,
                            include_endpoints=False, return_on_first_point=True
                        )
                    else:
                        # otherwise, simply use the average
                        inflection_point = x - scanning_step_size / 2

                    # if we return_on_first_point (as we do in iteration), just return
                    if return_on_first_point:
                        return inflection_point
                    else:
                        # otherwise check that it's not redundant and add it to the list
                        if inflection_point not in key_points:
                            key_points.append(inflection_point)

            second_difference = this_second_difference
        first_difference = this_difference
        last_f_x = f_x

    if return_on_first_point:
        # something has gone a little wrong, because we did an extra iteration to find the key point more exactly,
        # but we didn't get any closer. So just return the average.
        return (domain_start + domain_end) / 2

    key_points.sort()
    # remove near duplicates by comparing each number to the next one and including it only if there's a gap bigger
    # than scanning_step_size. (We have to add the last key point back on, since it's not the first of an adjacent pair)
    key_points = [x for x, y in zip(key_points[:-1], key_points[1:]) if abs(x - y) > scanning_step_size] + key_points[-1:]

    if include_endpoints:
        if len(key_points) > 0 and abs(key_points[0] - domain_start) < scanning_step_size:
            key_points.pop(0)
        if len(key_points) > 0 and abs(key_points[-1] - domain_end) < scanning_step_size:
            key_points.pop(-1)
        key_points = [domain_start, *key_points, domain_end]

    return key_points


def _curve_shape_from_start_mid_and_end_levels(start_level, halfway_level, end_level):
    # utility for finding the curvature given the level halfway as a guide
    if start_level == end_level:
        # if the end_level equals the start_level, then the best we can do is a flat line
        # hopefully this happens because it truly is a flat line, not a bump or a trough
        return 0
    if not min(start_level, end_level) < halfway_level < max(start_level, end_level):
        raise ValueError("Halfway level must be strictly between start and end levels, or equal to both.")
    halfway_level_normalized = (halfway_level - start_level) / (end_level - start_level)
    return 2 * math.log(1 / halfway_level_normalized - 1)


# ----------------------------------------- curvature to % filled utilities ----------------------------------------


def _get_filled_amount_from_curvature(curvature):
    """
    For a given segment going from low to high, the proportion of the area between the low mark and the curve divided
    by the area between the low mark and the high mark can vary from 0 (as curvature approaches infinity) to 1
    (as curvature approaches negative infinity), with a value of 0.5 when the curvature is zero. This calculates that
    via the equation `P_filled =  1 / S - 1 / (e^S - 1)`.
    """
    return 1 / curvature - 1 / (math.exp(curvature) - 1) if curvature != 0 else 0.5


"""
Since the equation getting proportion filled from curvature is not invertible analytically, we use a table look-up
to do the inverse. We use a resolution of 0.001 for the range between -20 and 20; outside of that range, the e^S term 
is so large that we can invert a simplified function analytically.
"""
_curvature_values = [x / 1000 for x in reversed(range(-20000, 20001))]
_filled_amount_as_function_of_curvature = [_get_filled_amount_from_curvature(s) for s in _curvature_values]


def _get_curvature_from_filled_amount(filled_amount):
    """
    For a given segment going from low to high, returns the curvature for a given portion of the area described above
    that is filled. from tests, the max error here is about 2 * 10^-9, occurring right at the switch from table lookup
    to using the simple substitute functions. Also it takes about 10^-5 seconds to calculate. Not too shabby!
    """
    assert 0 < filled_amount < 1
    if 0.05 <= filled_amount <= 0.95:
        index = bisect.bisect(_filled_amount_as_function_of_curvature, filled_amount)
        upper_percent = _filled_amount_as_function_of_curvature[index]
        lower_percent = _filled_amount_as_function_of_curvature[index - 1]
        fractional_part = (filled_amount - lower_percent) / (upper_percent - lower_percent)
        return _curvature_values[index - 1] + \
               fractional_part * (_curvature_values[index] - _curvature_values[index - 1])
    elif filled_amount > 0.95:
        # for curvature < -20, 1 / S - 1 / (e^S - 1) is almost exactly 1 / S + 1
        # thus we can just act like P_filled = 1 / S + 1, and thus S = 1 / (P_filled - 1)
        return 1 / (filled_amount - 1)
    else:
        # for curvature > 20, 1 / S - 1 / (e^S - 1) is almost exactly 1 / S
        # thus we can just act like P_filled = 1 / S, and thus S = 1 / P_filled
        return 1 / filled_amount
