import math
import bisect


def _make_envelope_segments_from_function(function, domain_start, domain_end, resolution_multiple=1,
                                          key_point_precision=100, key_point_iterations=5):
    from .envelope_segment import EnvelopeSegment
    assert isinstance(resolution_multiple, int) and resolution_multiple > 0
    key_points = _get_extrema_and_inflection_points(function, domain_start, domain_end,
                                                    key_point_precision, key_point_iterations)
    if resolution_multiple > 1:
        key_points = [l + k * (r - l) / resolution_multiple
                      for l, r in zip(key_points[:-1], key_points[1:])
                      for k in range(resolution_multiple)] + [key_points[-1]]

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
        if not (is_strictly_monotonic or is_constant):
            # if we are splitting it, add a key point halfway and try again without incrementing
            key_points.insert(i + 1, halfway_point)
            continue

        segments.append(EnvelopeSegment.from_endpoints_and_halfway_level(
            segment_start, segment_end,
            segment_start_value, segment_end_value, segment_halfway_value
        ))
        i += 1

    return segments


def _get_extrema_and_inflection_points(function, domain_start, domain_end, resolution=100, iterations=5,
                                       include_endpoints=True, return_on_first_point=False):
    assert resolution >= 10 or iterations == 1, "Resolution should be at least 10 if iteration is being used"
    key_points = []
    value = None
    first_difference = None
    second_difference = None
    step = (domain_end - domain_start) / resolution
    for x in range(0, resolution):
        t = domain_start + x * step
        this_value = function(t)

        if value is not None:
            # some rounding is necessary to avoid floating point inaccuracies from creating false sign changes
            this_difference = round(this_value - value, 10)
            if first_difference is not None:
                # check if first difference changes sign in the first derivative
                if this_difference * first_difference < 0:
                    # there's been a change of sign, so there's a local min or max somewhere between the last
                    # step and this one. If we are iterating further, search for the precise location
                    if iterations > 1:
                        extremum = _get_extrema_and_inflection_points(
                            function, t - step, t, max(10, int(resolution / 2)), iterations - 1,
                            include_endpoints=False, return_on_first_point=True
                        )
                    else:
                        # otherwise, simply use the average
                        extremum = t - step / 2

                    # if we return_on_first_point (as we do in iteration), just return
                    if return_on_first_point:
                        return extremum
                    else:
                        # otherwise check that it's not redundant and add it to the list
                        if extremum not in key_points:
                            key_points.append(extremum)

                this_second_difference = round(this_difference - first_difference, 10)

                if second_difference is not None:
                    # check if second difference changes sign
                    if this_second_difference * second_difference < 0:
                        # there's been a change of sign, so there's an inflection point somewhere between the last
                        # step and this one. If we are iterating further, search for the precise location
                        if iterations > 1:
                            inflection_point = _get_extrema_and_inflection_points(
                                function, t - step, t, max(10, int(resolution / 2)), iterations - 1,
                                include_endpoints=False, return_on_first_point=True
                            )
                        else:
                            # otherwise, simply use the average
                            inflection_point = t - step / 2

                        # if we return_on_first_point (as we do in iteration), just return
                        if return_on_first_point:
                            return inflection_point
                        else:
                            # otherwise check that it's not redundant and add it to the list
                            if inflection_point not in key_points:
                                key_points.append(inflection_point)

                second_difference = this_second_difference
            first_difference = this_difference
        value = this_value

    if return_on_first_point:
        # something has gone a little wrong, because we did an extra iteration to find the key point more exactly,
        # but we didn't get any closer. So just return the average.
        return (domain_start + domain_end) / 2

    if include_endpoints:
        return [domain_start] + key_points + [domain_end]
    else:
        return key_points


def _curve_shape_from_start_mid_and_end_levels(start_level, halfway_level, end_level):
    # utility for finding the curvature given the level halfway as a guide
    if start_level == end_level:
        # if the end_level equals the start_level, then the best we can do is a flat line
        # hopefully this happens because it truly is a flat line, not a bump or a trough
        return 0
    assert min(start_level, end_level) < halfway_level < max(start_level, end_level), \
        "Halfway level must be strictly between start and end levels, or equal to both."
    halfway_level_normalized = (halfway_level - start_level) / (end_level - start_level)
    return 2 * math.log(1 / halfway_level_normalized - 1)


# ----------------------------------------- curvature to % filled utilities ----------------------------------------


def get_filled_amount_from_curvature(curvature):
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
_filled_amount_as_function_of_curvature = [get_filled_amount_from_curvature(s) for s in _curvature_values]


def get_curvature_from_filled_amount(filled_amount):
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
        return _curvature_values[index - 1] + fractional_part * (_curvature_values[index] - _curvature_values[index - 1])
    elif filled_amount > 0.95:
        # for curvature < -20, 1 / S - 1 / (e^S - 1) is almost exactly 1 / S + 1
        # thus we can just act like P_filled = 1 / S + 1, and thus S = 1 / (P_filled - 1)
        return 1 / (filled_amount - 1)
    else:
        # for curvature > 20, 1 / S - 1 / (e^S - 1) is almost exactly 1 / S
        # thus we can just act like P_filled = 1 / S, and thus S = 1 / P_filled
        return 1 / filled_amount
