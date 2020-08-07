"""
expenvelope Example: Value at Discontinuities

Demonstrates that, at points of discontinuity, "value_at" defaults to the right-hand limit. If the left-hand limit
is desired, the "from_left" flag can be set to True.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  SCAMP (Suite for Computer-Assisted Music in Python)                                           #
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

from expenvelope import Envelope

discontinuous_envelope = Envelope.from_levels_and_durations((3, 4, -2, -5), (2, 0, 3), (1, 0, -5))
discontinuous_envelope.show_plot(resolution=300)

print(discontinuous_envelope.value_at(2))
print(discontinuous_envelope.value_at(2, from_left=True))
