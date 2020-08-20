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

from expenvelope.json_serializer import SavesToJSON


class Seed(SavesToJSON):

    def __init__(self, foo):
        self.foo = foo

    def _to_dict(self) -> dict:
        return {"foo": self.foo}

    @classmethod
    def _from_dict(cls, json_dict):
        return cls(**json_dict)


class Fruit(SavesToJSON):

    def __init__(self, seeds):
        self.seeds = seeds

    def _to_dict(self) -> dict:
        return {"seeds": self.seeds}

    @classmethod
    def _from_dict(cls, json_dict):
        return cls(**json_dict)


fruit_with_seeds = Fruit([Seed(6), Seed(-80), Fruit([Seed(-1), Seed(3)]), Seed("hello")])
print(fruit_with_seeds)
json_version = fruit_with_seeds.json_dumps()
print(json_version)
print(Fruit.json_loads(json_version))
