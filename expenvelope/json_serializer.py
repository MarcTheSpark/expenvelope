"""
Module containing the :class:`SavesToJSON` abstract class for serializing complex objects back and forth to JSON files.
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

import json
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Type
from copy import deepcopy


class SavesToJSONMeta(ABCMeta):
    """
    Used to keep track of all known subclasses of SavesToJSON, so that objects can be reconstructed.
    """
    names_to_types = {}
    types_to_names = {}
    
    def __new__(mcs, clsname, bases, attrs):
        new_class = super(SavesToJSONMeta, mcs).__new__(mcs, clsname, bases, attrs)
        SavesToJSONMeta.names_to_types[clsname] = new_class
        SavesToJSONMeta.types_to_names[new_class] = clsname
        return new_class


T = TypeVar('T', bound='SavesToJSON')


class SavesToJSON(metaclass=SavesToJSONMeta):

    """
    Abstract class that, when implemented, gives the ability to save to and from JSON objects.
    Children must implement the ``_to_dict`` and ``_from_dict`` functions which convert back and forth
    between an instance and a dictionary representing that instance's data. The data in such a dictionary
    may contain only standard json-serializable types (lists, dicts, ints, floats, strings, etc.) and other
    objects that implement the SavesToJSON interface,
    """

    @abstractmethod
    def _to_dict(self) -> dict:
        """
        Should define a dictionary representation of this object. Any objects nested within that dictionary must inherit
        from SavesToJSON.
        """
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, json_dict):
        """
        Should define how one reconstructs an object of this class from its dictionary representation (as returned by
        ``_to_dict``).
        """

    def json_dumps(self) -> str:
        """
        Dump this object as a JSON string. This uses a custom encoder that recognizes and appropriately converts any
        attributes that are object inheriting from SavesToJSON.
        """
        return json.dumps(self, default=SavesToJSON._encoder_default, sort_keys=True, indent=4)

    def save_to_json(self, file_path: str) -> None:
        """
        Save this object to a JSON file using the given path. This uses a custom encoder that recognizes and
        appropriately converts any attributes that are object inheriting from SavesToJSON.

        :param file_path: path for saving the file
        """
        with open(file_path, "w") as file:
            json.dump(self, file, default=SavesToJSON._encoder_default, sort_keys=True, indent=4)

    @classmethod
    def json_loads(cls, s: str) -> T:
        """
        Load this object from a JSON string. This uses a custom decoder that looks for a "_type" key in any
        object/dictionary being parsed and converts it to the class specified (assuming it a subclass of
        SavesToJSON).

        :param s: a string representing this object in JSON format
        """
        out = json.loads(s, object_hook=SavesToJSON._decoder_object_hook)
        if cls != SavesToJSON and not isinstance(out, cls):
            raise ValueError(
                "Trying to load object of type {object_type} using `{correct_type}.json_loads`. Use "
                "`{object_type}.json_loads` or generic `SavesToJSON.json_loads` instead.".format(
                    object_type=type(out).__name__, correct_type=cls.__name__
                )
            )
        return out

    @classmethod
    def load_from_json(cls: Type[T], file_path: str) -> T:
        """
        Load this object from a JSON file with the given path. This uses a custom decoder that looks for a
        "_type" key in any object/dictionary being parsed and converts it to the class specified (assuming it
        a subclass of SavesToJSON).

        :param file_path: path for loading the file
        """
        with open(file_path, "r") as file:
            out = json.load(file, object_hook=SavesToJSON._decoder_object_hook)
            if cls != SavesToJSON and not isinstance(out, cls):
                raise ValueError(
                    "Trying to load object of type {object_type} using `{correct_type}.load_from_json`. Use "
                    "`{object_type}.load_from_json` or generic `SavesToJSON.load_from_json` instead.".format(
                        object_type=type(out).__name__, correct_type=cls.__name__
                    )
                )
            return out

    @staticmethod
    def _decoder_object_hook(json_object):
        if "_type" in json_object:
            if json_object["_type"] not in SavesToJSONMeta.names_to_types:
                raise json.JSONDecodeError("Object type {} not understood.".format(json_object["_type"]))
            object_type = SavesToJSONMeta.names_to_types[json_object["_type"]]
            del json_object["_type"]
            return object_type._from_dict(json_object)
        return json_object

    @staticmethod
    def _encoder_default(obj):
        if hasattr(obj, "_to_dict"):
            converted = obj._to_dict()
            converted["_type"] = SavesToJSONMeta.types_to_names[type(obj)]
            return converted
        return obj

    def duplicate(self: T) -> T:
        """
        Returns a copy of this object by serializing to and from JSON.
        """
        return type(self)._from_dict(deepcopy(self._to_dict()))
