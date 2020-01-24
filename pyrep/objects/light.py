from typing import Union
from pyrep.objects.object import Object
from pyrep.const import ObjectType


class Light(Object):
    """A light.
    """

    def __init__(self, name_or_handle: Union[str, int]):
        super().__init__(name_or_handle)

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.LIGHT
