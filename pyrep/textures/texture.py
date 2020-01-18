from pyrep.backend import sim
from pyrep.errors import *


class Texture(object):
    """Base class for V-REP textures."""

    def __init__(self, texture_id: int):
        self._texture_id = texture_id

    @staticmethod
    def create(texture_path: str, interpolate=True, decal_mode=False,
               repeat_along_u=False, repeat_along_v=False,
               uv_scaling=[1., 1.], xy_g=[0., 0., 0.]) -> 'Texture':
        """Creates a texture in the scene.

        :param texture_path: Path to the texture image file.
        :param interpolate: Do not interpolate adjacent color patches.
        :param decal_mode: Apply the texture in decal mode.
        :param repeat_along_u: Repeat the texture along the U direction.
        :param repeat_along_v: Repeat the texture along the V direction.
        :param uv_scaling: The desired scaling of the texture along the U and V
        directions.
        :param xy_g: The texture x/y shift and the texture gamma rotation.
        :return: The created texture object.
        """
        options = 0
        if interpolate:
            options |= 1
        if decal_mode:
            options |= 2
        if repeat_along_u:
            options |= 4
        if repeat_along_v:
            options |= 8

        # Create the texture
        h_texture = sim.simCreateTexture(texture_path, options, None,
                                         uv_scaling, xy_g, 0, None)
        texture_id = sim.simGetShapeTextureId(h_texture)

        # Update the texture. If we don't do this (and a texture has previously
        # been applied to an object) changes won't get picked up by the
        # POV-Camera, although those using OpenGL seem to be able to.
        data = sim.simReadTexture(texture_id, 0)
        sim.simWriteTexture(texture_id, 0, data)

        # Create a texture object
        texture = Texture(texture_id)
        texture._handle = h_texture

        return texture

    def remove(self) -> None:
        """Removes this texture from the scene.

        :raises: ObjectAlreadyRemoved if the object is no longer on the scene.
        """
        try:
            sim.simRemoveObject(self._handle)
        except RuntimeError as e:
            raise ObjectAlreadyRemovedError(
                'The texture was already deleted.') from e

    def __eq__(self, other: 'Texture'):
        return self.get_texture_id() == other.get_texture_id()

    def get_texture_id(self) -> int:
        """Gets the texture id.

        :return: The internal texture id.
        """
        return self._texture_id

