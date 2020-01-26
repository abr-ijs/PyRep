from typing import Union, List
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.backend import sim
import numpy as np


class Light(Object):
    """Lights are objects which allow you to light a scene.
    """

    def __init__(self, name_or_handle: Union[str, int]):
        super().__init__(name_or_handle)

        # Save the light's current parameters
        (self._initial_state,
         self._initial_diffuse_part,
         self._initial_specular_part) = sim.simGetLightParameters(
             self.get_handle())

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.LIGHT

    def randomize_params(self,
                         seed: int=None,
                         p_active: float=0.5,
                         diffuse_part_ranges: List[float]=None,
                         diffuse_part_sigmas: List[float]=[0.1, 0.1, 0.1],
                         specular_part_ranges: List[float]=None,
                         specular_part_sigmas: List[float]=[0.1, 0.1, 0.1]):
        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)

        # Activate the light with probability p_active
        if np.random.random() < p_active:
            sim.simSetLightParameters(self.get_handle(), 0,
                                      self._initial_diffuse_part,
                                      self._initial_specular_part)
        else:
            sim.simSetLightParameters(self.get_handle(), 1,
                                      self._initial_diffuse_part,
                                      self._initial_specular_part)

        # Get current diffuse part or uniformly sample it
        if diffuse_part_ranges is None:
            diffuse_part = self._initial_diffuse_part
        else:
            diffuse_part_ranges = np.asarray(diffuse_part_ranges)
            diffuse_part = list(np.random.uniform(diffuse_part_ranges[:, 0],
                                                  diffuse_part_ranges[:, 1],
                                                  len(diffuse_part_ranges)))

        # Add Gaussian noise to diffuse_part
        if diffuse_part_sigmas is not None:
            diffuse_part = [
                np.random.normal(diffuse_part[0], diffuse_part_sigmas[0], 1),
                np.random.normal(diffuse_part[1], diffuse_part_sigmas[1], 1),
                np.random.normal(diffuse_part[2], diffuse_part_sigmas[2], 1)]

        # Get current specular part or uniformly sample it
        if specular_part_ranges is None:
            specular_part = self._initial_specular_part
        else:
            specular_part_ranges = np.asarray(specular_part_ranges)
            specular_part = list(np.random.uniform(specular_part_ranges[:, 0],
                                                   specular_part_ranges[:, 1],
                                                   len(specular_part_ranges)))

        # Add Gaussian noise to specular_part
        if specular_part_sigmas is not None:
            specular_part = [
                np.random.normal(specular_part[0], specular_part_sigmas[0], 1),
                np.random.normal(specular_part[1], specular_part_sigmas[1], 1),
                np.random.normal(specular_part[2], specular_part_sigmas[2], 1)]
