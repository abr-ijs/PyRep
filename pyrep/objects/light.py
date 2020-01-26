from typing import Union, List
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.backend import sim
import numpy as np


class Light(Object):
    """A light.
    """

    def __init__(self, name_or_handle: Union[str, int]):
        super().__init__(name_or_handle)

        # Save the light's current parameters
        self.initial_position = self.get_position()
        self.initial_orientation = self.get_orientation()
        (self.initial_state,
         self.initial_diffuse_part,
         self.initial_specular_part) = sim.simGetLightParameters(
             self.get_handle())

    def _get_requested_type(self) -> ObjectType:
        return ObjectType.LIGHT

    def randomize(self,
                  seed: int=None,
                  position_ranges: List[float]=None,
                  orientation_ranges: List[float]=None,
                  position_sigmas: List[float]=[0.01, 0.01, 0.01],
                  orientation_sigmas: List[float]=[0.01, 0.01, 0.01],
                  p_active: float=0.5):
        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)

        # Get current position or uniformly sample it
        if position_ranges is None:
            position = self.initial_position
        else:
            position_ranges = np.asarray(position_ranges)
            position = list(np.random.uniform(position_ranges[:, 0],
                                              position_ranges[:, 1],
                                              len(position_ranges)))

        # Get current orientation or uniformly sample it
        if orientation_ranges is None:
            orientation = self.initial_orientation
        else:
            orientation_ranges = np.asarray(orientation_ranges)
            orientation = list(np.random.uniform(orientation_ranges[:, 0],
                                                 orientation_ranges[:, 1],
                                                 len(orientation_ranges)))

        # Add Gaussian noise to position
        position = [
            np.random.normal(position[0], position_sigmas[0], 1),
            np.random.normal(position[1], position_sigmas[1], 1),
            np.random.normal(position[2], position_sigmas[2], 1)]

        # Add Gaussian noise to orientation
        orientation = [
            np.random.normal(orientation[0], orientation_sigmas[0], 1),
            np.random.normal(orientation[1], orientation_sigmas[1], 1),
            np.random.normal(orientation[2], orientation_sigmas[2], 1)]

        # Set new position & orientation
        self.set_position(position)
        self.set_orientation(orientation)

        # Activate the light with probability p_active
        if np.random.random() < p_active:
            sim.simSetLightParameters(self.get_handle(), 0,
                                      self.initial_diffuse_part,
                                      self.initial_specular_part)
        else:
            sim.simSetLightParameters(self.get_handle(), 1,
                                      self.initial_diffuse_part,
                                      self.initial_specular_part)
