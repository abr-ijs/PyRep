import pyrep
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.backend import sim
from typing import Union, List
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
                         diffuse_part_ranges: List[List[float]]=None,
                         diffuse_part_mean: List[float]=None,
                         diffuse_part_cov: [List[float], List[List[float]]]
                         =[0.0001, 0.0001, 0.0001],
                         specular_part_ranges: List[List[float]]=None,
                         specular_part_mean: List[float]=None,
                         specular_part_cov: [List[float], List[List[float]]]
                         =[0.0001, 0.0001, 0.0001]):
        """Randomize light activation, diffuse part and specular part.

        Both the diffuse part and the specular part may be either:
        (a) uniformly sampled within specified [low, high] ranges
            (e.g. ranges=[[0., 1.], [0., 1.], [0., 1.]], mean=None, cov=None),
        (b) sampled from the normal distribution
            (e.g. ranges=None, mean=[0.5, 0.5, 0.5],
            cov=[0.001, 0.001, 0.001]),
        (c) specified as their initial value, then perturbed with
            Gaussian noise (e.g. ranges=None, mean=None,
            cov=[0.001, 0.001, 0.001]), or
        (d) uniformly sampled within specified ranges, then perturbed with
            Gaussian noise (e.g. ranges=[[0., 1.], [0., 1.], [0., 1.]],
            mean=None, cov=[0.001, 0.001, 0.001]).

        :param p_active: A probability specifying the likelihood of the light
            being switched on (with 1 - p_active being the likelihood of the
            light being switched off).
        :param diffuse_part_ranges: A 3x2 matrix of values specifying
            [low,high] ranges from which each of the (r,g,b) light diffuse part
            values should be uniformly sampled. If set to None, the initial
            diffuse part values will be used.
        :param diffuse_part_mean: A list of 3 values specifying the mean
            (r,g,b) diffuse part values for Gaussian sampling. If set to None,
            the initial diffuse part values will be used.
        :param diffuse_part_cov: Either a 3-vector of values specifying the
            diagonal of the covariance matrix for the Gaussian sampling
            calculation, or a 3x3 matrix specifying the entire matrix. If set
            to None, Gaussian sampling will not be used.
        :param specular_part_ranges: A 3x2 matrix of values specifying
            [low,high] ranges from which each of the (r,g,b) light specular
            part values should be uniformly sampled. If set to None, the
            initial specular part values will be used.
        :param specular_part_mean: A list of 3 values specifying the mean
            (r,g,b) specular part values for Gaussian sampling. If set to None,
            the initial specular part values will be used.
        :param specular_part_cov: Either a 3-vector of values specifying the
            diagonal of the covariance matrix for the Gaussian sampling
            calculation, or a 3x3 matrix specifying the entire matrix. If set
            to None, Gaussian sampling will not be used.
        :return: The new position.
        """
        # Input validation
        assert(diffuse_part_ranges is None or
               np.asarray(diffuse_part_ranges).shape == (3, 2))
        assert(diffuse_part_mean is None or
               np.asarray(diffuse_part_mean).shape == (3,))
        assert(diffuse_part_cov is None or
               np.asarray(diffuse_part_cov).shape == (3,) or
               np.asarray(diffuse_part_cov).shape == (3, 3))
        assert(not (diffuse_part_ranges is None and
                    diffuse_part_mean is None and diffuse_part_cov is None))
        assert(specular_part_ranges is None or
               np.asarray(specular_part_ranges).shape == (3, 2))
        assert(specular_part_mean is None or
               np.asarray(specular_part_mean).shape == (3,))
        assert(specular_part_cov is None or
               np.asarray(specular_part_cov).shape == (3,) or
               np.asarray(specular_part_cov).shape == (3, 3))
        assert(not (specular_part_ranges is None and
                    specular_part_mean is None and specular_part_cov is None))

        # Seed the random number generator
        if seed is not None:
            np.random.seed(seed)

        # Decide whether or not to use the initial diffuse_part
        if diffuse_part_ranges is None and diffuse_part_mean is None:
            diffuse_part_mean = self._initial_diffuse_part

        # Sample
        diffuse_part = pyrep.PyRep.random_sample(seed=seed,
                                                 ranges=diffuse_part_ranges,
                                                 mean=diffuse_part_mean,
                                                 cov=diffuse_part_cov)

        # Decide whether or not to use the initial specular_part
        if specular_part_ranges is None and specular_part_mean is None:
            specular_part_mean = self._initial_specular_part

        # Sample
        specular_part = pyrep.PyRep.random_sample(seed=seed,
                                                  ranges=specular_part_ranges,
                                                  mean=specular_part_mean,
                                                  cov=specular_part_cov)

        # Activate the light with probability p_active
        if np.random.random() < p_active:
            active = 0
        else:
            active = 1
            sim.simSetLightParameters(self.get_handle(), active,
                                      list(diffuse_part),
                                      list(specular_part))

        return active, diffuse_part, specular_part
