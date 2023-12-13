import gym
import numpy as np


class CastObservation(gym.ObservationWrapper):
    """Cast observations to a given type.

    Args:
        env: Env to wrap.
        dtype: Data type object.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env, dtype, noise=None):
        super().__init__(env)
        self.dtype = dtype
        self.noise = noise
        self._np_random = np.random.RandomState()

    def observation(self, observation):
        self.original_observation = observation
        if self.noise is not None:
            observation+=self._np_random.randn(len(self.noise))*self.noise
        return observation.astype(self.dtype, copy=False)

    def seed(self, seed):
        super().seed(seed)
        self._np_random.seed(seed)
        

class CastObservationToFloat32(CastObservation):
    """Cast observations to float32, which is commonly used for NNs.

    Args:
        env: Env to wrap.

    Attributes:
        original_observation: Observation before casting.
    """

    def __init__(self, env, noise=None):
        super().__init__(env, np.float32, noise)

        
