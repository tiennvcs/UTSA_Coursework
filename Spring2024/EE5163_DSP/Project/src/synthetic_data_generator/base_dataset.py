from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Abstract base class for denoising model.
    """
    def __init__(self):
        """
        :param thetamin: int, minimum used in uniform sampling of the parameters
        :param thetamax: int,  minimum used in uniform sampling of the parameters
        """
        # Stack data together and combine parameter sets to make calcs more efficient
        self.name = "Abstract class for dataset generator"

    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError()