# global
import abc

# local
import ivy
from ivy import Array


class _ArrayWithLayersExperimental(abc.ABC):

    # existing methods...

    def global_lp_pool(self: Array, p: int, data_format: str = 'NCHW') -> Array:
        """
        Compute the Lp norm of the input tensor over the entire tensor, per channel.

        Parameters
        ----------
        p : int
            The exponent value in the norm formulation.
        data_format : str, optional
            The data format of the input tensor, by default 'NCHW'.

        Returns
        -------
        ivy.Array
            The result of the pooling operation.
        """
        return ivy.global_lp_pool(self, p, data_format)