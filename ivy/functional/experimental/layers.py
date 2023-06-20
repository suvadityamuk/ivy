# global
import ivy

# local
from ivy import reduce_sum, pow, reshape


def global_lp_pool(x: ivy.Array, p: int, data_format: str = 'NCHW') -> ivy.Array:
    """
    Compute the Lp norm of the input tensor over the entire tensor, per channel.

    Parameters
    ----------
    x : ivy.Array
        Input tensor.
    p : int
        The exponent value in the norm formulation.
    data_format : str, optional
        The data format of the input tensor, by default 'NCHW'.

    Returns
    -------
    ivy.Array
        The result of the pooling operation.
    """
    if data_format == 'NHWC':
        x = reshape(x, (x.shape[0], -1, x.shape[-1]))
    else:
        x = reshape(x, (x.shape[0], x.shape[1], -1))

    return pow(reduce_sum(pow(ivy.abs(x), p), -1), 1 / p)