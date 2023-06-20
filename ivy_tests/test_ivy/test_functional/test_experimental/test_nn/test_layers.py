# global
from hypothesis import strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test

# existing tests...

@st.composite
def global_lp_pool_args(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=4,
            max_num_dims=4,
            min_dim_size=2,
            max_dim_size=5,
        )
    )
    p = draw(st.integers(min_value=1, max_value=3))
    data_format = draw(st.sampled_from(['NCHW', 'NHWC']))
    return dtype, x, p, data_format

@handle_test(
    fn_tree="functional.ivy.experimental.global_lp_pool",
    dtype_x_and_args=global_lp_pool_args(),
    test_gradients=st.just(False),
)
def test_global_lp_pool(
    *,
    dtype_x_and_args,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, p, data_format = dtype_x_and_args
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        p=p,
        data_format=data_format,
    )

# more tests...