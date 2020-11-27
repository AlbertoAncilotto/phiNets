import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from phi_net_source import Hswish, phi_net


def get_flops(test_params):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model=phi_net(res0=test_params[0], B0=test_params[1], d=1, alpha0=test_params[2], a=1, beta=test_params[3], t_zero=test_params[4], first_conv_filters=test_params[5], b1_filters=test_params[6], b2_filters=test_params[7], classes=10, squeeze_excite=False, h_swish=False, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, test_params[0], test_params[0], 3)), conv5_percent=test_params[8])
            
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops

