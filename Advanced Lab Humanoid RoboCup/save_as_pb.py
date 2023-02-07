import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


if __name__=='__main__':

    name = "cnn_1C_BN_[16]_[16_1]_mean_norm"
    input_path = 'h5_files/' + name + "1.h5"
    pb_path = name + "_frozen1.pb"
    # output layer num
    layer_num = str(2)
    inference_path = 'pb_inference_files/' + name + '_inference1.pb'

    """ Create, compile and train model... """
    K.clear_session()
    K.set_learning_phase(0)
    model = tf.keras.models.load_model(input_path)
    # freeze graph
    frozen_graph = freeze_session(K.get_session(input_path),
                                  output_names=[out.op.name for out in model.outputs])

    """ save pb"""
    tf.io.write_graph(frozen_graph, logdir="pb_frozen_files", name=pb_path, as_text=False)

    """ get input and output names of graph """
    gf = tf.GraphDef()
    gf.ParseFromString(open('pb_frozen_files/' + pb_path, 'rb').read())
    name = [n.name + '=>' + n.op for n in gf.node if n.op in ('Sigmoid', 'Placeholder')]
    print('name of intput layer and output layer')
    print(name)

    """ pb to inference graph """
    with tf.gfile.FastGFile('pb_frozen_files/' + pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['conv2d_input'],
                                                                      ['activation_' + layer_num + '/Sigmoid'],
                                                                      tf.float32.as_datatype_enum)
        graph_def = TransformGraph(graph_def, ['module_apply_default/hub_input/Sub'],
                                   ['activation_' + layer_num + '/Sigmoid'],
                                   ['remove_nodes(op=PlaceholderWithDefault)',
                                    'strip_unused_nodes(type=float, shape=\"1,224,224,3\")',
                                    'sort_by_execution_order'])

        with tf.gfile.FastGFile(inference_path, 'wb') as file:
            file.write(graph_def.SerializeToString())

