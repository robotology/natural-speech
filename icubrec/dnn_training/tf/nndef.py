#!/usr/bin/env python3

import numpy as np


class nndef:
    fformat = '{:.6e}'

    def parse_from_tf_graph(graph, opname):
        activation_op = nndef.get_op(graph, opname)
        net = []
        while nndef.is_activation(activation_op):
            net.insert(0, nndef.parse_layer(activation_op))
            activation_op = net[0]['input']
        return net

    def is_activation(op):
        return op.type in ["Softmax", "Relu"]

    def get_op(graph, opname):
        op = graph.get_operation_by_name(opname)
        if op is None:
            error = "{} operator doesn't exist in the graph."
            raise ValueError(error.format(opname))
        return op

    def parse_layer(activation_op):
        if not nndef.is_activation(activation_op):
            error = "{} operation is not an activation function."
            raise Error(error.format(activation_op.name))
        layer = {'activation': activation_op}
        for tensor in activation_op.inputs[0].op.inputs:
            op = tensor.op
            if op.type == "MatMul":
                matmul_op = op
            if op.type == "Identity":
                layer['biases'] = op.inputs[0]
        for tensor in matmul_op.inputs:
            op = tensor.op
            if op.type in ["Relu", "QueueDequeueManyV2", "Placeholder"]:
                layer['input'] = op
            if op.type == "Identity":
                layer['weights'] = op.inputs[0]
        return layer

    def convert_for_htk(net, feature_type="FBANK_D_A_Z", nb_frames=11):
        # Header
        htk_def = nndef.generate_htk_header(net[0], feature_type, nb_frames)
        # Layers definition
        labels = [str(i) for i in range(2, len(net) + 1)]
        labels.append("out")
        source = "<{}>".format(feature_type)
        for layer, label in zip(net, labels):
            layer_def = nndef.generate_htk_layer(
                layer, label, nb_frames, source)
            htk_def.extend(layer_def)
            source = "~L \"layer{}\"".format(label)
            nb_frames = 1
        # Net definition
        htk_def.extend(nndef.generate_htk_net_def(labels))
        return htk_def

    def generate_htk_header(first_layer, feature_type, nb_frames):
        header = ["~o"]
        input_size = int(first_layer['weights'].get_shape()[0].value / nb_frames)
        header.append("<STREAMINFO> 1 {}".format(input_size))
        header.append("<VECSIZE> {}<{}>".format(input_size, feature_type))
        return header

    def generate_htk_layer(layer, label, nb_frames, source):
        # Bias vector
        layer_def = nndef.generate_htk_vector_def(layer['biases'], label)
        # Weight matrix
        matrix_def = nndef.generate_htk_matrix_def(layer['weights'], label)
        layer_def.extend(matrix_def)
        # Input features
        nb_features = int(layer['weights'].get_shape()[0].value / nb_frames)
        feat_def = nndef.generate_htk_feat_def(
            nb_features, nb_frames, source, label)
        layer_def.extend(feat_def)
        # Layer definition
        activation_type = layer['activation'].type
        layer_def.extend(nndef.generate_htk_layer_def(activation_type, label))
        return layer_def

    def generate_htk_vector_def(vector, label):
        vector_def = ["~V \"layer{}_bias\"".format(label)]
        vector_def.append("<VECTOR> {}".format(vector.get_shape()[0].value))
        nparray = vector.eval()
        vector_def.append(' '.join(nndef.fformat.format(v) for v in nparray))
        return vector_def

    def generate_htk_matrix_def(matrix, label):
        matrix_def = ["~M \"layer{}_weight\"".format(label)]
        height = matrix.get_shape()[0].value
        width = matrix.get_shape()[1].value
        matrix_def.append("<MATRIX> {} {}".format(width, height))
        nparray = matrix.eval().T
        matrix_def.extend([' '.join(nndef.fformat.format(v) for v in rows) for rows in nparray])
        return matrix_def

    def generate_htk_feat_def(nb_features, nb_frames, source, label):
        feat_def = ["~F \"layer{}_in_feamix\"".format(label)]
        feat_def.append("<NUMFEATURES> 1 {}".format(nb_features * nb_frames))
        feat_def.append("<FEATURE> 1 {}".format(nb_features))
        feat_def.append("<SOURCE>")
        feat_def.append(source)
        feat_def.append("<CONTEXTSHIFT> {}".format(nb_frames))
        delta = int(nb_frames / 2)
        context = ' '.join([str(i) for i in range(-delta, delta + 1)])
        feat_def.append(context)
        return feat_def

    def generate_htk_layer_def(activation_type, label):
        layer_def = ["~L \"layer{}\"".format(label)]
        layer_def.append("<BEGINLAYER>")
        layer_def.append("<LAYERKIND> \"PERCEPTRON\"")
        layer_def.append("<INPUTFEATURE>")
        layer_def.append("~F \"layer{}_in_feamix\"".format(label))
        layer_def.append("<WEIGHT>")
        layer_def.append("~M \"layer{}_weight\"".format(label))
        layer_def.append("<BIAS>")
        layer_def.append("~V \"layer{}_bias\"".format(label))
        layer_def.append("<ACTIVATION> \"{}\"".format(activation_type.upper()))
        layer_def.append("<ENDLAYER>")
        return layer_def

    def generate_htk_net_def(labels):
        net_def = ["~N \"DNN1\""]
        net_def.append("<BEGINANN>")
        net_def.append("<NUMLAYERS> {}".format(len(labels) + 1))
        i = 2
        for l in labels:
            net_def.append("<LAYER> {}".format(i))
            net_def.append("~L \"layer{}\"".format(l))
            i +=1
        net_def.append("<ENDANN>")
        return net_def
