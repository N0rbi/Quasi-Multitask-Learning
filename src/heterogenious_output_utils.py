# parses expressions that look like this:
# ({activ1} {unit_num1})x{num_out1} ({activ} {unit_num2})x{num_out2}
import re
import dynet
from lib.mnnl import FFSequencePredictor, Layer
from lib.mmappers import ACTIVATION_MAP
parse_exp = re.compile(r'(\(([a-z]*) ([0-9]*)\)x([0-9]*))')


def is_query_valid(query):
    return len(parse_exp.findall(query)) != 0


def query_to_dynet_builder(query):

    def output_generator(model, in_dim, out_dim):
        for layer in get_layer_params(query):
            mlp_activation, mlp = layer
            yield FFSequencePredictor(Layer(model, in_dim * 2, out_dim, dynet.softmax, mlp=int(mlp),
                                                 mlp_activation=ACTIVATION_MAP[mlp_activation]))
    return output_generator


def get_layer_params(query):
    output_list = parse_exp.findall(query)
    layers = []
    for output_type in output_list:
        mlp_activation = output_type[1]
        mlp = output_type[2]
        output_type_times = output_type[3]
        for _ in range(int(output_type_times)):
            layers.append((mlp_activation, mlp,))
    return layers

def get_output_number(query):
    return sum([int(output_type[3]) for output_type in parse_exp.findall(query)])