from blocks import *
from wide_resnet import *
import onnx
import torch
import pandas as pd

string_to_conv = {
    'Conv' : Conv,
    'DConvA2' : DConvA2,
    'DConvA4' : DConvA4,
    'DConvA8' :  DConvA8,
    'DConvA16' : DConvA16,
    'DConvG16' : DConvG16,
    'DConvG8' :  DConvG8,
    'DConvG4' :  DConvG4,
    'DConvG2' :  DConvG2,
    'DConv' :    DConv,
    'ConvB2' :   ConvB2,
    'ConvB4' :   ConvB4,
    'A2B2' :     A2B2,
    'A4B2' :     A4B2,
    'A8B2' :     A8B2,
    'A16B2' :    A16B2,
    'G16B2' :    G16B2,
    'G8B2' :     G8B2,
    'G4B2' :     G4B2,
    'G2B2' :     G2B2
}

def update_model(model, convs):
    convs = str(convs)[1:-1].rstrip().replace(',', '')
    cs = convs.replace('\n', ' ').split()
    prefix = len("'models.blocks.")
    suffix = len("'>")

    r = [w for w in cs if '<class' not in w]
    blocks = [c[prefix:-suffix] for c in r]
    convs  = [string_to_conv[c] for c in blocks]

    for i, c in enumerate(convs):
        model = update_block(i, model, c)

    return model, blocks

# --- Load a model -----------------------------------------------------------

directory = 'random_models.csv'
in_file = open(directory)
data = pd.read_csv(in_file)

net = WideResNet(40, 2, Conv, BasicBlock)
net, _ = update_model(net, data['convs'][0])

# --- Export PyTorch model to Onnx -------------------------------------------

onnx_models = dict()
torch_input = Variable(torch.rand(1, 3, 32, 32))

# export to onnx
save_name = 'test_model.onnx'
torch.onnx.export(net, torch_input, save_name)   # <- GIVES ERROR
