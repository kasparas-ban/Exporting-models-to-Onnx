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

# --- Create a model -----------------------------------------------------------

model = WideResNet(40, 2, Conv, BasicBlock)

# Take a random block (block 0 in this case), and replace it
# with any other block (G2B2 in this case)
model = update_block(0, model, string_to_conv['G2B2'])

# --- Export PyTorch model to Onnx -------------------------------------------

torch_input = Variable(torch.rand(1, 3, 32, 32))
save_name = 'test_model.onnx'
torch.onnx.export(model, torch_input, save_name)   # <- GIVES ERROR
