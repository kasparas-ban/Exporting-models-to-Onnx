# Exporting-models-to-Onnx

We are trying to export a model created in PyTorch to Onnx. The model is created with function:
```
model = WideResNet(40, 2, Conv, BasicBlock)
```
and then, one convolutional block of that model is swapped with a different kind of block, like so:
```
model = update_block(block_number, model, block_type)
```
In the particular example that is provided in the code, we exchange the first block (block_number=0) with another block of type G2B2.
We try to export this updated model with
```
torch.onnx.export(model, torch_input, save_name)
```
but the following error occurs:
```
Traceback (most recent call last):
  File "export_model.py", line 42, in <module>
    torch.onnx.export(model, torch_input, save_name)   # <- GIVES ERROR
  File "/home/kasparas/.local/lib/python3.7/site-packages/torch/onnx/__init__.py", line 132, in export
    strip_doc_string, dynamic_axes)
  File "/home/kasparas/.local/lib/python3.7/site-packages/torch/onnx/utils.py", line 64, in export
    example_outputs=example_outputs, strip_doc_string=strip_doc_string, dynamic_axes=dynamic_axes)
  File "/home/kasparas/.local/lib/python3.7/site-packages/torch/onnx/utils.py", line 329, in _export
    _retain_param_name, do_constant_folding)
  File "/home/kasparas/.local/lib/python3.7/site-packages/torch/onnx/utils.py", line 213, in _model_to_graph
    graph, torch_out = _trace_and_get_graph_from_model(model, args, training)
  File "/home/kasparas/.local/lib/python3.7/site-packages/torch/onnx/utils.py", line 174, in _trace_and_get_graph_from_model
    raise RuntimeError("state_dict changed after running the tracer; "
RuntimeError: state_dict changed after running the tracer; something weird is happening in your model!
```

For some reason PyTorch fails to export the model, that otherwise is fully functional.

If the blocks are not swapped, the network is exported successfully.

# To reproduce the error

To get the same error, run `export_model.py` file.

# Environment

PyTorch Version : 1.2.0

OS: Ubuntu

Python version: 3.7.3
