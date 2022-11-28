
import torch

from animesr.archs.vsr_arch import MSRSWVSR

model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)

print(model)

model.load_state_dict(torch.load("D:\\repo\\AnimeSR\\weights\\AnimeSR_v2.pth"),strict=False)

input_names = ["input"]
output_names = ["output"]
f1=torch.rand((1,6,540,640))
x=(f1)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "weights/AnimeSR_v2.pth",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = input_names,   # the model's input names
                  output_names = output_names) #                  dynamic_axes={'input' : {3 : 'width', 2: 'height'}})

print('Successfully converted model to onnx')