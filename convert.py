import argparse
import torch
import os
from arch.vsr_arch import MSRSWVSR

my_parser = argparse.ArgumentParser(description=" ")
my_parser.add_argument("--input", metavar="--input", type=str, help="input model",default="weights/AnimeSR_v2.pth")
my_parser.add_argument("--output", metavar="--output", type=str, help="output model")
my_parser.add_argument("--height", metavar="--height", type=int, help="height")
my_parser.add_argument("--width", metavar="--width", type=int, help="width")
my_parser.add_argument("--trtexec", action="store_true", help="trtexec")

args = my_parser.parse_args()


model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)


model.load_state_dict(torch.load(args.input), strict=False)
input_names = ["input"]
output_names = ["output"]
f1 = torch.rand((1, 6, args.height, args.width))
x = f1
print(model(f1).shape)
torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    "animesr-temp.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=16,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=input_names,  # the model's input names
    output_names=output_names,
    dynamic_axes={'input' : {3 : 'width', 2: 'height'}} )#
del model
os.system(f"python3 -m onnxsim animesr-temp.onnx {args.output}.onnx")
if args.trtexec:
    os.system(
        f" trtexec --onnx={args.output}.onnx --optShapes=input:1x6x{args.height}x{args.width} --saveEngine={args.output}.engine"
    )