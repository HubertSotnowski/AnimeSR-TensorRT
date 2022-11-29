import vapoursynth as vs
import sys
import os
import os.path
core = vs.core
core.num_threads = 8
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
clip =  core.lsmas.LWLibavSource(r'testsrc.mp4')
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip_pos1 = clip[2:]
clip_pos2 = clip.std.Trim(first=1,last=clip.num_frames-2)
clip_pos3 = clip.std.Trim(first=0,last=clip.num_frames-3)
print(clip_pos1.num_frames)
output = core.trt.Model([clip_pos3,clip_pos2,clip_pos1], engine_path=f"2.engine",num_streams=1)

output = vs.core.resize.Bicubic(output, format=vs.YUV420P8, matrix_s="709")
output.set_output()