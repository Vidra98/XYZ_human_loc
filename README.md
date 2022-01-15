
# Human localization for interaction with a mobile furniture

![Project resume image](README_img/project_graph.png)

As you can see the networks takes as input a video/image and provide as output a json file with position in pixels and relative/absolute depth depending on the scaling mode selected. Depending on the scaling mode, ground truth may be necessary. The ground truth can be given in a json files containing one depth image array per line. 
The code necessary for it's generation from a D435i camera is provided in acquisition folder.

# Requirement 

# Command

## From video files 

To run from video input, please put your input file in 

```
python3 -m Test --source=input/wheelchair2 \
--video-output=output/output.mp4 \
--video-fps=30 \
--json-output=output/json_output.json \
--depth_model='midas' \
--model_type='dpt_hybrid' \
--checkpoint mobilenetv2 \
--shift-scale \
--GT_depth_file=/input/data2.json
```

## From D435i camera 
