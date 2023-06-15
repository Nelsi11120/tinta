# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np
from model import sketch_recognizer
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# pylint: disable=W0613

LABELS = Path("labels.txt").read_text().splitlines()


def sketch_recognition(input_sketch: np.ndarray) -> dict:
    """
    Takes in the user input, which in this case is a sketched image, and returns the prediction.

    Args:
        input_sketch: the input image as a numpy array

    Returns:
        confidences: the top five predictions, as a dictionary whose keys are class labels
                     and whose values are confidence probabilities
    """
    input_sketch = Image.fromarray(input_sketch).resize((28, 28))
    input_sketch = T.ToTensor()(input_sketch)

    with torch.no_grad():
        out = sketch_recognizer(input_sketch.unsqueeze(0))

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, k=out.shape[-1])

    confidences = {LABELS[i]: v.item() for i, v in zip(indices, values)}

    return confidences


ARTICLE_OPENING = """
<center> 
Ghostee the ghost doesn't know hunger, but he has been trained to recognize your sketch of what he thinks
is the best food: pizza, hamburger, hot-dog, donut, or... broccoli! Have fun drawing and helping Ghostee.
</center> 
"""

ARTICLE_ENDING = """
<center> 
Check out my <a href="https://www.nelsonantunes.com">blog</a> where I provide more context about this project, 
or the dedicated <a href="https://github.com/Nelsi11120/tinta">GitHub</a> repository to get the code.
</center>
"""

sketch_pad = gr.Sketchpad(
    shape=(1024, 1024), brush_radius=24, label="Draw on sketch pad"
)

demo = gr.Interface(
    fn=sketch_recognition,
    inputs=sketch_pad,
    outputs="label",
    title="Sketch Recognition System",
    description=ARTICLE_OPENING,
    theme=gr.themes.Soft(),
    article=ARTICLE_ENDING,
    flagging_options=["👍", "👎"],
)

demo.queue()
demo.launch(server_name="0.0.0.0")
