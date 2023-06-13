# -*- coding: utf-8 -*-
import gradio as gr
import numpy as np


# pylint: disable=W0613


def sketch_recognition(input_sketch: np.ndarray) -> dict:
    """
    Takes in the user input, which in this case is a sketched image, and returns the prediction.

    Args:
        input_sketch: the input image as a numpy array

    Returns:
        confidences: the top five predictions, as a dictionary whose keys are class labels
                     and whose values are confidence probabilities
    """
    confidences = {
        "laptop": 0.1,
        "cup of coffee": 0.3,
        "ghost": 0.2,
        "gift": 0.2,
        "other": 0.2,
    }
    return confidences


ARTICLE_OPENING = """
<center> 
The bot was trained to recognize your sketch of a laptop, a cup of coffee, a ghost or a gift. Have fun drawing and 
try to fool it!
</center> 
"""

ARTICLE_ENDING = """
<center> 
Check out my <a href="https://www.nelsonantunes.com">blog</a> where I provide more context about this bot.
</center>
"""

sketch_pad = gr.Sketchpad(shape=(512, 512), brush_radius=24, label="Draw on sketch pad")

demo = gr.Interface(
    fn=sketch_recognition,
    inputs=sketch_pad,
    outputs="label",
    title="Sketch Recognition System",
    description=ARTICLE_OPENING,
    theme=gr.themes.Soft(),
    article=ARTICLE_ENDING,
    flagging_options=["👍 Correct", "👎 Incorrect"],
)

demo.queue()
demo.launch(server_name="0.0.0.0")
