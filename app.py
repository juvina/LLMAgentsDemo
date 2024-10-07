import random
from dotenv import load_dotenv
import chainlit as cl
import openai
import base64
from agents.base_agent import Agent



load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable
client = wrap_openai(openai.AsyncClient())


gen_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 500
}

function_keywords = [
    'get_now_playing_movies(',
    'get_showtimes(',
    'buy_ticket(',
    'get_reviews(',
    'pick_random_movie(',
]

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image that the user sends. 
Once they send an image, generate a plan as described below in markdown format. Ask the user if the plan looks good to them.

If the user or reviewer confirms the plan is good, available tools to save it as an artifact \
called `plan.md`. If the user has feedback on the plan, revise the plan, and save it using \
the tool again. A tool is available to update the artifact. Your role is only to plan the \
project. You will not implement the plan, and will not write any code.

If the plan has already been saved, no need to save it again unless there is feedback. Do not \
use the tool again if there are no changes.

For the contents of the markdown-formatted plan, create two sections, "Overview" and "Milestones".

In a section labeled "Overview", analyze the image, and describe the elements on the page, \
their positions, and the layout of the major sections.

Using vanilla HTML and CSS, discuss anything about the layout that might have different \
options for implementation. Review pros/cons, and recommend a course of action.

In a section labeled "Milestones", describe an ordered set of milestones for methodically \
building the web page, so that errors can be detected and corrected early. Pay close attention \
to the aligment of elements, and describe clear expectations in each milestone. Do not include \
testing milestones, just implementation.

Milestones should be formatted like this:

 - [ ] 1. This is the first milestone
 - [ ] 2. This is the second milestone
 - [ ] 3. This is the third milestone
"""

# Create an instance of the Agent class
planning_agent = Agent(name="Planning Agent", client=client, prompt=PLANNING_PROMPT)


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])

    # Processing images exclusively
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read the first image and encode it to base64
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        message_history.append({"role": "user", "content": message.content})
    
    response_message = await planning_agent.execute(message_history)

    message_history.append({"role": "assistant", "content": response_message})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
