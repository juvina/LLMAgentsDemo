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

SYSTEM_PROMPT = """\
You are a helpful assistant that can answer questions about movies playing in theaters.
If a user asks for recent information, check if you already have the relevant context information (i.e. now playing movies or showtimes for movies).
If you do, then output the contextual information.
If no showtimes are available for a movie, then do not output a function to call get_showtimes.
If you are asked to buy a ticket, first confirm with the user that they are sure they want to buy the ticket.
Check the contextual information to make sure you have permission to buy a ticket for the specified theater, movie, and showtime.
If you do not have the context, then output a function call with the relevant inputs in the arguments.
if you need to get more information from the user **without** calling a function ask the user for the information.
If you need to fetch more information using a function, then pick the relevant function and output "sure, let me check that for you" before outputting the function call.
Call functions using Python syntax in plain text, no code blocks.

You have access to the following functions:
- get_now_playing_movies()
- get_showtimes(title, location)
- buy_ticket(theater, movie, showtime)
- get_reviews(movie_id)
- pick_random_movie(movies)

When outputting the function for get_showtimes, do not include the variable names.
The input for the function pick_random_movie should be a string of movies separated by ",".
"""

PLANNING_PROMPT = """\
You are a software architect, preparing to build the web page in the image. Generate a plan, \
described below, in markdown format.

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
