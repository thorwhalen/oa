"""Illustrate stories with OpenAI's DALL-E model."""

import html

import oa
from i2 import postprocess

# TODO: Protect item gets (obj[k]) from KeyError/IndexError. Raise meaningful error.

# @code_to_dag
# def make_children_story():
#     story_text = make_it_rhyming(story)
#     image = get_illustration(story_text, image_style)
#     page = aggregate_story_and_image(image, rhyming_story)

make_it_rhyming_prompt = """Act as a children's book author. 
Write a rhyming story about the following text:
###
{story}"""

illustrate_prompt = """Act as an illustrator, expert in the style: {image_style}.
Describe an image that would illustrate the following text:
###
{text}
"""

dalle_prompt = """Image style: {image_style}
{image_description}
"""

topic_points_prompt = """
I will give you a topic/subject and you will list {n_talking_points} talking points of 
the main ideas/subtopics/points of this topic.
Each talking point should be between {min_n_words} and {max_n_words} words. 

Your answer should be in the form of a bullet point list, 
and nothing but a bullet point list. Each bullet point contains the talking point only.

The topic is: 

{topic}
"""

topic_points_json_prompt = """
I will give you a topic/subject and you will list {n_talking_points} of the main 
ideas/subtopics/points of this topic, 
including a`title` and {min_n_words} to {max_n_words} word `description` of the topic. 

Your answer should be in the form of a valid JSON string, and nothing but a valid JSON.
The JSON should have the format `{{title: description, title: description, ...}}`

The topic is: 

{topic}
"""

DFLT_IMAGE_STYLE = "drawing"


def extract_first_text_choice(response) -> str:
    return response.choices[0].text.strip()


# TODO: Put back min_n_words and max_n_words as arguments once code_to_dag supports
#  more control over function injection (such as only taking the subset of arguments
#  mentioned by the FuncNodes, and auto-editing FuncNode binds.
@postprocess(extract_first_text_choice)
def topic_points(topic, n_talking_points=3) -> str:
    print(f"topic_points: topic={topic}, n_talking_points={n_talking_points}")
    min_n_words = 5
    max_n_words = 20
    prompt = topic_points_prompt.format(
        topic=topic,
        n_talking_points=n_talking_points,
        min_n_words=min_n_words,
        max_n_words=max_n_words,
    )
    return oa.complete(prompt, max_tokens=2048, n=1, engine="text-davinci-003")


def _repair_json(json_str):
    t = json_str
    if t.startswith("`"):
        t = t[1:]
    if t.endswith("`"):
        t = t[:-1]
    return t


def topic_points_json(topic, n_talking_points=3, min_n_words=15, max_n_words=40) -> str:
    prompt = topic_points_prompt.format(
        topic=topic,
        n_talking_points=n_talking_points,
        min_n_words=min_n_words,
        max_n_words=max_n_words,
    )
    t = oa.complete(prompt, max_tokens=2048, n=1, engine="text-davinci-003")
    t = extract_first_text_choice(t)
    return _repair_json(t)


# @postprocess(extract_first_text_choice)
def make_it_rhyming(story, *, max_tokens=512, **chat_param) -> str:
    prompt = make_it_rhyming_prompt.format(story=story)
    return oa.chat(prompt, max_tokens=max_tokens, n=1, **chat_param)


# @postprocess(extract_first_text_choice)
def get_image_description(
    story_text: str, image_style=DFLT_IMAGE_STYLE, max_tokens=256, **chat_param
) -> str:
    prompt = illustrate_prompt.format(text=story_text, image_style=image_style)
    return oa.chat(prompt, max_tokens=max_tokens, n=1, **chat_param)


def get_image_url(image_description, image_style=DFLT_IMAGE_STYLE):
    prompt = dalle_prompt.format(
        image_description=image_description, image_style=image_style
    )
    return oa.dalle(prompt, n=1)


def get_illustration(story_text: str, image_style=DFLT_IMAGE_STYLE):
    image_description = get_image_description(story_text, image_style)
    url = get_image_url(image_description, image_style)
    return url


def _format_for_html_display(input_string: str) -> str:
    escaped_string = html.escape(input_string)
    newline_replaced_string = escaped_string.replace("\n", "<br>")
    html_formatted_string = f"<p>{newline_replaced_string}</p>"
    return html_formatted_string


def aggregate_story_and_image(image_url, story_text):
    """Produces an html page with the image and story text"""

    story_text = _format_for_html_display(story_text)
    html = f"""<html>
    <body>
    <img src="{image_url}" />
    <p>{story_text}</p>
    </body>
    </html>"""

    return html
