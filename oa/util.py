"""oa utils"""

# from importlib.resources import files
import os
import i2
import dol
import re
import getpass

import graze
from functools import partial

import openai  # pip install openai (see https://pypi.org/project/openai/)


def get_package_name():
    """Return current package name"""
    # return __name__.split('.')[0]
    # TODO: See if this works in all cases where module is in highest level of package
    #  but meanwhile, hardcode it:
    return "oai"


# get app data dir path and ensure it exists
pkg_name = get_package_name()
_root_app_data_dir = i2.get_app_data_folder()
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR",
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir, verbose=f"Making app dir: {app_data_dir}")
djoin = partial(os.path.join, app_data_dir)

_open_api_key_env_name = 'OPENAI_API_KEY'
_api_key = os.environ.get(_open_api_key_env_name, None)
if _api_key is None:
    # TODO: Figure out a way to make input response invisible (using * or something)
    _api_key = getpass.getpass(
        f"Please set your OpenAI API key and press enter to continue."
        f"I will put it in the environment variable {_open_api_key_env_name} "
    )
openai.api_key = _api_key


_grazed_dir = dol.ensure_dir(os.path.join(app_data_dir, 'grazed'))
grazed = graze.Graze(rootdir=_grazed_dir)


def normalized_file_name(prompt: str) -> str:
    """Convert prompt to a normalized valid file/folder name

    >>> normalized_file_name("This is a prompt")
    'this is a prompt'
    >>> normalized_file_name("This is: a PROMPT!  (with punctuation)")
    'this is a prompt with punctuation'
    """
    return re.sub(r"\W+", " ", prompt).lower()


def prompt_path(prompt, prefix=""):
    filepath = os.path.join(prefix, normalized_file_name(prompt))
    return dol.ensure_dir(filepath)


prompt_image_path = partial(prompt_path, prefix=djoin("dalle"))


# def dalle(
#         prompt, n=2, size="512x512", **image_create_params
# ):
#     image_resp = openai.Image.create(
#         prompt=prompt, n=n, size=size, **image_create_params
#     )
#
#
#     folder_name = prompt_to_folder_name(prompt)
#     prompt_folder = dol.ensure_dir(prompt_to_folder(prompt))
#     prompt_folder
