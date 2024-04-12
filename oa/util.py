"""oa utils"""

from importlib.resources import files
import os
from functools import partial, lru_cache
from typing import Mapping, Union

import i2
import dol
import graze
from config2py import (
    get_config,
    ask_user_for_input,
    get_configs_local_store,
    simple_config_getter,
    user_gettable,
)

import openai  # pip install openai (see https://pypi.org/project/openai/)


def get_package_name():
    """Return current package name"""
    # return __name__.split('.')[0]
    # TODO: See if this works in all cases where module is in highest level of package
    #  but meanwhile, hardcode it:
    return 'oa'


# get app data dir path and ensure it exists
pkg_name = get_package_name()
data_files = files(pkg_name) / 'data'
templates_files = data_files / 'templates'
_root_app_data_dir = i2.get_app_data_folder()
app_data_dir = os.environ.get(
    f'{pkg_name.upper()}_APP_DATA_DIR',
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir, verbose=f'Making app dir: {app_data_dir}')
djoin = partial(os.path.join, app_data_dir)

# By default will look for configs in environment variables, then in the files
# (on mac/linus, this would be in ~/.config/<pkg_name>/configs)
# then ask the user.
config_getter = simple_config_getter(pkg_name)

# _open_api_key_env_name = 'OPENAI_API_KEY'
# _api_key = os.environ.get(_open_api_key_env_name, None)
# if _api_key is None:
#     _api_key = getpass.getpass(
#         f"Please set your OpenAI API key and press enter to continue. "
#         f"I will put it in the environment variable {_open_api_key_env_name} "
#     )
# openai.api_key = _api_key


configs_local_store = get_configs_local_store(pkg_name)

_DFLT_CONFIGS = {
    'OPENAI_API_KEY_ENV_NAME': 'OPENAI_API_KEY',
    'OA_DFLT_TEMPLATES_SOURCE_ENV_NAME': 'OA_DFLT_TEMPLATES_SOURCE',
    'OA_DFLT_ENGINE': 'gpt-3.5-turbo-instruct',
    'OA_DFLT_MODEL': 'gpt-3.5-turbo',
}

# write the defaults to the local store, if key missing there
for k, v in _DFLT_CONFIGS.items():
    if k not in configs_local_store:
        configs_local_store[k] = v


config_sources = [
    configs_local_store,  # look in the local store
    os.environ,  # look in the environment variables
    user_gettable(
        configs_local_store
    ),  # ask the user (and save response in local store)
]

# The main config getter for this package
config_getter = get_config(sources=config_sources)


# Get the OPENAI_API_KEY_ENV_NAME and DFLT_TEMPLATES_SOURCE_ENV_NAME
OPENAI_API_KEY_ENV_NAME = config_getter('OPENAI_API_KEY_ENV_NAME')
DFLT_TEMPLATES_SOURCE_ENV_NAME = config_getter('OA_DFLT_TEMPLATES_SOURCE_ENV_NAME')

# TODO: Understand the model/engine thing better and merge defaults if possible
DFLT_ENGINE = config_getter('OA_DFLT_ENGINE')
DFLT_MODEL = config_getter('OA_DFLT_MODEL')


# Have a particular way to get this api key
@lru_cache
def get_api_key_from_config():
    return get_config(
        OPENAI_API_KEY_ENV_NAME,
        sources=[
            # Try to find it in oa config
            configs_local_store,
            # Try to find it in os.environ (environmental variables)
            os.environ,
            # If not, ask the user to input it
            lambda k: ask_user_for_input(
                f'Please set your OpenAI API key and press enter to continue. '
                "If you don't have one, you can get one at "
                'https://platform.openai.com/account/api-keys. ',
                mask_input=True,
                masking_toggle_str='',
                egress=lambda v: configs_local_store.__setitem__(k, v),
            ),
        ],
    )


openai.api_key = get_api_key_from_config()


@lru_cache
def mk_client(api_key=None, **client_kwargs):
    api_key = api_key or get_api_key_from_config()
    return openai.OpenAI(api_key=api_key, **client_kwargs)


_grazed_dir = dol.ensure_dir(os.path.join(app_data_dir, 'grazed'))
grazed = graze.Graze(rootdir=_grazed_dir)


chatgpt_templates_dir = os.path.join(templates_files, "chatgpt")

DFLT_TEMPLATES_SOURCE = get_config(
    DFLT_TEMPLATES_SOURCE_ENV_NAME,
    sources=[os.environ],
    default=f"{chatgpt_templates_dir}",
)


# TODO: This is general: Bring this in dol or dolx
def _extract_folder_and_suffixes(
    string: str, default_suffixes=(), *, default_folder='', root_sep=':', suffix_sep=','
):
    root_folder, *suffixes = string.split(root_sep)
    if root_folder == "":
        root_folder = default_folder
    if len(suffixes) == 0:
        suffixes = default_suffixes
    elif len(suffixes) == 1:
        suffixes = suffixes[0].split(suffix_sep)
    else:
        raise ValueError(
            f"template_store must be a path to a directory of templates, "
            f"optionally followed by a colon and a list of file suffixes to use"
        )
    return root_folder, suffixes


def mk_template_store(template_store: Union[Mapping, str]):
    if isinstance(template_store, Mapping):
        return template_store
    elif isinstance(template_store, str):
        root_folder, suffixes = _extract_folder_and_suffixes(template_store)
        suffix_filter = dol.filt_iter.suffixes(suffixes)
        return suffix_filter(dol.TextFiles(root_folder))
    else:
        raise TypeError(
            f"template_store must be a Mapping or a path to a directory of templates"
        )
