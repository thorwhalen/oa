"""oa utils"""

from importlib.resources import files
import os
import i2
import dol

import graze
from functools import partial
from config2py import get_config, ask_user_for_input, get_configs_local_store
import openai  # pip install openai (see https://pypi.org/project/openai/)


def get_package_name():
    """Return current package name"""
    # return __name__.split('.')[0]
    # TODO: See if this works in all cases where module is in highest level of package
    #  but meanwhile, hardcode it:
    return "oa"


# get app data dir path and ensure it exists
pkg_name = get_package_name()
data_files = files(pkg_name) / "data"
templates_files = data_files / "templates"
_root_app_data_dir = i2.get_app_data_folder()
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR",
    os.path.join(_root_app_data_dir, pkg_name),
)
app_data_dir = dol.ensure_dir(app_data_dir, verbose=f"Making app dir: {app_data_dir}")
djoin = partial(os.path.join, app_data_dir)

# _open_api_key_env_name = 'OPENAI_API_KEY'
# _api_key = os.environ.get(_open_api_key_env_name, None)
# if _api_key is None:
#     _api_key = getpass.getpass(
#         f"Please set your OpenAI API key and press enter to continue. "
#         f"I will put it in the environment variable {_open_api_key_env_name} "
#     )
# openai.api_key = _api_key


configs_local_store = get_configs_local_store(pkg_name)

openai.api_key = get_config(
    'OPENAI_API_KEY',
    sources=[
        # Try to find it in oa config
        configs_local_store,
        # Try to find it in os.environ (environmental variables)
        os.environ,
        # If not, ask the user to input it
        lambda k: ask_user_for_input(
            f"Please set your OpenAI API key and press enter to continue. "
            "If you don't have one, you can get one at "
            "https://platform.openai.com/account/api-keys. ",
            mask_input=True,
            masking_toggle_str='',
            egress=lambda v: configs_local_store.__setitem__(k, v),
        )
    ],
)


_grazed_dir = dol.ensure_dir(os.path.join(app_data_dir, 'grazed'))
grazed = graze.Graze(rootdir=_grazed_dir)












