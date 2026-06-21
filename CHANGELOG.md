# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/);
each section corresponds to a git version tag (which is also the release
published to PyPI). Entries are commit subjects and PR titles, verbatim.

## [0.1.50] - 2026-06-17

### Added

- feat(tools): make infer_schema_from_verbal_description backend-injectable ([#14](https://github.com/thorwhalen/oa/pull/14))

## [0.1.49] - 2026-05-16

- ci: wire OPENAI_API_KEY via [tool.wads.ci.env.test_envvars]

## [0.1.48] - 2026-05-16

- ci(workflow): wire OPENAI_API_KEY secret into validation jobs
- test: root conftest sets dummy OPENAI_API_KEY before collection
- ci(wads): fix env defaults section structure
- ci(wads): set dummy OPENAI_API_KEY env default
- ci: migrate Modern -> uv (wads-migrate ci-to-uv)

## [0.1.47] - 2026-05-15

- test(pytest): ignore opt-in modules during collection
- Clean up formatting in pyproject.toml for consistency and readability
- Refactor notebook to replace 'ten_constrained_answers_counts' with 'poll' for consistency and clarity; remove outdated PDF file.
- 0.1.46: oa - constraining AI answers (notebook)
- 0.1.45:
- Add constrained_answer function and update tools module
- Add 'dill' to the requirements in setup.cfg
- Add OPENAI_API_KEY to CI workflow
- modernization
- chores

### Added

- feat: Add OpenAI Audio tools for transcription and text-to-speech functionalities

### Changed

- refactor: Update type hints to use built-in collection types and improve code consistency
- refactor: Rename get_app_data_folder to get_app_config_folder

### Fixed

- fix(deps): add pydantic and tabled to core deps
- fix(deps): add numpy and pandas to core deps
- fix(build): pack oa package in wheel (was packing only data); drop legacy setup.py
- fix: Update prompt_json_function to assign json_schema to the returned function

## [0.1.44] - 2025-10-04

- minor changes

### Fixed

- fix: update search function to correctly extract file IDs from response output
- fix: update search function to use Responses API for improved document retrieval
- fix: search: The previous vector store search implementation was failing semantic query tests because it used a low-level vector search API that was unable to correctly interpret the intent of complex natural language questions. This led to inaccurate results, such as returning documents unrelated to the query's semantic meaning. To fix this, the search functionality was re-implemented to leverage the OpenAI Assistants API. The new approach creates a temporary assistant with the file_search tool, which uses a powerful model to better understand the user's intent, retrieve relevant file citations, and provide significantly more accurate semantic search results.
- fix: comment out search function test case for CI compatibility

## [0.1.43] - 2025-09-29

- commented out the doctests of chats.py

### Fixed

- fix: comment out search function test case for CI compatibility
- fix: add beautifulsoup4 to install_requires in setup.cfg
- fix: repaired the chat.py (ChatDacc) stuff (which broke because of chatGPT html changes
- fix: Import Pipe in chats module

## [0.1.42] - 2025-08-13

- Small readme change to trigger CI.

## [0.1.41] - 2025-08-13

- Maintenance only (all commits in this range were CI version bumps / housekeeping).

## [0.1.40] - 2025-07-21

- Refactor vector store functionality and add search capabilities

### Added

- feat: enhance docs_to_vector_store and mk_search_func_for_oa_vector_store with client parameter and improved file handling

### Fixed

- fix: correct import statement for OaFiles and fix syntax error in docs_to_vector_store

## [0.1.39] - 2025-07-01

### Changed

- refactor: kwargs_from_args_and_kwargs -> map_arguments & args_and_kwargs_from_kwargs -> mk_args_and_kwargs

## [0.1.38] - 2025-06-24

- pytest ignoring _resources.py instead of _resources

### Added

- feat: Enhance embeddings function to support extra parameters when texts is None
- feat: Allow optional texts parameter in embeddings function to return a partial function

## [0.1.37] - 2025-04-04

### Changed

- refactor: rmoved verbose logging of ensure_dir when mapping app_data_dir because then doctests would fail (and no non-hacky way to ignore output while still running line (see https://stackoverflow.com/questions/1024411/can-python-doctest-ignore-some-output-lines))

## [0.1.36] - 2025-03-29

### Fixed

- fix: Ensure unique names are preserved in string_format_embodier and prompt_function

## [0.1.35] - 2025-03-27

### Added

- feat: Add model_information_dict to the oa module imports

## [0.1.34] - 2025-03-19

- Update ci.yml
- Update ci.yml
- chore: working on resource tools to get API pricing info automatically

### Added

- feat: A _resources module with tools to make SSOT information on openai (e.g. model information and prices)

### Fixed

- fix: doctest

## [0.1.33] - 2025-03-14

### Added

- feat: add optional regex filtering to list_engine_ids function

## [0.1.32] - 2025-03-11

### Fixed

- fix: streamline validation_vector handling in validate_texts_for_embeddings

## [0.1.31] - 2025-02-23

- gitattributes

### Added

- feat: infer_schema_from_verbal_description

### Fixed

- fix: Commenting out ChatDacc tests -- broken because html structure changed.

## [0.1.30] - 2025-01-20

### Added

- feat: parsing out urls from chats
- feat: enhance ChatDacc with new extract_turns and find_url_keys to find paths with urls.

### Docs

- docs: update README with ChatDacc usage instructions and features overview

## [0.1.29] - 2025-01-17

### Added

- feat: update prompt function to use map_arguments and mk_args_and_kwargs for improved argument handling
- feat: enhance ChatDacc with improved content handling and new merged turn data function
- feat: add ChatDacc import to oa module
- feat: chats module

### Fixed

- fix: doctests

## [0.1.28] - 2024-12-13

- chore: add ju to deps
- misc: add keywords to num_tokens
- comment on embeddings
- doc: add ingress arg description
- 0.1.22:
- 0.1.21:
- 0.1.20:
- 0.1.19:
- merge
- 0.1.18:
- chore: add deps and project urls
- chore: write tests and refactor embeddings

### Added

- feat: embeddings batches requests by default, plus more control options (egress, batch_callback)
- feat: chunk_iterable (chunker that handles dicts the way we'd want)
- feat: enhance batch processing with new utilities and improved JSON schema fix: doctests
- feat: Parsing out information example in notebook
- feat: add control over model in prompt_json_function
- feat: prompt_json_function more robust and easier to use
- feat: add generic_json_schema
- feat: prompt_json_function
- feat: add chat_models info
- feat: batch functionality and openai stores
- feat: integrate signature forwarding and response_format
- feat: mk_batch_file_embeddings_task and mk_embeddings_batch_file
- feat: utc_to_human
- feat: add batch model info to embeddings_models (info dict)
- feat: add model_information_dict as attr of model_information function
- feat: sentiment_analysis
- feat: add dimensions arg to oa.embeddings
- feat: 0.1.17 pub

### Changed

- refactor: chat model arg now keyword only

### Fixed

- fix: attempt to fix import error
- fix: import error
- fix: utc_int_to_iso_date and iso_date_to_utc_int for timezone insensitivity
- fix: A missing variable

### Docs

- docs: mk_batch_file_embeddings_task
- docs: some docs error
- docs: add readme content

## [0.1.17] - 2024-04-19

- doc: comments on embeddings

### Added

- feat: model_information, compute_price, text_is_valid
- feat: num_tokens in root with global default
- feat: num_tokens (using tiktoken)
- feat: notebook

### Fixed

- fix: openai API yaml API specs source
- fix: model_information_dict

## [0.1.15] - 2024-04-12

### Added

- feat: add ingress to prompt_function
- feat: stripping all configs of config_getter
- feat: change way configs are sourced to enable more easy parametrization
- feat: two new prompt templates and a prompt templating demo notebook
- feat: edit setup.py

### Fixed

- fix: stripping
- fix: stripping of configs

## [0.1.13] - 2024-04-11

### Added

- feat: issue [#7](https://github.com/thorwhalen/oa/pull/7) resolution, but actually working and with doctest
- feat: be able to escape placeholder markers -- closes [#7](https://github.com/thorwhalen/oa/pull/7)

## [0.1.11] - 2024-03-15

### Fixed

- fix: 0.1.11

## [0.1.10] - 2024-03-15

- Update setup.cfg

## [0.1.8] - 2024-02-26

### Fixed

- fix: ci

## [0.1.7] - 2024-02-23

- 0.1.7:
- 0.1.6:
- Update setup.cfg
- Merge branches 'main' and 'main' of github.com:thorwhalen/oa

### Added

- feat: prompt_func has option to return prompt without executing and add name, doc, module & egress
- feat: embeddings and cosine_similarity
- feat: add notes on my migration in notebook

### Fixed

- fix: skipping swagger parsing failure (since openai update)
- fix: all those things that OpenAI's update broke
- fix: MANIFEST.in

## [0.1.3] - 2024-01-10

- Create MANIFEST.in
- 0.1.3:
- Update ci.yml
- Update ci.yml
- Update setup.cfg
- 0.0.15:
- Update ci.yml
- ci
- Update README.md [bump minor]
- Update ci.yml
- Update openai_specs.py
- Update openai_specs.py
- Update openai_specs.py
- Update ci.yml
- Update setup.cfg
- Update ci.yml
- Update setup.cfg
- Update setup.cfg
- Update ci.yml
- CI
- 0.0.13: repaired a bug, made less dependent on third-party, and added synopsis template
- 0.0.12: feat: added some docs
- 0.0.11: feat: PromptFuncs and ask.ai
- 0.0.10: feat: prompt_function
- 0.0.9:
- chore: removed ci
- misc
- 0.0.8: chore: doc edits
- 0.0.7: feat: raw and facades
- 0.0.6: construction elements

### Added

- feat: user control over default template store
- feat: new templates for i2i
- feat: template funcs 1st arg be PK
- feat: templated_function
- feat: Timer
- feat: misc
- feat: more control in interface funcs for story illustration

### Fixed

- fix: remove failing "Tag Repository With New Version Number" CI section
- fix: add package data
- fix: CI (2)
- fix: CI
