"""Batch functionality """

from typing import Optional, Union, Callable, List
from functools import partial

from oa.util import batch_endpoints
from oa.base import (
    _prepare_embeddings_args,
    _raise_if_any_invalid,
    DFLT_EMBEDDINGS_MODEL,
    NOT_GIVEN,
    TextOrTexts,
    mk_client,
)

# --------------------------------------------------------------------------------------
# useful information

batch_field_descriptions = {
    "id": "The unique identifier of the batch.",
    "status": "The status of the batch. See the `status_enum_descriptions` for possible values.",
    "input_file_id": "The ID of the file that contains the batch's input data.",
    "output_file_id": "The ID of the file that contains the batch's output data, if the batch completes successfully.",
    "created_at": "A timestamp of when the batch was created.",
    "completed_at": "A timestamp of when the batch was completed.",
    "failed_at": "A timestamp of when the batch failed.",
    "cancelled_at": "A timestamp of when the batch was cancelled.",
    "in_progress_at": "A timestamp of when the batch started processing.",
    "finalizing_at": "A timestamp of when the batch entered the finalizing stage.",
    "expired_at": "A timestamp of when the batch expired, meaning it did not complete within the time window.",
    "expires_at": "The time at which the batch will expire if not completed.",
    "error_file_id": "The ID of the file that contains detailed error messages if the batch fails.",
    "request_counts": "Contains counts of the total, completed, and failed requests in the batch.",
    "errors": "An array containing errors related to individual requests within the batch, if any exist.",
    "completion_window": "The maximum time window allowed for the batch to complete.",
}

status_enum_descriptions = {
    "created": "The batch has been created but has not started processing yet. See the `created_at` field for when it was created.",
    "in_progress": "The batch is currently being processed. See the `in_progress_at` field for when processing started.",
    "failed": "The batch encountered an error during processing. See the `failed_at` field for when it failed, and the `error_file_id` for details on the failure.",
    "completed": "The batch has successfully completed processing. See the `completed_at` field for when it was completed, and `output_file_id` for the output file.",
    "cancelled": "The batch was cancelled before completion. See the `cancelled_at` field for when it was cancelled.",
    "finalizing": "The batch is finalizing its results. See the `finalizing_at` field for when it entered this stage.",
    "expired": "The batch exceeded its completion window and was terminated. See the `expired_at` field for when it expired.",
    "failed_partially": "Some requests in the batch failed while others succeeded. See `errors` for the failed requests, and `output_file_id` for any partial output.",
}

# --------------------------------------------------------------------------------------
# batch embeddings utils

import time
from dataclasses import dataclass


def random_custom_id(prefix='custom_id-', suffix=''):
    """Make a random custom_id by using the current time in nanoseconds"""
    return f"{prefix}{int(time.time() * 1e9)}{suffix}"


# @dataclass
# class EmbeddingsMaker:
#     texts: TextOrTexts,

#     custom_id: str = None,
#     validate: Optional[Union[bool, Callable]] = True,
#     valid_text_getter=_raise_if_any_invalid,
#     model=DFLT_EMBEDDINGS_MODEL,
#     client=None,
#     dimensions: Optional[int] = NOT_GIVEN,
#     **extra_embeddings_params,


def _rm_not_given_values(d):
    return {k: v for k, v in d.items() if v is not NOT_GIVEN}


def _mk_embeddings_request_body(
    text_or_texts,
    model=DFLT_EMBEDDINGS_MODEL,
    dimensions: Optional[int] = NOT_GIVEN,
    **extra_embeddings_params,
):
    return _rm_not_given_values(
        dict(
            input=text_or_texts,
            model=model,
            dimensions=dimensions,
            **extra_embeddings_params,
        )
    )


def _mk_task_request_dict(
    body, custom_id=None, *, endpoint=DFLT_EMBEDDINGS_MODEL, method='POST'
):

    if custom_id is None:
        custom_id = random_custom_id('embeddings_batch_id-')

    return {
        "custom_id": custom_id,
        "method": method,
        "url": endpoint,
        "body": body,
    }


def mk_batch_file_embeddings_task(
    texts: TextOrTexts,
    *,
    custom_id: Optional[str] = None,
    validate: Optional[Union[bool, Callable]] = True,
    valid_text_getter=_raise_if_any_invalid,
    # client=None,
    model=DFLT_EMBEDDINGS_MODEL,
    dimensions: Optional[int] = NOT_GIVEN,
    **extra_embeddings_params,
) -> Union[dict, List[dict]]:
    texts, texts_type, keys = _prepare_embeddings_args(
        validate, texts, valid_text_getter, model
    )

    _body = partial(
        _mk_embeddings_request_body,
        model=model,
        dimensions=dimensions,
        **extra_embeddings_params,
    )
    _task = partial(
        _mk_task_request_dict, custom_id=custom_id, endpoint=batch_endpoints.embeddings
    )

    if keys is not None:
        # return a list of tasks, using the keys as custom_ids
        return [_task(_body(text), custom_id=key) for key, text in zip(keys, texts)]
    else:
        return _task(_body(texts))


from operator import attrgetter
from lkj import value_in_interval
from dol import Pipe

create_at_within_range = value_in_interval(get_val=attrgetter('created_at'))


def batches_within_range(batches_base, min_date, max_date=None):
    return list(
        filter(create_at_within_range(min_val=min_date, max_val=max_date), batches_base)
    )


def request_counts(batch_list):
    t = pd.DataFrame([x.to_dict() for x in batch_list])
    tt = pd.DataFrame(t.request_counts.values.tolist())
    return tt.sum()


check_batch_requests = Pipe(batches_within_range, request_counts)


from oa.util import utc_int_to_iso_date
from datetime import timedelta


# Define custom error classes
# TODO: Rethink these. They're not really errors (except perhaps failed and canceled)
#   Perhaps some state object is more appropriate?
class BatchError(ValueError):  # Or RuntimeError? Or just Exception?
    pass


class BatchInProgressError(BatchError):
    pass


class BatchCancelledError(BatchError):
    pass


class BatchExpiredError(BatchError):
    pass


class BatchFinalizingError(BatchError):
    pass


class BatchFailedError(BatchError):
    pass


# TODO: I do NOT like the dependency on oa_stores here!
# TODO: Not sure if function or object with a "handle" __call__ method is better here
def get_output_file_data(batch: 'Batch', *, oa_stores):
    """
    Get the output file data for a batch, if it has completed successfully.

    """
    
    try:
        batch_obj = oa_stores.batches_base[batch]
    except KeyError:
        raise KeyError(f"Batch {batch} not found.")

    if batch_obj.status == 'completed':
        # Return the output file if the batch completed successfully
        return oa_stores.files_base[batch_obj.output_file_id]

    else:
        if batch_obj.status == 'failed':
            # Raise an error if the batch failed
            error_obj = BatchFailedError(
                f"Batch {batch} failed "
                f"at {utc_int_to_iso_date(batch_obj.failed_at)}. "
                f"Check out {batch_obj.error_file_id} for more information."
            )

        elif batch_obj.status == 'in_progress':
            # Calculate the time elapsed between creation and when it started processing
            time_elapsed = timedelta(
                seconds=(batch_obj.in_progress_at - batch_obj.created_at)
            )
            error_obj = BatchInProgressError(
                f"Batch {batch} is still in progress. "
                f"Started processing at {utc_int_to_iso_date(batch_obj.in_progress_at)}, "
                f"{time_elapsed.total_seconds() // 3600:.0f} hours and "
                f"{(time_elapsed.total_seconds() % 3600) // 60:.0f} minutes after it was created."
            )

        elif batch_obj.status == 'cancelled':
            # Provide information when the batch was cancelled
            error_obj = BatchCancelledError(
                f"Batch {batch} was cancelled "
                f"at {utc_int_to_iso_date(batch_obj.cancelled_at)}."
            )

        elif batch_obj.status == 'expired':
            # Notify that the batch expired and provide timestamps
            error_obj = BatchExpiredError(
                f"Batch {batch} expired at {utc_int_to_iso_date(batch_obj.expired_at)}. "
                f"Completion window was {batch_obj.completion_window} hours."
            )

        elif batch_obj.status == 'finalizing':
            # Provide information when the batch entered the finalizing stage
            error_obj = BatchFinalizingError(
                f"Batch {batch} is in the finalizing stage as of "
                f"{utc_int_to_iso_date(batch_obj.finalizing_at)}. "
                f"Please check back later for the final results."
            )

        # Attach the batch object to the error for debugging/context purposes
        error_obj.batch_obj = batch_obj
        raise error_obj


# --------------------------------------------------------------------------------------
# # Old functions for embeddings batch tasks
# import tempfile
# from pathlib import Path
# import json
# from oa.util import Sig.replace_kwargs_using


# @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
# def saved_embeddings_task(texts, **embeddings_params):
#     task_dict = mk_batch_file_embeddings_task(texts, **embeddings_params)
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
#     Path(temp_file.name).write_text(json.dumps(task_dict))
#     return temp_file.name


# @Sig.replace_kwargs_using(mk_batch_file_embeddings_task)
# def batch_input_file_for_embeddings(
#     texts: TextOrTexts, *, purpose='batch', **embeddings_params
# ):
#     _saved_embeddings_task = saved_embeddings_task(texts, **embeddings_params)

#     client = client or mk_client()
#     batch_input_file = client.files.create(
#         file=open(_saved_embeddings_task, 'rb'), purpose=purpose
#     )
#     batch_input_file._local_filepath = _saved_embeddings_task
#     return batch_input_file


# def batch_input_file_for_embeddings(
#     filepath: str, *, purpose='batch', client=None, **embeddings_params
# ):

#     client = client or mk_client()
#     batch_input_file = client.files.create(
#         file=open(filepath, 'rb'), purpose=purpose
#     )
#     batch_input_file._local_filepath = filepath
#     return batch_input_file
