"""Test cases for the oa.util module."""

from typing import Any, Tuple, Dict
from oa.util import ProcessingManager, Status, Result


def test_processing_manager_all_complete():
    """
    Test that all items are processed when they complete immediately.
    """

    # Define a processing function that always returns 'completed'
    def processing_function(item: Any) -> Tuple[Status, Result]:
        return 'completed', f"Result for {item}"

    # Define a handle_status_function that removes items when completed
    def handle_status_function(item: Any, status: Status, result: Result) -> bool:
        return status == 'completed'

    # Define a wait_time_function that doesn't wait
    def wait_time_function(cycle_duration: float, local_vars: Dict) -> float:
        return 0.0

    pending_items = {'item1': 'data1', 'item2': 'data2', 'item3': 'data3'}

    manager = ProcessingManager(
        pending_items=pending_items.copy(),
        processing_function=processing_function,
        handle_status_function=handle_status_function,
        wait_time_function=wait_time_function,
    )

    manager.process_items()

    assert manager.status is True
    expected_completed_items = {k: f"Result for {v}" for k, v in pending_items.items()}
    assert manager.completed_items == expected_completed_items
    assert manager.cycles == 1


def test_processing_manager_user_story():
    """
    User Story Test for ProcessingManager

    This test simulates a scenario where a set of tasks are processed using the ProcessingManager.
    It demonstrates the following behaviors:
    - Initialization with a mix of tasks.
    - Handling of tasks with different statuses: 'in_progress', 'completed', 'failed'.
    - Updating task statuses over multiple cycles.
    - Removal of tasks based on the status handling function.
    - Use of the wait time function to control cycle timing.
    - Tracking of cycles and completed tasks.
    """

    import time
    from typing import Any, Tuple, Dict

    # Simulate a set of tasks with initial data
    pending_items = {
        'task1': {'data': 'data1'},  # Will complete after first cycle
        'task2': {'data': 'data2'},  # Will remain in progress
        'task3': {'data': 'data3'},  # Will fail after first cycle
        'task4': {'data': 'data4'},  # Will complete after two cycles
        'task5': {'data': 'data5'},  # Will fail and then be retried
    }

    # Dictionary to keep track of task statuses and retries
    task_statuses = {
        'task1': 'in_progress',
        'task2': 'in_progress',
        'task3': 'in_progress',
        'task4': 'in_progress',
        'task5': 'in_progress',
    }

    retry_counts = {
        'task5': 0,  # Will retry on failure
    }

    def processing_function(item: Any) -> Tuple[Status, Result]:
        """
        Simulates processing of a task.
        """
        task_id = item['task_id']
        current_status = task_statuses[task_id]

        # Simulate status transitions
        if task_id == 'task1' and current_status == 'in_progress':
            # Task1 completes after first cycle
            task_statuses[task_id] = 'completed'
            result = f"Result for {task_id}"
            return 'completed', result

        elif task_id == 'task2':
            # Task2 remains in progress indefinitely
            result = None
            return 'in_progress', result

        elif task_id == 'task3' and current_status == 'in_progress':
            # Task3 fails after first cycle
            task_statuses[task_id] = 'failed'
            result = f"Error in {task_id}"
            return 'failed', result

        elif task_id == 'task4' and current_status == 'in_progress':
            # Task4 completes after two cycles
            task_statuses[task_id] = 'in_progress_2'
            result = None
            return 'in_progress', result
        elif task_id == 'task4' and current_status == 'in_progress_2':
            task_statuses[task_id] = 'completed'
            result = f"Result for {task_id}"
            return 'completed', result

        elif task_id == 'task5':
            # Task5 fails once and then retries
            if retry_counts['task5'] == 0:
                retry_counts['task5'] += 1
                result = f"Temporary error in {task_id}"
                return 'failed', result
            else:
                task_statuses[task_id] = 'completed'
                result = f"Result for {task_id} after retry"
                return 'completed', result

        else:
            # Default case
            result = None
            return 'in_progress', result

    def handle_status_function(item: Any, status: Status, result: Result) -> bool:
        """
        Determines whether to remove the task based on its status.
        """
        task_id = item['task_id']

        if status == 'completed':
            # Task is completed; remove it
            print(f"Task {task_id} completed with result: {result}")
            return True
        elif status == 'failed':
            if task_id == 'task5' and retry_counts['task5'] <= 1:
                # Retry task5 once on failure
                print(f"Task {task_id} failed with error: {result}. Retrying...")
                return False  # Keep in pending_items for retry
            else:
                # For other tasks or after retry, remove the task
                print(f"Task {task_id} failed with error: {result}. Not retrying.")
                return True
        else:
            # Task is still in progress; keep it
            print(f"Task {task_id} is in progress.")
            return False

    def wait_time_function(cycle_duration: float, local_vars: Dict) -> float:
        """
        Determines how long to wait before the next cycle.
        """
        status_check_interval = local_vars['self'].status_check_interval
        sleep_duration = max(0, status_check_interval - cycle_duration)
        print(f"Waiting for {sleep_duration:.2f} seconds before next cycle.")
        return sleep_duration

    # Add task IDs to the items for easy tracking
    for task_id, item in pending_items.items():
        item['task_id'] = task_id

    # Initialize the ProcessingManager with the pending tasks
    manager = ProcessingManager(
        pending_items=pending_items,
        processing_function=processing_function,
        handle_status_function=handle_status_function,
        wait_time_function=wait_time_function,
        status_check_interval=1.0,  # Check every 1 second
        max_cycles=5,  # Limit to 5 cycles to prevent infinite loops
    )

    # Record the start time
    start_time = time.time()

    # Start the processing loop
    manager.process_items()

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Assertions to check that the manager behaved as expected
    # Task1 and Task4 should be completed
    assert 'task1' in manager.completed_items
    assert 'task4' in manager.completed_items

    # Task3 should have failed and been removed
    assert 'task3' in manager.completed_items

    # Task5 should have retried and then completed
    assert 'task5' in manager.completed_items

    # Task2 should still be in pending_items (since it remains in progress)
    assert 'task2' in manager.pending_items

    # The manager should have run for the expected number of cycles
    assert manager.cycles == manager.max_cycles or manager.status is True

    # Output the final state for verification
    print("\nFinal State:")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Cycles executed: {manager.cycles}")
    print(f"Completed tasks: {list(manager.completed_items.keys())}")
    print(f"Pending tasks: {list(manager.pending_items.keys())}")

    # Ensure that the test completes without errors
    assert True


test_processing_manager_all_complete()
test_processing_manager_user_story()
