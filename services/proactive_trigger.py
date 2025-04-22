import time
import threading
import logging
from typing import Callable, Optional, Any
import schedule # Using the 'schedule' library (add to requirements.txt)

logger = logging.getLogger(__name__)

class ProactiveTrigger:
    \"\"\"
    Triggers a predefined callback function based on schedules or events.
    Currently supports scheduled interval triggers.
    File system event triggering can be added later.
    \"\"\"

    def __init__(self):
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        # Use the schedule library's default scheduler instance
        self.scheduler = schedule

    def add_scheduled_job(self, interval_seconds: int, job_func: Callable[[], Any], job_name: Optional[str] = None):
        \"\"\"
        Adds a job to be run periodically.

        Args:
            interval_seconds: How often to run the job in seconds.
            job_func: The function to call when the job runs.
            job_name: An optional name for the job for logging/identification.
        \"\"\"
        if interval_seconds <= 0:
            logger.error("Scheduled job interval must be positive.")
            return
            
        job_id = job_name or f"job_{id(job_func)}"
        logger.info(f"Scheduling job '{job_id}' to run every {interval_seconds} seconds.")
        # Schedule the job. Tagging can help manage jobs.
        self.scheduler.every(interval_seconds).seconds.do(job_func).tag(job_id)

    def _run_scheduler(self):
        \"\"\"Internal method run in a separate thread to check pending jobs.\"\"\"
        logger.info("ProactiveTrigger scheduler thread started.")
        while not self._stop_event.is_set():
            try:
                 self.scheduler.run_pending()
                 # Sleep for a short duration to avoid busy-waiting
                 # Adjust sleep time based on the granularity needed vs CPU usage
                 time.sleep(1) 
            except Exception as e:
                 logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                 # Avoid crashing the thread, wait a bit longer before retrying
                 time.sleep(5)
        logger.info("ProactiveTrigger scheduler thread stopped.")

    def start(self):
        \"\"\"Starts the trigger mechanism in a background thread.\"\"\"
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("ProactiveTrigger is already running.")
            return

        if not self.scheduler.get_jobs():
             logger.warning("ProactiveTrigger starting, but no jobs are scheduled.")
             # Decide if we should start the thread anyway or wait for jobs?
             # Start anyway for now, jobs can be added later.

        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("ProactiveTrigger started.")

    def stop(self):
        \"\"\"Stops the trigger mechanism gracefully.\"\"\"
        if not self._scheduler_thread or not self._scheduler_thread.is_alive():
            logger.info("ProactiveTrigger is not running.")
            return

        logger.info("Stopping ProactiveTrigger...")
        self._stop_event.set()
        # Wait for the scheduler thread to finish
        self._scheduler_thread.join(timeout=5) # Wait up to 5 seconds
        if self._scheduler_thread.is_alive():
             logger.warning("Scheduler thread did not stop gracefully after timeout.")
        else:
             logger.info("ProactiveTrigger stopped.")
        self._scheduler_thread = None
        
    def clear_jobs(self, tag: Optional[str] = None):
        \"\"\"Clears all scheduled jobs, or jobs with a specific tag.\"\"\"
        if tag:
            logger.info(f"Clearing scheduled jobs with tag: {tag}")
            self.scheduler.clear(tag)
        else:
            logger.info("Clearing all scheduled jobs.")
            self.scheduler.clear()

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define some dummy job functions
    def my_periodic_task():
        logger.info("Periodic task is running!")

    def another_task():
        logger.info("Another task executed.")

    trigger = ProactiveTrigger()

    # Schedule jobs
    trigger.add_scheduled_job(5, my_periodic_task, job_name="task1")
    trigger.add_scheduled_job(12, another_task)

    # Start the trigger system
    trigger.start()

    try:
        # Keep the main thread alive to observe the scheduled jobs
        # In a real application, this might be the main event loop or server process
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping triggers...")
    finally:
        # Stop the trigger system gracefully
        trigger.stop()
        logger.info("Application finished.") 