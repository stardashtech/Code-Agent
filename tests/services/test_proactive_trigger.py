import unittest
from unittest.mock import patch, MagicMock, call
import time
import threading

# Adjust import path as needed
from services.proactive_trigger import ProactiveTrigger
import schedule # Need to interact with the mocked schedule instance

class TestProactiveTrigger(unittest.TestCase):

    def setUp(self):
        # It's often easier to patch the schedule module used by the class 
        # rather than the global schedule instance if tests run concurrently.
        # However, ProactiveTrigger uses the global instance directly. 
        # So we patch methods on the global schedule for testing.
        self.patcher_schedule_run = patch('schedule.run_pending')
        self.patcher_schedule_jobs = patch('schedule.get_jobs')
        self.patcher_schedule_every = patch('schedule.every')
        self.patcher_schedule_clear = patch('schedule.clear')
        self.patcher_time_sleep = patch('time.sleep') # Patch time.sleep to speed up tests

        self.mock_run_pending = self.patcher_schedule_run.start()
        self.mock_get_jobs = self.patcher_schedule_jobs.start()
        self.mock_every = self.patcher_schedule_every.start()
        self.mock_clear = self.patcher_schedule_clear.start()
        self.mock_sleep = self.patcher_time_sleep.start()
        
        # Mock the chainable methods of schedule
        self.mock_job = MagicMock()
        self.mock_unit = MagicMock(do=MagicMock(return_value=self.mock_job))
        self.mock_every.return_value = MagicMock(seconds=self.mock_unit)

        self.trigger = ProactiveTrigger()
        # Ensure schedule is cleared before each test by the object itself
        self.trigger.scheduler.clear()

    def tearDown(self):
        self.patcher_schedule_run.stop()
        self.patcher_schedule_jobs.stop()
        self.patcher_schedule_every.stop()
        self.patcher_schedule_clear.stop()
        self.patcher_time_sleep.stop()
        # Explicitly clear the global schedule instance after tests if necessary
        schedule.clear()

    def test_add_scheduled_job(self):
        \"\"\"Test adding a valid scheduled job.\"\"\"
        mock_callback = MagicMock()
        self.trigger.add_scheduled_job(10, mock_callback, job_name="test_job")
        
        # Check that schedule.every(10).seconds.do(mock_callback).tag('test_job') was called
        self.mock_every.assert_called_once_with(10)
        self.mock_unit.do.assert_called_once_with(mock_callback)
        self.mock_job.tag.assert_called_once_with("test_job")

    def test_add_scheduled_job_invalid_interval(self):
        \"\"\"Test adding a job with zero or negative interval.\"\"\"
        mock_callback = MagicMock()
        self.trigger.add_scheduled_job(0, mock_callback)
        self.trigger.add_scheduled_job(-5, mock_callback)
        # Ensure schedule.every was not called
        self.mock_every.assert_not_called()
        
    def test_start_stop_scheduler_thread(self):
         \"\"\"Test starting and stopping the scheduler thread.\"\"\"
         # Add a job so get_jobs returns something
         self.mock_get_jobs.return_value = [self.mock_job] 
         self.trigger.add_scheduled_job(5, MagicMock(), job_name="dummy")
         
         self.trigger.start()
         # Check thread is created and started (difficult to assert directly without more intrusive mocks)
         self.assertTrue(self.trigger._scheduler_thread.is_alive())
         
         # Allow the loop to run a few times (mock sleep helps here)
         time.sleep(0.1)
         self.mock_run_pending.assert_called()
         
         self.trigger.stop()
         # Thread should be stopped (join is called)
         # We mock sleep, so join should be fast
         self.assertFalse(self.trigger._scheduler_thread.is_alive())
         # Ensure stop event was set
         self.assertTrue(self.trigger._stop_event.is_set())
         
    def test_start_when_already_running(self):
        \"\"\"Test calling start when the trigger is already running.\"\"\"
        self.mock_get_jobs.return_value = [self.mock_job]
        self.trigger.start()
        thread1 = self.trigger._scheduler_thread
        
        # Call start again
        self.trigger.start()
        thread2 = self.trigger._scheduler_thread
        
        self.assertIs(thread1, thread2) # Should be the same thread instance
        self.assertTrue(thread1.is_alive()) # Still running
        # Ensure start logic was effectively skipped on second call
        
        self.trigger.stop() # Cleanup
        
    def test_stop_when_not_running(self):
         \"\"\"Test calling stop when the trigger is not running.\"\"\"
         # Should not raise error and log appropriately (checking logs is harder here)
         self.trigger.stop() 
         self.assertIsNone(self.trigger._scheduler_thread)
         
    def test_clear_jobs_all(self):
         \"\"\"Test clearing all jobs.\"\"\"
         self.trigger.clear_jobs()
         self.mock_clear.assert_called_once_with() # Called without tag
         
    def test_clear_jobs_with_tag(self):
         \"\"\"Test clearing jobs with a specific tag.\"\"\"
         self.trigger.clear_jobs(tag="important")
         self.mock_clear.assert_called_once_with("important")

if __name__ == '__main__':
    unittest.main() 