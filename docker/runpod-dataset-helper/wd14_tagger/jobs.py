# autotag/jobs.py
import threading

autotag_jobs = {}
autotag_lock = threading.Lock()