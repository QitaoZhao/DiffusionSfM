import os
import signal
import subprocess
import sys
import time


def submitit_job_watcher(jobs, check_period: int = 15):
    job_out = {}

    try:
        while True:
            job_states = [job.state for job in jobs.values()]
            state_counts = {
                state: len([j for j in job_states if j == state])
                for state in set(job_states)
            }

            n_done = sum(job.done() for job in jobs.values())

            for job_name, job in jobs.items():
                if job_name not in job_out and job.done():
                    job_out[job_name] = {
                        "stderr": job.stderr(),
                        "stdout": job.stdout(),
                    }

                    exc = job.exception()
                    if exc is not None:
                        print(f"{job_name} crashed!!!")
                        if job_out[job_name]["stderr"] is not None:
                            print("===== STDERR =====")
                            print(job_out[job_name]["stderr"])
                    else:
                        print(f"{job_name} done!")

            print("Job states:")
            for state, count in state_counts.items():
                print(f"  {state:15s} {count:6d} ({100.*count/len(jobs):.1f}%)")

            if n_done == len(jobs):
                print("All done!")
                return

            time.sleep(check_period)

    except KeyboardInterrupt:
        for job_name, job in jobs.items():
            if not job.done():
                print(f"Killing {job_name}")
                job.cancel(check=False)


def get_jid():
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    return os.environ["SLURM_JOB_ID"]


def signal_helper(signum, frame):
    print(f"Caught signal {signal.Signals(signum).name} on for the this job")
    jid = get_jid()
    cmd = ["scontrol", "requeue", jid]
    try:
        print("calling", cmd)
        rtn = subprocess.check_call(cmd)
        print("subprocc", rtn)
    except:
        print("subproc call failed")
    return sys.exit(10)


def bypass(signum, frame):
    print(f"Ignoring signal {signal.Signals(signum).name} on for the this job")


def init_slurm_signals():
    signal.signal(signal.SIGCONT, bypass)
    signal.signal(signal.SIGCHLD, bypass)
    signal.signal(signal.SIGTERM, bypass)
    signal.signal(signal.SIGUSR2, signal_helper)
    print("SLURM signal installed", flush=True)


def init_slurm_signals_if_slurm():
    if "SLURM_JOB_ID" in os.environ:
        init_slurm_signals()
