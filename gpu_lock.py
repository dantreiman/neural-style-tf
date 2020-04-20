import os
from simpleflock import SimpleFlock
import subprocess
import sys

LOCK_DIR = '/tmp/gpu_locks'

NVIDIA_SMI = '/usr/bin/nvidia-smi'


if not os.path.isdir(LOCK_DIR):
    try:
        os.mkdir(LOCK_DIR)
    except:
        print('%s already exists', LOCK_DIR)


def num_gpus():
    command_args = [
        NVIDIA_SMI,
        '--query-gpu=gpu_name,gpu_bus_id,vbios_version',
        '--format=csv']
    try:
        csv = subprocess.check_output(command_args)
        csv_lines = str(csv).split('\\n')
        return len(csv_lines) - 2  # one for header row, one for trailing newline.
    except:
        return 0


def lock_path_for_gpu(gpu):
    return os.path.join(LOCK_DIR, '%d.lock' % gpu)


def is_locked(gpu):
    return os.path.isfile(lock_path_for_gpu(gpu))


def get_available_gpus():
    n_gpus = num_gpus()
    print('num_gpus: %d' % n_gpus)
    all_gpus = set(range(n_gpus))
    for gpu in range(n_gpus):
        if is_locked(gpu):
            all_gpus.remove(gpu)
    return list(all_gpus)


def lock_gpus(gpus):
    """Locks n gpus, returns list of GPU ids"""
    for gpu in gpus:
        with open(lock_path_for_gpu(gpu), 'w') as lockfile:
            lockfile.write('locked\n')


def acquire_gpus(required_gpus, timeout=60):
    """Lock the required number of GPUs, return a list."""
    with SimpleFlock('/tmp/gpu_locks/global.lock', timeout):
        available_gpus = get_available_gpus()
        print('available_gpus: %s' % str(available_gpus))
        if (len(available_gpus) < required_gpus):
            print('Need %d GPUs available to run!' % required_gpus)
            sys.exit(1)
        selected_gpus = available_gpus[:required_gpus]
        lock_gpus(selected_gpus)
    return selected_gpus


def unlock_gpus(gpus):
    """Given list of GPU ids, unlock all"""
    for gpu in gpus:
        lock_path = lock_path_for_gpu(gpu)
        try:
            os.remove(lock_path)
        except:
            print('Failed to remove gpu lock at %s' % lock_path)
