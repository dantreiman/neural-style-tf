import os
import subprocess

LOCK_DIR = '/tmp/gpu_locks'

NVIDIA_SMI = '/usr/bin/nvidia-smi'


if not os.path.isdir(LOCK_DIR):
    os.mkdir(LOCK_DIR)


def num_gpus():
    command_args = [
        NVIDIA_SMI,
        '--query-gpu=gpu_name,gpu_bus_id,vbios_version',
        '--format=csv']
    try:
        csv = subprocess.check_output(command_args)
        csv_lines = str(csv).split('\\n')
        return csv_lines - 2  # one for header row, one for trailing newline.
    except:
        return 0


def lock_path_for_gpu(gpu):
    return os.path.join(LOCK_DIR, '%d.lock' % gpu)


def is_locked(gpu):
    return os.path.isfile(lock_path_for_gpu(gpu))


def available_gpus():
    n_gpus = num_gpus()
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


def unlock_gpus(gpus):
    """Given list of GPU ids, unlock all"""
    for gpu in gpus:
        os.remove(lock_path_for_gpu(gpu))
