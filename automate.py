import os
from queue import Queue
import subprocess
import threading

# Directory to store rendered image series.  Will be created if does not already exist.
image_series_dir = '/tmp'

# Directory to render final output to
final_render_dir = '/data/synced_outputs/leigh_woods/s06_tests'

# Path to the version of neural_style.py
style_transfer_script_dir = '/data/home/dan.treiman/neural-style-z'

# Number of GPUs to use
n_gpus = 4

# List of params to pass to the style transfer function.  Params take the form of tuples, where the first element is the
# argument, the second is the value.  If the value is a list, then the script is run once for each element of the list.
params = [
    # Static params
    ('--initial_frame', 0),

    # Searchable params
    ('--content_weight', [50, 500, 1000]),
    ('--tv_weight', [1, 0.1]),
]


# Build list of runnable commands
commands = [
    ['/usr/bin/python3', os.path.join(style_transfer_script, 'neural_style.py')]
]
for name, value in params:
    if isinstance(value, list):
        # Param contains a list of values to search.
        expanded_commands = []
        # Create a copy of commands for each case in the search list.
        for v in value:
            commands_copy = [c.copy() for c in commands]
            for c in commands_copy:
                c.extend([name, v])
            expanded_commands.extend(commands_copy)
        commands = expanded_commands
    else:
        # Param contains a single value
        for c in commands:
            c.extend([name, value])


# Add image series / output dir to command
for i, c in enumerate(commands):
    output_dir = os.path.join(image_series_dir, 'image_series_%d' % i)
    c.extend(['--output_dir', output_dir])
    print('%d: %s' % (i, c))


# Create queue to ensure we only run one process or each GPU.
gpu_queue = Queue()
for i in range(n_gpus):
    gpu_queue.put(i)


# Launch commands in sub-processes, assigning each to an available GPU.
for i, command in enumerate(commands):
    output_dir = os.path.join(image_series_dir, 'image_series_%d' % i)
    movie_path = os.path.join(final_render_dir, '%d.mp4' % i)
    gpu = gpu_queue.get()
    process = subprocess.Popen(
        command,
        cwd = style_transfer_script_dir,
        env = {
            'LD_LIBRARY_PATH': '',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES': str(gpu)
        }
    )
    def process_thread_body():
        """Returns gpu to available GPU queue after subprocess completes"""
        process.wait()
        gpu_queue.put(gpu)
        # Run FFMPEG command over image series to render final
        ffmpeg_command = ['/usr/bin/ffmpeg', movie_path]

    process_thread = threading.Thread(process_thread_body)
    process_thread.start()
