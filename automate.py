import os
import subprocess
import threading

# ---------------------------------------- System Settings ----------------------------------------

FFMPEG = 'usr/bin/ffmpeg'

PIP3 = '/usr/local/bin/pip3'

PYTHON3 = '/usr/bin/python3'

QSUB = '/opt/sge6/bin/linux-x64/qsub'

# SGE queue to run style transfer on
RENDER_QUEUE = 'gpu.q'

# SGE queue to encode movies on
ENCODE_QUEUE = 'cpu.q'

# ---------------------------------------- Configurable Settings ----------------------------------------

# Directory to store rendered image series.  Will be created if does not already exist.
image_series_dir = '/s3fs/tmp/leigh_woods/s06'

# Directory to render final output to
final_render_dir = '/data/synced_outputs/leigh_woods/s06'

# Prefix to append to filenames of rendered movies
output_prefix = 's06'

# Path to the version of neural_style.py
style_transfer_script_dir = '/data/home/dan.treiman/neural-style-z'

# Start index to use to label output files
output_index_start = 0

# List of params to pass to the style transfer function.  Params take the form of tuples, where the first element is the
# argument, the second is the value.  If the value is a list, then the script is run once for each element of the list.
params = [
    # Static params
    ('--initial_frame', 0),

    # Searchable params
    ('--content_weight', [50, 500, 1000]),
    ('--tv_weight', [1, 0.1]),
]


# ---------------------------------------- Derived Settings ----------------------------------------


log_path = os.path.join(final_render_dir, 'render_log.txt')


# ---------------------------------------- Initialization ----------------------------------------

# Check if output directories exist, create if necessary
if not os.path.isdir(image_series_dir):
    os.mkdir(image_series_dir)

if not os.path.isdir(final_render_dir):
    os.mkdir(final_render_dir)

# Open log file
log_file = open(log_path, 'w+')

# Log settings to log file
log_file.write('image_series_dir = \'%s\'\n' % image_series_dir)
log_file.write('final_render_dir = \'%s\'\n' % final_render_dir)
log_file.write('output_prefix = \'%s\'\n' % output_prefix)
log_file.write('style_transfer_script_dir = \'%s\'\n' % style_transfer_script_dir)
log_file.write('output_index_start = %d\n' % output_index_start)
log_file.write('params = %s\n' % repr(params))


# Build list of runnable commands
commands = [
    ['/usr/bin/python3', os.path.join(style_transfer_script_dir, 'neural_style.py')]
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


# Launch commands in sub-processes, assigning each to an available GPU.
for i, command in enumerate(commands):
    output_index = i + output_index_start
    output_dir = os.path.join(image_series_dir, '%s_%d' % (output_prefix, output_index))
    movie_path = os.path.join(final_render_dir, '%s_%d.mp4' % (output_prefix, output_index))
    #process = subprocess.Popen(
    #    command,
    #    cwd = style_transfer_script_dir,
    #    env = {
    #        'LD_LIBRARY_PATH': '',
    # #        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    #         'CUDA_VISIBLE_DEVICES': str(gpu)
    #     }
    # )
    # # Run FFMPEG command over image series to render final
    # ffmpeg_command = ['/usr/bin/ffmpeg', movie_path]



log_file.close()