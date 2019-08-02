import os
import pprint
import subprocess

# ---------------------------------------- System Settings ----------------------------------------

FFMPEG = '/usr/bin/ffmpeg'

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
style_transfer_script_dir = '/data/home/dan.treiman/neural-style-tf'

# Start index to use to label output files
output_index_start = 0

# List of params to pass to the style transfer function.  Params take the form of tuples, where the first element is the
# argument, the second is the value.  If the value is a list, then the script is run once for each element of the list.
params = [
    # Default params
    ('--model_weights', '/s3fs/models/VGG19/imagenet-vgg-verydeep-19.mat'),
    ('--video',),
    ('--backward_optical_flow_frmt', 'backward_{}_{}.flo'),
    ('--verbose',),
    ('--print_iterations', 50),
    ('--init_frame_type', 'prev_warped'),

    # Static params
    ('--initial_frame', 0),
    ('--start_frame', 0),
    ('--end_frame', 158),
    ('--content_frame_frmt', 's5_{}.png'),
    ('--content_frame_digits', 3),
    ('--depth_frame_frmt', 's5_depth_02_{}.exr'),
    ('--depth_index_offset', 40),
    ('--first_frame_type', 'content'),
    ('--optimizer', 'adam'),
    ('--video_input_dir', '/s3fs/content/video/leigh_woods_clips/s05'),
    ('--flow_input_dir', '/s3fs/content/video/leigh_woods_clips/s05_flow'),
    ('--depth_input_dir', '/s3fs/content/video/leigh_woods_clips/s05_depth'),

    # Searchable params
    ('--style_imgs', '/s3fs/content/images/style_images/tamani/raices_upper1080.png'),
    ('--learning_rate', 0.5),
    ('--first_frame_iterations', 1000),
    ('--frame_iterations', 400),
    ('--depth_lookback', 10),
    ('--content_weight', [50, 500, 1000]),
    ('--tv_weight', [1, 0.1]),
    ('--temporal_weight', 1000),
    ('--octaves', 2),
    ('--downsample_method', 'gaussian'),
]

# ---------------------------------------- Derived Settings ----------------------------------------


log_path = os.path.join(final_render_dir, 'render_log.txt')
print('Logging output to %s' % log_path)

job_log_path = os.path.join(final_render_dir, 'logs')


def get_param(name):
    matches = [p for p in params if p[0] == name]
    return matches[0][1] if matches else None


content_frame_frmt = get_param('--content_frame_frmt')
content_frame_digits = get_param('--content_frame_digits')

# ---------------------------------------- Initialization ----------------------------------------

# Check if output directories exist, create if necessary
if not os.path.isdir(image_series_dir):
    os.mkdir(image_series_dir)

if not os.path.isdir(final_render_dir):
    os.mkdir(final_render_dir)

if not os.path.isdir(job_log_path):
    os.mkdir(job_log_path)


# Open log file
log_file = open(log_path, 'w+')

# Log settings to log file
log_file.write('image_series_dir = \'%s\'\n' % image_series_dir)
log_file.write('final_render_dir = \'%s\'\n' % final_render_dir)
log_file.write('output_prefix = \'%s\'\n' % output_prefix)
log_file.write('style_transfer_script_dir = \'%s\'\n' % style_transfer_script_dir)
log_file.write('output_index_start = %d\n' % output_index_start)
log_file.write('params = ')
pprint.pprint(params, stream=log_file, indent=4)


# Build list of runnable commands
commands = [
    [PYTHON3, os.path.join(style_transfer_script_dir, 'neural_style.py')]
]
for param in params:
    if len(param) == 1:
        # param contains a key only
        for c in commands:
            c.append(param[0])
        continue
    name, value = param
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
        # Param contains a single key-value pair
        for c in commands:
            c.extend([name, value])


# Add image series / output dir to command
for i, c in enumerate(commands):
    output_index = i + output_index_start
    output_dir = os.path.join(image_series_dir, '%s_styled_%d' % (output_prefix, output_index))
    c.extend(['--video_output_dir', output_dir])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    qsub_command = [
        QSUB,
        '-terse',
        '-b', 'y',
        '-q', RENDER_QUEUE,
        '-v', 'LD_LIBRARY_PATH=/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:',
        '-o', os.path.join(job_log_path, '%s_%d_render.stdout' % (output_prefix, output_index)),
        '-e', os.path.join(job_log_path, '%s_%d_render.stderr' % (output_prefix, output_index))
    ]
    qsub_command.extend([str(a) for a in c])

    jid_string = subprocess.check_output(qsub_command)
    jid = int(jid_string)

    log_file.write('\n%d render (job %d)\n' % (output_index, jid))
    log_file.write(' '.join(qsub_command))
    log_file.write('\n')

    movie_path = os.path.join(final_render_dir, '%s_%d.mp4' % (output_prefix, output_index))
    encode_qsub_command = [
        QSUB,
        '-terse',
        '-b', 'y',
        '-q', ENCODE_QUEUE,
        '-hold_jid', str(jid),
        '-o', os.path.join(job_log_path, '%s_%d_encode.stdout' % (output_prefix, output_index)),
        '-e', os.path.join(job_log_path, '%s_%d_encode.stderr' % (output_prefix, output_index)),
        FFMPEG,
        '-y',
        '-r', '25',
        '-f', 'image2',
        '-i', os.path.join(output_dir, content_frame_frmt.format(
            '%0{}d'.format(content_frame_digits)
        )),
        '-vcodec', 'libx264',
        '-crf', '25',
        '-pix_fmt', 'yuv420p',
        movie_path
    ]

    encode_jid_string = subprocess.check_output(encode_qsub_command)
    encode_jid = int(encode_jid_string)
    log_file.write('\n%d encode (job %d)\n' % (output_index, encode_jid))
    log_file.write(' '.join(encode_qsub_command))
    log_file.write('\n\n')


log_file.close()
