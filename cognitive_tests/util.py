import logging
#from colorlog import ColoredFormatter
import os


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
#
# formatter = ColoredFormatter(
#     "%(log_color)s[%(asctime)s] %(message)s",
# #    datefmt='%H:%M:%S.%f',
#     datefmt=None,
#     reset=True,
#     log_colors={
#         'DEBUG':    'cyan',
#         'INFO':     'white,bold',
#         'INFOV':    'cyan,bold',
#         'WARNING':  'yellow',
#         'ERROR':    'red,bold',
#         'CRITICAL': 'red,bg_white',
#     },
#     secondary_log_colors={},
#     style='%'
# )
# ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
