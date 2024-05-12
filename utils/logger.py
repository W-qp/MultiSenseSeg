# TODO:Creat logger
import logging
import os


def logger(net_type, log_dir=''):
    log_dir = log_dir + f"{net_type}_logger.log"
    logging.shutdown()
    if os.path.isfile(log_dir):
        os.remove(log_dir)

    log = logging.getLogger()
    log.setLevel('INFO')

    if not log.handlers:
        file = logging.FileHandler(log_dir)
        ssh = logging.StreamHandler()

        file.setFormatter(logging.Formatter(fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
        ssh.setFormatter(logging.Formatter(fmt='%(message)s'))

        log.addHandler(file)
        log.addHandler(ssh)
