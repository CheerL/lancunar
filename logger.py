# -*- coding:utf-8 -*-
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from functools import wraps


def no_same_name(cls):
    obj_dict = {}

    @wraps(cls)
    def _no_same_name(name, *args, **kw):
        if name not in obj_dict:
            obj_dict[name] = cls(name, *args, **kw)
        return obj_dict[name]
    return _no_same_name


@no_same_name
class Logger(logging.Logger):
    """
    LogHandler
    """

    def __init__(self, name, log_path, level=logging.DEBUG, handlers=['file', 'stream']):
        super().__init__(name, level)
        self.log_path = log_path
        self.file_handler = None
        self.stream_handler = None
        self.log_format = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        logging.Logger.__init__(self, self.name, level=level)
        for handler in handlers:
            getattr(self, 'set_{}_handler'.format(handler))()

    def set_file_handler(self, level=None):
        """
        set file handler
        :param level:
        :return:
        """
        file_name = os.path.join(self.log_path, '{name}.log'.format(name=self.name))
        # 设置日志回滚, 保存在log目录, 一天保存一个文件, 保留15天
        file_handler = TimedRotatingFileHandler(
            filename=file_name,
            when='D',
            interval=1,
            backupCount=15,
            encoding='utf-8'
        )
        file_handler.suffix = '%Y%m%d.log'
        if not level:
            file_handler.setLevel(self.level)
        else:
            file_handler.setLevel(level)
        formatter = logging.Formatter(self.log_format, self.date_format)

        file_handler.setFormatter(formatter)
        self.file_handler = file_handler
        self.addHandler(file_handler)

    def set_stream_handler(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(self.log_format, self.date_format)
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(self.level)
        else:
            stream_handler.setLevel(level)
        self.stream_handler = stream_handler
        self.addHandler(stream_handler)

    def resetName(self, name):
        """
        reset name
        :param name:
        :return:
        """
        self.name = name
        self.remove_file_handler()
        self.set_file_handler()

    def remove_file_handler(self):
        if self.file_handler:
            self.removeHandler(self.file_handler)
            self.file_handler = None

    def remove_stream_handler(self):
        if self.stream_handler:
            self.removeHandler(self.stream_handler)
            self.stream_handler = None


if __name__ == '__main__':
    logger = Logger('test')
