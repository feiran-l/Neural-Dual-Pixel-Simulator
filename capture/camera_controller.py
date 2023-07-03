#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


class BaseCameraController(object):
    def __init__(self, device_ports, name, output_root_path):
        """

        :param list of str device_ports: serial number of connected cafmeras
        :param list of str name: device names
        :param str output_root_path: storage directory
        """
        self.device_ports = device_ports
        self.name = name
        self.output_root_path = output_root_path

        self.device_names = []
        self.cfg_fps = -1
        self.cfg_iso = -1
        self.cfg_shutter_speed = -1
        self.cfg_gain = -1

        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)

    def __len__(self):
        """
        return the number of photos taken by one shoot.
        """
        raise NotImplementedError

    def init(self):
        raise NotImplementedError

    def close(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @classmethod
    def available_devices(cls):
        """
        :return: list of available devices
        :rtype: tuple of str, str: name of the device and port
        """
        raise NotImplementedError

    def set_to_manual_mode(self):
        """
        set camera mode to manual mode and turn off all of post processing.
        """
        raise NotImplementedError

    def set_config(self, shutter_speed, iso, fps, gain):
        self.cfg_fps = fps
        self.cfg_shutter_speed = shutter_speed
        self.cfg_iso = iso
        self.cfg_gain = gain

    def shoot(self, file_name_prefix):
        """
        Take photo and output the files under:
        self.output_root_path/{INDEX_OF_CAMERA_DEVICE}/{file_name_prefix}{suffix}.{ext}
        File writes may be done when self.transfer() is called.
        :param str file_name_prefix:
        """
        raise NotImplementedError

    def transfer(self):
        """
        Maybe needed for gphoto?
        Even if the class does not require this function, please overwrite this by ``pass``.
        """
        raise NotImplementedError

    def preview(self, fps=-1, is_save=False):
        raise NotImplementedError
