# -*- coding: utf-8 -*-

import os
import json


def config_multi_thread():

    os.environ["MKL_NUM_THREADS"] = "1"

    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    os.environ["OMP_NUM_THREADS"] = "1"

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    os.environ["MKL_DYNAMIC"] = "FALSE"

    os.environ["OMP_DYNAMIC"] = "FALSE"

