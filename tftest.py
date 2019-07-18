import tensorflow as tf
import keras as ks

from __future__ import print_function

import pandas as pd
pd.__version__

import nems_lbhb.baphy as nb
import nems_lbhb.io as nio
from nems import epoch as ep
import os

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sgn
import scipy.io as sio
import collections

from nems_lbhb.strf.strf import tor_tuning
from nems_lbhb.strf.torc_subfunctions import interpft, strfplot, strf_torc_pred, strf_est_core



>>> from __future__ import print_function
...
... import math
...
... from IPython import display
... from matplotlib import cm
... from matplotlib import gridspec
... from matplotlib import pyplot as plt
... import numpy as np
... import pandas as pd
... from sklearn import metrics
... import tensorflow as tf
... from tensorflow.python.data import Dataset
...
... tf.logging.set_verbosity(tf.logging.ERROR)
... pd.options.display.max_rows = 10
... pd.options.display.float_format = '{:.1f}'.format



