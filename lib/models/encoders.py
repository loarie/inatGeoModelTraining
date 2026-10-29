import math

import numpy as np
import tensorflow as tf


class CoordEncoder:
    def __init__(self, encoding_strategy, raster=None):
        self.encoding_strategy = encoding_strategy
        self.raster = raster

    def encode(self, locs, normalize=True):
        if normalize:
            locs = self._normalize_coords(locs)

        if self.encoding_strategy == "sinusoidal":
            loc_feats = self._encode_loc_sinusoidal(locs)
        elif self.encoding_strategy == "sinusoidal_hd":
            loc_feats = self._encode_loc_sinusoidal_hd(locs)

        if self.raster is not None:
            context_feats = self._bilinear_interpolate(locs, self.raster)
            loc_feats = np.concatenate((loc_feats, context_feats), 1)

        return loc_feats

    def encode_tf(self, locs, normalize=True):
        """Same output as encode(), but pure TensorFlow ops (no numpy), so it can run inside a
        tf.function / XLA on the GPU. The raster is copied to the device once, on first use."""
        if normalize:
            locs = self._normalize_coords(locs)

        if self.encoding_strategy == "sinusoidal":
            loc_feats = self._encode_loc_sinusoidal(locs)
        elif self.encoding_strategy == "sinusoidal_hd":
            loc_feats = self._encode_loc_sinusoidal_hd(locs)

        if self.raster is not None:
            loc_feats = tf.concat([loc_feats, self._bilinear_interpolate_tf(locs)], 1)

        return loc_feats

    def _bilinear_interpolate_tf(self, loc_ip):
        # TF port of _bilinear_interpolate (same arithmetic, so results match to float rounding)
        if not hasattr(self, "_raster_tf"):
            self._raster_tf = tf.constant(np.nan_to_num(self.raster, nan=0.0), dtype=tf.float32)
        data = self._raster_tf
        height, width = self.raster.shape[0], self.raster.shape[1]

        loc = (loc_ip + 1) / 2.0
        x = loc[:, 0] * (width - 1)
        y = (1 - loc[:, 1]) * (height - 1)

        x_floor = tf.floor(x)
        y_floor = tf.floor(y)
        xx = tf.cast(x_floor, tf.int32)
        yy = tf.cast(y_floor, tf.int32)
        xx_plus = tf.minimum(xx + 1, width - 1)
        yy_plus = tf.minimum(yy + 1, height - 1)
        dx = tf.expand_dims(x - x_floor, 1)
        dy = tf.expand_dims(y - y_floor, 1)

        def at(rows, cols):
            return tf.gather_nd(data, tf.stack([rows, cols], axis=1))

        return (
            at(yy, xx) * (1 - dx) * (1 - dy)
            + at(yy, xx_plus) * dx * (1 - dy)
            + at(yy_plus, xx) * (1 - dx) * dy
            + at(yy_plus, xx_plus) * dx * dy
        )

    def num_input_feats(self):
        if self.encoding_strategy == "sinusoidal_hd":
            coord_feats = 14
        else:
            coord_feats = 4
        if self.raster is not None:
            return coord_feats + self.raster.shape[-1]
        else:
            return coord_feats

    def _normalize_coords(self, locs):
        # locs is in lon {-180, 180}, lat {-90, 90}
        # output is in the range [-1, 1]
        norm_locs = tf.stack(
            [
                locs[:, 0] / 180.0,
                locs[:, 1] / 90.0,
            ],
            axis=1,
        )
        return norm_locs

    def _encode_loc_sinusoidal(self, loc_ip, concat_dim=1):
        # assumes inputs location are in range -1 to 1
        # location is lon, lat
        feats = tf.concat(
            [tf.sin(loc_ip * math.pi), tf.cos(loc_ip * math.pi)], axis=concat_dim
        )
        return feats

    def _encode_loc_sinusoidal_hd(self, loc_ip, concat_dim=1):
        # assumes inputs location are in range -1 to 1
        # location is lon, lat

        lng = tf.expand_dims(loc_ip[:,0], 1)
        lat = tf.expand_dims(loc_ip[:,1], 1)

        feats = tf.concat([
            lng,
            lat,
            
            tf.sin(lng * math.pi),
            tf.cos(lng * math.pi), 
            tf.sin(lat * math.pi), 

            tf.sin(2 * lng * math.pi),
            tf.cos(2 * lng * math.pi), 
            tf.sin(2 * lat * math.pi), 

            tf.sin(4 * lng * math.pi),
            tf.cos(4 * lng * math.pi), 
            tf.sin(4 * lat * math.pi), 

            tf.sin(8 * lng * math.pi),
            tf.cos(8 * lng * math.pi), 
            tf.sin(8 * lat * math.pi), 
        ], axis=concat_dim)

        return feats


    def _bilinear_interpolate(self, loc_ip, data, remove_nans_raster=True):
        # this is almost entirely exactly from the sinr paper source
        # with some minor modifications for pytortch -> numpy/tf

        # loc is N x 2 vector, where each row is [lon,lat] entry
        #   each entry spans range [-1,1]
        # data is H x W x C, height x width x channel data matrix
        # op will be N x C matrix of interpolated features

        assert data is not None

        # map to [0,1], then scale to data size
        loc = (loc_ip + 1) / 2.0
        # latitude goes from +90 on top to bottom
        # longitude goes from -9i0 to 90 left to right
        loc = tf.stack(
            [
                loc[:, 0],
                1 - loc[:, 1],
            ],
            axis=1,
        )

        assert not np.any(np.isnan(loc))

        if remove_nans_raster:
            # replace with mean value (0 will be mean post-normalization)
            data[np.isnan(data)] = 0.0

        # cast locations into pixel space
        loc = tf.stack(
            [
                loc[:, 0] * (data.shape[1] - 1),
                loc[:, 1] * (data.shape[0] - 1),
            ],
            axis=1,
        )

        # integer pixel coordinates
        loc_int = np.floor(loc).astype(int)
        xx = loc_int[:, 0]
        yy = loc_int[:, 1]
        xx_plus = xx + 1
        xx_plus[xx_plus > (data.shape[1] - 1)] = data.shape[1] - 1
        yy_plus = yy + 1
        yy_plus[yy_plus > (data.shape[0] - 1)] = data.shape[0] - 1

        loc_delta = loc - np.floor(loc)
        dx = np.expand_dims(loc_delta[:, 0], 1)
        dy = np.expand_dims(loc_delta[:, 1], 1)

        interp_val = (
            data[yy, xx, :] * (1 - dx) * (1 - dy)
            + data[yy, xx_plus, :] * dx * (1 - dy)
            + data[yy_plus, xx, :] * (1 - dx) * dy
            + data[yy_plus, xx_plus, :] * dx * dy
        )

        return interp_val
