# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os

import numpy as np
import torch
import datetime
from netCDF4 import Dataset as DS
from ai_models_gfs.model import Model

import ai_models_fourcastnetv2_gfs.fourcastnetv2 as nvs

LOG = logging.getLogger(__name__)


class FourCastNetv2(Model):
    # Download
    download_url = "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/{file}"
    download_files = ["weights.tar", "global_means.npy", "global_stds.npy"]

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

    param_level_pl = (
        ["t", "u", "v", "z", "r"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    ordering = [
        "10u",
        "10v",
        "100u",
        "100v",
        "2t",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]

    # Output
    expver = "sfno"

    def __init__(self, precip_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.n_lat = 721
        self.n_lon = 1440
        self.hour_steps = 6

        self.backbone_channels = len(self.ordering)

        self.checkpoint_path = os.path.join(self.assets, "weights.tar")

    def load_statistics(self):
        path = os.path.join(self.assets, "global_means.npy")
        LOG.info("Loading %s", path)
        self.means = np.load(path)
        self.means = self.means[:, : self.backbone_channels, ...]
        self.means = self.means.astype(np.float32)

        path = os.path.join(self.assets, "global_stds.npy")
        LOG.info("Loading %s", path)
        self.stds = np.load(path)
        self.stds = self.stds[:, : self.backbone_channels, ...]
        self.stds = self.stds.astype(np.float32)

    def load_model(self, checkpoint_file):
        model = nvs.FourierNeuralOperatorNet()

        model.zero_grad()
        # Load weights

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        weights = checkpoint["model_state"]
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # Make sure the parameter names are the same as the checkpoint
        # need to use strict = False to avoid this error message when
        # using sfno_76ch::
        # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # Missing key(s) in state_dict: "module.trans_down.weights",
        # "module.itrans_up.pct",
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        return model

    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if reverse:
            new_data = data * self.stds + self.means
        else:
            new_data = (data - self.means) / self.stds
        return new_data

    def run(self):
        self.load_statistics()

        all_fields = self.all_fields
        all_fields = all_fields.sel(
            param_level=self.ordering, remapping={"param_level": "{param}{levelist}"}
        )
        all_fields = all_fields.order_by(
            {"param_level": self.ordering},
            remapping={"param_level": "{param}{levelist}"},
        )

        all_fields_numpy = all_fields.to_numpy(dtype=np.float32)
        all_fields_numpy_copy = np.copy(all_fields_numpy)

        all_fields_numpy = self.normalise(all_fields_numpy)

        model = self.load_model(self.checkpoint_path)

        # Run the inference session
        input_iter = torch.from_numpy(all_fields_numpy).to(self.device)

        # sample_sfc = all_fields.sel(param="2t")[0]
        self.write_input_fields(all_fields)

        torch.set_grad_enabled(False)

        #Dictionary to hold output and variable mappings
        if 'n' in self.nc_or_grib:
            out,mapping,varmap = initialize_nc_dict(self.lead_time,self.hour_steps)

        #Save initial conditions to output dictionary and write to grib
        for k, fs in enumerate(all_fields):
            if 'n' in self.nc_or_grib:
                shortname = fs.handle.get("shortName")
                level = fs.handle.get("level")
                mappedvar = varmap[shortname]
                if level!=0:
                    levelidx = mapping[level]
                    out[mappedvar]['values'][0,levelidx,:,:] = all_fields_numpy_copy[ k, ...]
                else:
                    out[mappedvar]['values'][0,:,:] = all_fields_numpy_copy[ k, ...]
            if 'g' in self.nc_or_grib:
                self.write(
                    all_fields_numpy_copy[ k, ...], check_nans=True, template=fs, step=0
                )

        with self.stepper(self.hour_steps) as stepper:
            for i in range(self.lead_time // self.hour_steps):
                output = model(input_iter)

                input_iter = output
                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of normalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s %s %s %s %s",
                            name,
                            np.mean(output[:, j].cpu().numpy()),
                            np.std(output[:, j].cpu().numpy()),
                            np.amin(output[:, j].cpu().numpy()),
                            np.amax(output[:, j].cpu().numpy()),
                        )

                # Save the results
                step = (i + 1) * self.hour_steps
                output = self.normalise(output.cpu().numpy(), reverse=True)

                if i == 0 and LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug("Mean/stdev of denormalised values: %s", output.shape)

                    for j, name in enumerate(self.ordering):
                        LOG.debug(
                            "    %s mean=%s std=%s min=%s max=%s",
                            name,
                            np.mean(output[:, j]),
                            np.std(output[:, j]),
                            np.amin(output[:, j]),
                            np.amax(output[:, j]),
                        )

                for k, fs in enumerate(all_fields):

                    #Save output to dictionary to write nc
                    if 'n' in self.nc_or_grib:
                        shortname = fs.handle.get("shortName")
                        level = fs.handle.get("level")
                        mappedvar = varmap[shortname]
                        if level!=0:
                            levelidx = mapping[level]
                            out[mappedvar]['values'][i+1,levelidx,:,:] = output[0, k, ...]
                        else:
                            out[mappedvar]['values'][i+1,:,:] = output[0, k, ...]

                    #Write grib
                    if 'g' in self.nc_or_grib:
                        self.write(
                            output[0, k, ...], check_nans=True, template=fs, step=step
                        )

                stepper(i, step)

        #Write nc
        if 'n' in self.nc_or_grib:
            write_nc(out,self.lead_time,self.hour_steps,self.date,self.time,self.ncpath)


def model(model_version, **kwargs):
    models = {
        "0": FourCastNetv2,
        "small": FourCastNetv2,
        "release": FourCastNetv2,
        "latest": FourCastNetv2,
    }
    return models[model_version](**kwargs)

def create_variable(f, name, dimensions, data, attrs):
    if name in ['time','level']:
        dtype = 'i4'
    else:
        dtype = 'f4'
    var = f.createVariable(name, dtype, dimensions,compression='zlib',complevel=4)
    var[:] = data
    for attr_name, attr_value in attrs.items():
        var.setncattr(attr_name, attr_value)

def initialize_nc_dict(lead_time,hour_steps):
    out = {
        'u10': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '10 metre U wind component', 'units': 'm s-1'
        },
        'v10': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '10 metre V wind component', 'units': 'm s-1'
        },
        'u100': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '100 metre U wind component', 'units': 'm s-1'
        },
        'v100': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '100 metre V wind component', 'units': 'm s-1'
        },
        't2': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': '2 metre temperature', 'units': 'K'
        },
        'sp': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': 'Surface pressure', 'units': 'Pa'
        },
        'msl': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': 'Pressure reduced to MSL', 'units': 'Pa'
        },
        'tcwv': {
            'values': np.zeros((lead_time // hour_steps + 1, 721, 1440)),
            'name': 'Precipitable water', 'units': 'kg m-2'
        },
        't': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Temperature', 'units': 'K'
        },
        'u': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'U component of wind', 'units': 'm s-1'
        },
        'v': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'V component of wind', 'units': 'm s-1'
        },
        'z': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Geopotential', 'units': 'm2 s-2'
        },
        'r': {
            'values': np.zeros((lead_time // hour_steps + 1, 13, 721, 1440)),
            'name': 'Relative humidity', 'units': '%'
        },
    }

    mapping = {
        50:12,
        100:11,
        150:10,
        200:9,
        250:8,
        300:7,
        400:6,
        500:5,
        600:4,
        700:3,
        850:2,
        925:1,
        1000:0
    }

    varmap = {
        "u":"u",
        "v":"v",
        "z":"z",
        "t":"t",
        "r":"r",
        "10u":"u10",
        "10v":"v10",
        "100u":"u100",
        "100v":"v100",
        "sp":"sp",
        "msl":"msl",
        "tcwv":"tcwv",
        "2t":"t2"
    }


    return out,mapping,varmap

def write_nc(out,lead_time,hour_steps,date,time,path):
    outdir = path
    f = DS(outdir, 'w', format='NETCDF4')
    f.createDimension('time', lead_time // hour_steps + 1)
    f.createDimension('level', 13)
    f.createDimension('longitude', 1440)
    f.createDimension('latitude', 721)

    year = str(date)[0:4]
    month = str(date)[4:6]
    day = str(date)[6:8]
    hh = str(int(time/100)).zfill(2)
    initdt = datetime.datetime.strptime(f"{year}{month}{day}{hh}","%Y%m%d%H")
    inityr = str(initdt.year)
    initmnth = str(initdt.month).zfill(2)
    initday = str(initdt.day).zfill(2)
    inithr = str(initdt.hour).zfill(2)
    times = []
    for i in np.arange(0,lead_time + hour_steps,hour_steps):
        times.append(int((initdt + datetime.timedelta(hours=int(i))).timestamp()))

    # Create time, longitude, latitude, and level variables in the NetCDF file
    create_variable(
        f, 'time', ('time',), np.array(times), {
            'long_name': 'Date and Time', 'units': 'seconds since 1970-1-1',
            'calendar': 'standard'
        }
    )
    create_variable(
        f, 'longitude', ('longitude',), np.arange(0, 360, 0.25), {
            'long_name': 'Longitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'latitude', ('latitude',), np.arange(-90, 90.25, 0.25)[::-1], {
            'long_name': 'Latitude', 'units': 'degree'
        }
    )
    create_variable(
        f, 'level', ('level',), np.array(
            [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        )[::-1], {'long_name': 'Isobaric surfaces', 'units': 'hPa'}
    )

    # Create variables for each meteorological parameter
    for variable in [
        'u10', 'v10', 'u100', 'v100', 't2', 'msl', 'sp', 'tcwv', 't', 'u', 'v', 'z', 'r'
    ]:
        dims = ('time', 'level', 'latitude', 'longitude') if variable in [
            'u', 'v', 'z', 't', 'r'
        ] else ('time', 'latitude', 'longitude')
        create_variable(
            f, variable, dims, out[variable]['values'], {
                'long_name': out[variable]['name'], 'units': out[variable]['units']
            }
        )

    f.Conventions = 'CF-1.8'
    f.model_name = 'FourCastNet'
    f.model_version = 'v2-small'
    f.initialization_model = 'GFS'
    f.initialization_time = '%s-%s-%sT%s:00:00' % (inityr,initmnth,initday,inithr)
    f.first_forecast_hour = str(0)
    f.last_forecast_hour = str(lead_time)
    f.forecast_hour_step = str(hour_steps)
    f.creation_time = (datetime.datetime.utcnow()).strftime('%Y-%m-%dT%H:%M:%S')
    f.close()
