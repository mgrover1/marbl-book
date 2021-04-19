# dFE Comparison Notebook
---
## Setup

### Imports
Notice this first line, we use the `%load_ext` and `%autoreload` to automatically update the packages used inline with the most recent modifications made

%load_ext autoreload
%autoreload 2

import os

from itertools import product

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import cmocean

import cartopy
import cartopy.crs as ccrs

import xpersist as xp
cache_dir = '/glade/p/cgd/oce/projects/cesm2-marbl/xpersist_cache/3d_fields'
if (os.path.isdir(cache_dir)):
    xp.settings['cache_dir'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

os.environ['CESMDATAROOT'] = '/glade/scratch/mclong/inputdata'
import pop_tools


import climo_utils as cu
import utils
import discrete_obs 

import plot

### Use `intake-esm` to search for variables, components, and experiments

import intake
catalog = intake.open_esm_datastore('data/campaign-cesm2-cmip6-timeseries.json')
df = catalog.search(experiment='historical', component='ocn', stream='pop.h').df
variables = df.variable.unique()

[v for v in variables if 'Fe' in v or 'iron' in v.lower() or 'sed' in v.lower()]

### Spin up dask cluster

cluster, client = utils.get_ClusterClient()
cluster.scale(12) #adapt(minimum_jobs=0, maximum_jobs=24)
client

### Read in the pop-grid

ds_grid = pop_tools.get_grid('POP_gx1v7')
ds_grid

## Operate on dataset using `xpersist` to cache output

nmolcm3_to_nM = 1e3
nmolcm2s_to_mmolm2yr = 1e-9 * 1e3 * 1e4 * 86400 * 365.
µmolm2d_to_mmolm2yr = 1e-3 * 365.

time_slice = slice("1990-01-15", "2015-01-15")
varlist = [
    'Fe',
    'IRON_FLUX',
    'Fe_RIV_FLUX',
    'pfeToSed',    
]
ds_list = []
for variable in varlist:
    xp_func = xp.persist_ds(cu.read_CESM_var, name=f'{variable}', trust_cache=True)    
    ds_list.append(xp_func(
        time_slice, 
        variable, 
        mean_dims=['member_id', 'time'], 
    ))
    
ds = xr.merge(ds_list)
with xr.set_options(keep_attrs=True):
    for v in ['Fe']:
        assert ds[v].attrs['units'] == 'mmol/m^3'
        ds[v] = ds[v] * nmolcm3_to_nM
        ds[v].attrs['units'] = 'nM'
    for v in ['Fe_RIV_FLUX', 'pfeToSed']:
        assert ds[v].attrs['units'] in ['nmol/cm^2/s', 'mmol/m^3 cm/s'], ds[v].attrs['units']
        ds[v] = ds[v] * nmolcm2s_to_mmolm2yr
        ds[v].attrs['units'] = 'mmol m$^{-2}$ yr$^{-1}$'        
    for v in ['IRON_FLUX']:
        assert ds[v].attrs['units'] in ['mmol/m^2/s'], ds[v].attrs['units']
        ds[v] = ds[v] * 86400. * 365. #* 1e4
        ds[v].attrs['units'] = 'mmol m$^{-2}$ yr$^{-1}$'        
            
file_fesedflux = '/glade/p/cesmdata/cseg/inputdata/ocn/pop/gx1v6/forcing/fesedfluxTot_gx1v6_cesm2_2018_c180618.nc'
file_feventflux = '/glade/p/cesmdata/cseg/inputdata/ocn/pop/gx1v6/forcing/feventflux_gx1v6_5gmol_cesm1_97_2017.nc'

dsi = xr.merge((
    xr.open_dataset(file_feventflux).rename({'FESEDFLUXIN': 'Fe_ventflux'}) * µmolm2d_to_mmolm2yr,
    xr.open_dataset(file_fesedflux).rename({'FESEDFLUXIN': 'Fe_sedflux'}) * µmolm2d_to_mmolm2yr,
)).rename({'z': 'z_t', 'y': 'nlat', 'x': 'nlon'})
dsi.Fe_ventflux.attrs['units'] = 'mmol m$^{-2}$ yr$^{-1}$'
dsi.Fe_sedflux.attrs['units'] = 'mmol m$^{-2}$ yr$^{-1}$'
ds = xr.merge((ds, dsi,))
ds['dz'] = ds_grid.dz.drop('z_t') # drop z_t because precision issues cause diffs


dsp = utils.pop_add_cyclic(ds)
for v in dsp.data_vars:
    if 'z_t' in dsp[v].dims:
        dsp[v] = dsp[v].sum('z_t')
dsp.info()


### Spin down your cluster

client.close()
cluster.close()
del client
del cluster

ds

### Use the `discrete_obs` tool to subset ofr a single lat/lon/depth, return a dataframe and its associated region mask

df = discrete_obs.open_datastream('dFe')
df.obs_stream.add_model_field(ds.Fe)
df.obs_stream.add_model_field(ds_grid.REGION_MASK, method='nearest')
df

### Look at the RIV_FLUX calculation

ds.Fe_RIV_FLUX.plot()

dsp.IRON_FLUX.plot()

## 

fields = ['IRON_FLUX', 'Fe_sedflux', 'Fe_ventflux', 'pfeToSed']

log_levels = [0., 0.001]
for scale in 10**np.arange(-3., 1., 1.):
    log_levels.extend(list(np.array([3., 6., 10.]) * scale))
    
levels = {k: log_levels for k in fields}


fig = plt.figure(figsize=(12, 6))
prj = ccrs.Robinson(central_longitude=305.0)

nrow, ncol = 2, 2 
gs = gridspec.GridSpec(
    nrows=nrow, ncols=ncol+1, 
    width_ratios=(1, 1, 0.02),
    wspace=0.15, 
    hspace=0.1,
)

axs = np.empty((nrow, ncol)).astype(object)
caxs= np.empty((nrow, ncol)).astype(object)
for i, j in product(range(nrow), range(ncol)):    
    axs[i, j] = plt.subplot(gs[i, j], projection=prj)
cax = plt.subplot(gs[:, -1])

cmap_field = cmocean.cm.dense


for n, field in enumerate(fields):
    
    i, j = np.unravel_index(n, axs.shape)
    
    ax = axs[i, j]
   
    cf = ax.contourf(
        dsp.TLONG,dsp.TLAT, dsp[field],
        levels=levels[field],
        extend='max',
        cmap=cmap_field,
        norm=colors.BoundaryNorm(levels[field], ncolors=cmap_field.N),
        transform=ccrs.PlateCarree(),
    )

    land = ax.add_feature(
        cartopy.feature.NaturalEarthFeature(
            'physical','land','110m',
            edgecolor='face',
            facecolor='gray'
        )
    )  
                             
    ax.set_title(field) #dsp[field].attrs['title_str'])
cb = plt.colorbar(cf, cax=cax, ticks=log_levels)
if 'units' in dsp[field].attrs:
    cb.ax.set_title(dsp[field].attrs['units'])
    cb.ax.set_yticklabels([f'{f:g}' for f in log_levels])
    
utils.label_plots(fig, [ax for ax in axs.ravel()], xoff=0.02, yoff=0)       
utils.savefig('iron-budget-maps.pdf')

plt.plot(df.dFe_obs, df.Fe, '.')
plt.xlabel('CESM2 dFe [nM]')
plt.ylabel('Obs dFe [nM]')
plt.plot([0, 5], [0, 5], 'r-')



df = discrete_obs.obs_datastream(ds.Fe, 'dFe')
df

