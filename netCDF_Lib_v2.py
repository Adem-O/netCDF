#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import *
from numpy import *
# get_ipython().run_line_magic('matplotlib', 'notebook')
import cartopy.crs as ccrs
import matplotlib.patches as patches
import warnings
from os import listdir
from os.path import isfile, join
from matplotlib.colors import *
from matplotlib import ticker, cm
from matplotlib.ticker import * 
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
import matplotlib.dates as mdates
import imageio.v2 as imageio
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
from matplotlib import container






## Function to plot NETCDF Files from NASA Oceancolor Level 3 and 4 Browser: Diffuse attenutation coefficient (Kd_490)
#### #### #### ####
def KD490_Plot_Data (file, **kwargs):
    ### Read and store .nc file using netCDF4 Package function 'Dataset'
    Kd_data = Dataset(file, format="NETCDF4")
    t0 = Kd_data.time_coverage_start
    tf = Kd_data.time_coverage_end
    period = Kd_data.temporal_range
    DateStart = t0[:-14]
    Dateend = tf[:-14]
    inst = '{} - {}'.format(Kd_data.instrument,Kd_data.platform)
    
    ### Optional: Print Data Information ###
    # print(Kd_data)
    
    ### Optional: Check netCDF Data Keys ###
    # Kd_data.variables.keys()
    
    ### Assuming standard NASA variable naming convetion
    ### Store variables as np.arrays for easy plotting
    Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])
    
    
    ### Overwrite int16 to NaN values
    Kd_490 = np.where(Kd_490 == -32767,NaN,Kd_490)
    
    ### Plotting ###
    fig, ax = plt.subplots(layout="constrained",subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7,7))
    ax.axis('off')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    lat1, lon1 = np.meshgrid(lon,lat)
    ax.set_extent([np.min(lon),np.max(lon),np.max(lat),np.min(lat)])
    cmap_adj = plt.colormaps.get_cmap("rainbow").copy()



    lev_exp = np.arange(np.log10(0.01), np.log10(5), 0.005)
    levs = np.power(10, lev_exp)
    cs = ax.contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
    cs.cmap.set_under('none')
    formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(5,1000))
    cbar = fig.colorbar(cs,orientation='horizontal' , ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)
    
    fig.suptitle('{} data\n From {} to {} ({})'.format(inst,DateStart,Dateend,period),size=16)

    
    cbar.ax.set_xlabel('$\mathrm{K_d}(490) [\mathrm{m}^{-1}]$')
    cs.cmap.set_under('none')
    cs.changed()

    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value),dpi=600)
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')

### Function to return a list of coordinates defining the region of intrest from a single position coordinates
### Input [lon,lat]
### Output [lon(westmost),lon(eastmost),Lat(northmost),Lat(Southmost)]

#### #### #### ####
def CoordsCalc(Area, Pos=[]):
    # Area in [km^2]
    # Position in [Lon, Lat]
    
    
    ### Accurate Conversion
    # longConv = 111.320*np.cos(np.deg2rad(Pos[1]))
    # latConv =  110.574
    
    ### Simple Conversion
    longConv = 111
    latConv =  111
    CalcCoords = np.array([np.round(Pos[0]-(Area/2)*(1/longConv),6), np.round(Pos[0]+(Area/2)*(1/longConv),6),np.round(Pos[1]+(Area/2)*(1/latConv),6), np.round(Pos[1]-(Area/2)*(1/latConv),6)])
    return CalcCoords


### Function to return a list of coordinates defining the region of intrest from A LIST of position coordinates
#### #### #### ####
def CoordsCalcList (Area, List=[]):
    CalcCoordsList = np.empty((0,4))
    for i in range(len(List)):
        Calc = [CoordsCalc(Area,List[i])]
        CalcCoordsList = np.append(CalcCoordsList,Calc,axis=0)
    return CalcCoordsList



## Function to visually plot region of intrest from a list of coordinates ###
#### #### #### ####
def Coords_Check(coords=[], **kwargs):
    
    ### Plotting ###
    fig,axs= plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,7),layout="constrained")
    fig.suptitle('Plotting Given Coordinates',fontsize=25)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].stock_img()
    axs[1].stock_img()
    
    lat_mins = []
    lat_maxs = []
    lon_mins = []
    lon_maxs = []

    for i in range(len(coords)):
        lat_mins = np.append(lat_mins,coords[i][2])
        lat_maxs = np.append(lat_maxs,coords[i][3])
        lon_mins = np.append(lon_mins,coords[i][0])
        lon_maxs = np.append(lon_maxs,coords[i][1])
        
    pad_lon_left = np.min(lon_mins)
    pad_lon_right = np.max(lon_maxs)
    pad_lat_upper = np.max(lat_maxs)
    pad_lat_lower = np.min(lat_mins)
    
    axs[0].set_extent([110,165,-5,-45])
    
    axs[1].set_extent([pad_lon_left*0.95,pad_lon_right*1.05,pad_lat_upper*0.9,pad_lat_lower*1.1])
    
    for i in range(len(coords)):
        h = np.round(coords[i][3] - coords[i][2],4)
        w = np.round(coords[i][1] - coords[i][0],4)
        c_lat = (coords[i][1] + coords[i][0])/2
        c_lon = (coords[i][3] + coords[i][2])/2

        axs[0].scatter(c_lat,c_lon,s=80,marker='x',color='r')
        axs[0].annotate('Station {}'.format(i+1),(c_lat-6,c_lon-0.3))
        
        rect = patches.Rectangle((coords[i][0],coords[i][2]), w, h, linewidth=2, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)    
        axs[1].scatter(c_lat,c_lon,s=20,marker='x',color='r')

    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')
            
#### #### #### ####           
def Coords_Check_avg(folderpath,coords=[],**kwargs):
    
    ### Plotting ###
    fig,axs= plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,7),layout="constrained")
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].stock_img()
    axs[1].stock_img()
    
    lat_mins = []
    lat_maxs = []
    lon_mins = []
    lon_maxs = []

    for i in range(len(coords)):
        lat_mins = np.append(lat_mins,coords[i][2])
        lat_maxs = np.append(lat_maxs,coords[i][3])
        lon_mins = np.append(lon_mins,coords[i][0])
        lon_maxs = np.append(lon_maxs,coords[i][1])
        
    pad_lon_left = np.min(lon_mins)
    pad_lon_right = np.max(lon_maxs)
    pad_lat_upper = np.max(lat_maxs)
    pad_lat_lower = np.min(lat_mins)
    
    axs[0].set_extent([110,165,-5,-45])
    
    axs[1].set_extent([pad_lon_left*0.95,pad_lon_right*1.05,pad_lat_upper*0.9,pad_lat_lower*1.1])
    
    for i in range(len(coords)):
        h = np.round(coords[i][3] - coords[i][2],4)
        w = np.round(coords[i][1] - coords[i][0],4)
        c_lat = (coords[i][1] + coords[i][0])/2
        c_lon = (coords[i][3] + coords[i][2])/2

        axs[0].scatter(c_lat,c_lon,s=80,marker='x',color='r')
        axs[0].annotate('Station {}'.format(i+1),(c_lat-6,c_lon-0.3))
        avg = np.round(np.nanmean(MultiFile_Reg_avg(folderpath, coords[i])[0].astype('f8')),5)
        axs[1].annotate(r'$\mathrm{K_d}(490)$ Avg = '+'{}'.format(avg),(c_lat+1,c_lon-0.3))
        rect = patches.Rectangle((coords[i][0],coords[i][2]), w, h, linewidth=2, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)    
        axs[1].scatter(c_lat,c_lon,s=14,marker='x',color='r')
    h_km = np.round(111*np.abs(h),0)
    w_km = np.round(111*np.abs(w),0)
    fig.suptitle('Plotting Given Coordinates (Data Period: {})\n Area: {} x {} [km]'.format(MultiFile_Reg_avg(folderpath, coords[0])[4][0],w_km,h_km),fontsize=25)
    
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value),dpi=600)
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')
## Function to plot NETCDF Files from NASA Oceancolor Level 3 and 4 Browser: Diffuse attenutation coefficient (Kd_490)
## Plot data over a specific region

# def KD490_Plot_Data_Region (file, coords =[], **kwargs):
#     ### Read and store .nc file using netCDF4 Package function 'Dataset'
#     Kd_data = Dataset(r"{}".format(file), format="NETCDF4")
    
    
#     ### Optional: Print Data Information ###
#     # print(Kd_data)
    
#     ### Optional: Check netCDF Data Keys ###
#     # Kd_data.variables.keys()
    
#     ### Assuming standard NASA variable naming convetion
#     ### Store variables as np.arrays for easy plotting
#     Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])
    
    
#     ### Overwrite int16 to NaN values
#     Kd_490 = np.where(Kd_490 == -32767,NaN,Kd_490)
    
    
#     ### Slicing Data for specified region
#     Left = coords[0]
#     Right = coords[1]
#     Upper = coords[2]
#     Lower = coords[3]

#     lat_uslice = np.min([i for i in range(len(lat)) if lat[i] < Upper])
#     lat_lslice = np.max([i for i in range(len(lat)) if lat[i] >= Lower])
#     lon_wslice = np.min([i for i in range(len(lon)) if lon[i] >= Left])
#     lon_eslice = np.max([i for i in range(len(lon)) if lon[i] < Right])

#     Kd_490_reduced = Kd_490[lat_uslice:lat_lslice, lon_wslice:lon_eslice]
#     lat_reduced = lat[lat_uslice:lat_lslice]
#     lon_reduced = lon[lon_wslice:lon_eslice]
    
#     t0 = Kd_data.time_coverage_start
#     tf = Kd_data.time_coverage_end
#     period = Kd_data.temporal_range
#     DateStart = t0[:-14]
#     Dateend = tf[:-14]
#     inst = '{} - {}'.format(Kd_data.instrument,Kd_data.platform)
        
#     ### Plotting ###
#     fig,axs= plt.subplots(1,3,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,7),layout="constrained")
    
#     fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n'.format(inst,DateStart,Dateend,period,coords[0],coords[1],coords[2],coords[3]),size=16)
    

#     axs[0].axis('off')
#     axs[1].axis('off')
#     axs[2].axis('off')
#     axs[0].stock_img()
#     axs[1].stock_img()
#     axs[2].stock_img()
#     pad_lon1 = (np.max(lon_reduced) - np.min(lon_reduced))*8
#     pad_lat1 = (np.max(lat_reduced) - np.min(lat_reduced))*8
#     pad_lon2 = (np.max(lon_reduced) - np.min(lon_reduced))*0.2
#     pad_lat2 = (np.max(lat_reduced) - np.min(lat_reduced))*0.2
#     lat1, lon1 = np.meshgrid(lon,lat)
    
#     cmap_adj = plt.colormaps.get_cmap("rainbow").copy()
#     lev_exp = np.arange(np.log10(np.nanmin(Kd_490)),np.log10(np.nanmax(Kd_490)),0.01)
#     levs = np.power(10, lev_exp)
#     cs1 = axs[0].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
#     cs1.cmap.set_under('none')
#     formatter = LogFormatter(10, labelOnlyBase=False)
#     cbar1 = fig.colorbar(cs1,fraction=0.046, pad=0.05,ticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5], format=formatter)

#     cmap_adj1 = plt.colormaps.get_cmap("inferno").copy()
#     lat1_reduced, lon1_reduced = np.meshgrid(lon_reduced,lat_reduced)
#     cs1 =  axs[0].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced,700, cmap=cmap_adj1, extend='both',norm='log')
#     axs[0].set_extent([np.min(lon),np.max(lon),np.max(lat),np.min(lat)])
#     cs1.cmap.set_under('none')
    
#     cs2 = axs[1].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
#     cs2 =  axs[1].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced,700, cmap=cmap_adj1, extend='both',norm='log')
#     axs[1].set_extent([np.min(lon_reduced)-pad_lon1,np.max(lon_reduced)+pad_lon1,np.min(lat_reduced)-pad_lat1,np.max(lat_reduced)+pad_lat1])
    
#     cs3 = axs[2].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
#     cs3 =  axs[2].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced,700, cmap=cmap_adj1, extend='both',norm='log')
#     axs[2].set_extent([np.min(lon_reduced)-pad_lon2,np.max(lon_reduced)+pad_lon2,np.min(lat_reduced)-pad_lat2,np.max(lat_reduced)+pad_lat2])
    
#     h = np.max(lat_reduced) - np.min(lat_reduced)
#     w = np.max(lon_reduced) - np.min(lon_reduced)
    
#     ext_h = (np.max(lat_reduced)+pad_lat1)-(np.min(lat_reduced)-pad_lat1)
#     ext_w = (np.max(lon_reduced)+pad_lon1)-(np.min(lon_reduced)-pad_lon1)
#     rect = patches.Rectangle((np.min(np.min(lon_reduced)-pad_lon1),np.min(np.min(lat_reduced)-pad_lat1)), ext_h,ext_w, linewidth=0.5, edgecolor='red', facecolor='none')
#     axs[0].add_patch(rect)
#     rect = patches.Rectangle((np.min(lon_reduced),np.min(lat_reduced)), w, h, linewidth=1, edgecolor='black', facecolor='none')
#     axs[1].add_patch(rect)
#     rect = patches.Rectangle((np.min(lon_reduced),np.min(lat_reduced)), w, h, linewidth=2.5, edgecolor='black', facecolor='none')
#     axs[2].add_patch(rect)
#     cbar = fig.colorbar(cs3,fraction=0.046, pad=0.05)
    
#     Avg = KD490_Region_Avg(file, coords)[0]
#     NaNs = KD490_Region_Avg(file, coords)[1]
#     NanAvg = (NaNs/KD490_Region_Avg(file, coords)[5])*100
#     axs[2].set_title(r'Avg $\mathrm{K_d}(490)$ = '+'{:.5f}\n Empty data points = {} ({:.1f}%)'.format(Avg,NaNs,NanAvg))
#     cbar.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')
#     cbar1.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')

#     cs1.changed()
#     cs2.changed()
#     cs3.changed()
    
#     for key, value in kwargs.items():
#         if key == 'save' :
#             print('Figure Saved Successfully')
#             fig.savefig('{}'.format(value),dpi=600)
#         else:
#             print('FIGURE NOT SAVED! Check kwarg is correct')    
    
# ### Calculate the average values of Kd490 for a given file and coordinates       

def KD490_Plot_Data_Region(file, coords=[], **kwargs):
    # Read and store .nc file using netCDF4 Package function 'Dataset'
    Kd_data = Dataset(r"{}".format(file), format="NETCDF4")

    # Optional: Print Data Information
    # print(Kd_data)

    # Assuming standard NASA variable naming convention
    # Store variables as np.arrays for easy plotting
    Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])

    # Overwrite int16 to NaN values
    Kd_490 = np.where(Kd_490 == -32767, np.nan, Kd_490)
#     Kd_490 = ma.masked_where(Kd_490 == -32767, Kd_490)
    # Slicing Data for specified region
    Left, Right, Upper, Lower = coords

    lat_uslice = np.min([i for i in range(len(lat)) if lat[i] < Upper])
    lat_lslice = np.max([i for i in range(len(lat)) if lat[i] >= Lower])
    lon_wslice = np.min([i for i in range(len(lon)) if lon[i] >= Left])
    lon_eslice = np.max([i for i in range(len(lon)) if lon[i] < Right])

    Kd_490_reduced = Kd_490[lat_uslice:lat_lslice, lon_wslice:lon_eslice]
    lat_reduced = lat[lat_uslice:lat_lslice]
    lon_reduced = lon[lon_wslice:lon_eslice]

    t0 = Kd_data.time_coverage_start
    tf = Kd_data.time_coverage_end
    period = Kd_data.temporal_range
    DateStart = t0[:-14]
    Dateend = tf[:-14]
    inst = '{} - {}'.format(Kd_data.instrument, Kd_data.platform)

    # Plotting
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7), layout="constrained")

    fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n'.format(inst, DateStart, Dateend, period, coords[0], coords[1], coords[2], coords[3]), size=16)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[0].stock_img()
    axs[1].stock_img()
    axs[2].stock_img()
    pad_lon1 = (np.max(lon_reduced) - np.min(lon_reduced)) * 8
    pad_lat1 = (np.max(lat_reduced) - np.min(lat_reduced)) * 8
    pad_lon2 = (np.max(lon_reduced) - np.min(lon_reduced)) * 0.2
    pad_lat2 = (np.max(lat_reduced) - np.min(lat_reduced)) * 0.2
    lat1, lon1 = np.meshgrid(lon, lat)




    cmap_adj = plt.colormaps.get_cmap("rainbow").copy()
    lev_exp = np.arange(np.log10(0.01), np.log10(5), 0.01)
    levs = np.power(10, lev_exp)
    cs1 = axs[0].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
    cs1.cmap.set_under('none')
    formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(5,1000))
    cbar1 = fig.colorbar(cs1, fraction=0.046, pad=0.05, ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)

    cmap_adj1 = plt.colormaps.get_cmap("inferno").copy()
    lat1_reduced, lon1_reduced = np.meshgrid(lon_reduced, lat_reduced)
    cs1 = axs[0].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
    axs[0].set_extent([np.min(lon), np.max(lon), np.max(lat), np.min(lat)])
    cs1.cmap.set_under('none')

    cs2 = axs[1].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
    cs2 = axs[1].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
    axs[1].set_extent([np.min(lon_reduced) - pad_lon1, np.max(lon_reduced) + pad_lon1, np.min(lat_reduced) - pad_lat1, np.max(lat_reduced) + pad_lat1])
    

    cs3 = axs[2].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
    cs3 = axs[2].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
    axs[2].set_extent([np.min(lon_reduced) - pad_lon2, np.max(lon_reduced) + pad_lon2, np.min(lat_reduced) - pad_lat2, np.max(lat_reduced) + pad_lat2])

    h = np.max(lat_reduced) - np.min(lat_reduced)
    w = np.max(lon_reduced) - np.min(lon_reduced)

    ext_h = (np.max(lat_reduced) + pad_lat1) - (np.min(lat_reduced) - pad_lat1)
    ext_w = (np.max(lon_reduced) + pad_lon1) - (np.min(lon_reduced) - pad_lon1)
    rect = patches.Rectangle((np.min(np.min(lon_reduced) - pad_lon1), np.min(np.min(lat_reduced) - pad_lat1)), ext_w, ext_h, linewidth=0.5, edgecolor='red', facecolor='none')
    axs[0].add_patch(rect)
    rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=1, edgecolor='black', facecolor='none')
    axs[1].add_patch(rect)
    rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=2.5, edgecolor='black', facecolor='none')
    axs[2].add_patch(rect)
    
    ticks_finer = np.logspace(np.log10(0.01),np.log10(5),80,endpoint=True)
    cbar = fig.colorbar(cs3, fraction=0.046, pad=0.05, ticks=ticks_finer, format=formatter)
    
    Avg = KD490_Region_Avg(file, coords)[0]
    NaNs = KD490_Region_Avg(file, coords)[1]
    NanAvg = (NaNs / KD490_Region_Avg(file, coords)[5]) * 100
    axs[2].set_title(r'Avg $\mathrm{K_d}(490)$ = '+'{:.5f}\n Empty data points = {} ({:.1f}%)'.format(Avg, NaNs, NanAvg))
    cbar.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')
    cbar1.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')

    cs1.changed()
    cs2.changed()
    cs3.changed()
    

    for key, value in kwargs.items():
        if key =='grid' and value == True:
            fig1, ax = plt.subplots(layout='constrained')
            ax.set_aspect('equal')
            ax.apply_aspect()
#             MarkerS = (ax.get_position().transformed(fig.transFigure).width*72/fig1.dpi)/(len(Kd_490_reduced[0][:])+1)
            scat = ax.scatter(lat1_reduced, lon1_reduced,c=Kd_490_reduced,cmap=cmap_adj1,s=200, marker='s')
            deg_dist = lat1_reduced[0][1] - lat1_reduced[0][0]
            kmDist = np.round(111*deg_dist,1)
            cbarscat = fig1.colorbar(scat , fraction=0.046, pad=0.05)
            ax.set_title('Data Points: {}\nResolution '.format(KD490_Region_Avg(file, coords)[5])+r'$\approx$'+'{} km'.format(kmDist))
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            cbarscat.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')      
#             fig1.savefig('savedfigs/LatexFigs/GridRegion.png', dpi=600)

        if key == 'save':
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value), dpi=600)

# #### #### ####
# def KD490_Plot_Data_Region(file, coords=[], **kwargs):
#     # Read and store .nc file using netCDF4 Package function 'Dataset'
#     Kd_data = Dataset(r"{}".format(file), format="NETCDF4")

#     # Optional: Print Data Information
#     # print(Kd_data)

#     # Assuming standard NASA variable naming convention
#     # Store variables as np.arrays for easy plotting
#     Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])

#     # Overwrite int16 to NaN values
#     Kd_490 = np.where(Kd_490 == -32767, np.nan, Kd_490)

#     # Slicing Data for specified region
#     Left, Right, Upper, Lower = coords

#     lat_uslice = np.min([i for i in range(len(lat)) if lat[i] < Upper])
#     lat_lslice = np.max([i for i in range(len(lat)) if lat[i] >= Lower])
#     lon_wslice = np.min([i for i in range(len(lon)) if lon[i] >= Left])
#     lon_eslice = np.max([i for i in range(len(lon)) if lon[i] < Right])

#     Kd_490_reduced = Kd_490[lat_uslice:lat_lslice, lon_wslice:lon_eslice]
#     lat_reduced = lat[lat_uslice:lat_lslice]
#     lon_reduced = lon[lon_wslice:lon_eslice]

#     t0 = Kd_data.time_coverage_start
#     tf = Kd_data.time_coverage_end
#     period = Kd_data.temporal_range
#     DateStart = t0[:-14]
#     Dateend = tf[:-14]
#     inst = '{} - {}'.format(Kd_data.instrument, Kd_data.platform)

#     # Plotting
#     fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7), layout="constrained")

#     fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n'.format(inst, DateStart, Dateend, period, coords[0], coords[1], coords[2], coords[3]), size=16)

#     axs[0].axis('off')
#     axs[1].axis('off')
#     axs[2].axis('off')
#     axs[0].stock_img()
#     axs[1].stock_img()
#     axs[2].stock_img()
#     pad_lon1 = (np.max(lon_reduced) - np.min(lon_reduced)) * 8
#     pad_lat1 = (np.max(lat_reduced) - np.min(lat_reduced)) * 8
#     pad_lon2 = (np.max(lon_reduced) - np.min(lon_reduced)) * 0.2
#     pad_lat2 = (np.max(lat_reduced) - np.min(lat_reduced)) * 0.2
#     lat1, lon1 = np.meshgrid(lon, lat)




#     cmap_adj = plt.colormaps.get_cmap("rainbow").copy()
#     lev_exp = np.arange(np.log10(0.01), np.log10(5), 0.01)
#     levs = np.power(10, lev_exp)
#     cs1 = axs[0].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
#     cs1.cmap.set_under('none')
#     formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(5,1000))
#     cbar1 = fig.colorbar(cs1, fraction=0.046, pad=0.05, ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)

#     cmap_adj1 = plt.colormaps.get_cmap("inferno").copy()
#     lat1_reduced, lon1_reduced = np.meshgrid(lon_reduced, lat_reduced)
#     cs1 = axs[0].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
#     axs[0].set_extent([np.min(lon), np.max(lon), np.max(lat), np.min(lat)])
#     cs1.cmap.set_under('none')

#     cs2 = axs[1].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
#     cs2 = axs[1].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
#     axs[1].set_extent([np.min(lon_reduced) - pad_lon1, np.max(lon_reduced) + pad_lon1, np.min(lat_reduced) - pad_lat1, np.max(lat_reduced) + pad_lat1])
    

#     cs3 = axs[2].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
#     cs3 = axs[2].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
#     axs[2].set_extent([np.min(lon_reduced) - pad_lon2, np.max(lon_reduced) + pad_lon2, np.min(lat_reduced) - pad_lat2, np.max(lat_reduced) + pad_lat2])

#     h = np.max(lat_reduced) - np.min(lat_reduced)
#     w = np.max(lon_reduced) - np.min(lon_reduced)

#     ext_h = (np.max(lat_reduced) + pad_lat1) - (np.min(lat_reduced) - pad_lat1)
#     ext_w = (np.max(lon_reduced) + pad_lon1) - (np.min(lon_reduced) - pad_lon1)
#     rect = patches.Rectangle((np.min(np.min(lon_reduced) - pad_lon1), np.min(np.min(lat_reduced) - pad_lat1)), ext_w, ext_h, linewidth=0.5, edgecolor='red', facecolor='none')
#     axs[0].add_patch(rect)
#     rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=1, edgecolor='black', facecolor='none')
#     axs[1].add_patch(rect)
#     rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=2.5, edgecolor='black', facecolor='none')
#     axs[2].add_patch(rect)
#     ticks_finer = np.logspace(np.log10(0.01),np.log10(5),200,endpoint=True)
#     cbar = fig.colorbar(cs3, fraction=0.046, pad=0.05, ticks=ticks_finer, format=formatter)
    
#     Avg = KD490_Region_Avg(file, coords)[0]
#     NaNs = KD490_Region_Avg(file, coords)[1]
#     NanAvg = (NaNs / KD490_Region_Avg(file, coords)[5]) * 100
#     axs[2].set_title(r'Avg $\mathrm{K_d}(490)$ = '+'{:.5f}\n Empty data points = {} ({:.1f}%)'.format(Avg, NaNs, NanAvg))
#     cbar.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')
#     cbar1.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')

#     cs1.changed()
#     cs2.changed()
#     cs3.changed()

#     for key, value in kwargs.items():
#         if key == 'save':
#             print('Figure Saved Successfully')
#             fig.savefig('{}'.format(value), dpi=600)
#         else:
#             print('FIGURE NOT SAVED! Check kwarg is correct')
            
#### #### #### ####
def KD490_Region_Avg (file, coords =[]):
    ### Read and store .nc file using netCDF4 Package function 'Dataset'
    Kd_data = Dataset(r"{}".format(file), format="NETCDF4")
    t0 = Kd_data.time_coverage_start
    tf = Kd_data.time_coverage_end
    period = Kd_data.temporal_range
    inst = '{} - {}'.format(Kd_data.instrument,Kd_data.platform)
    ### Optional: Print Data Information ###
    # print(Kd_data)
    
    ### Optional: Check netCDF Data Keys ###
    # Kd_data.variables.keys()
    
    ### Assuming standard NASA variable naming convetion
    ### Store variables as np.arrays for easy plotting
    Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])
    
    
    ### Overwrite int16 to NaN values
    Kd_490 = np.where(Kd_490 == -32767,NaN,Kd_490)
    
    
    ### Slicing Data for specified region
    Left = coords[0]
    Right = coords[1]
    Upper = coords[2]
    Lower = coords[3]

    lat_uslice = np.min([i for i in range(len(lat)) if lat[i] < Upper])
    lat_lslice = np.max([i for i in range(len(lat)) if lat[i] >= Lower])
    lon_wslice = np.min([i for i in range(len(lon)) if lon[i] >= Left])
    lon_eslice = np.max([i for i in range(len(lon)) if lon[i] < Right])

    Kd_490_reduced = Kd_490[lat_uslice:lat_lslice, lon_wslice:lon_eslice]
    lat_reduced = lat[lat_uslice:lat_lslice]
    lon_reduced = lon[lon_wslice:lon_eslice]
    
    a = np.nanmean(Kd_490_reduced)
    b = (np.count_nonzero(np.isnan(Kd_490_reduced)))
    c = len(Kd_490_reduced)*len(Kd_490_reduced[0])
    return [a,b,t0, tf, period, c, inst]




### Calculate the average value in a given region over an extended period of time (multiple files)


#### #### #### ####
def MultiFile_Reg_avg (folderpath, coords=[]):
    ### Function to take an average of the Kd490 data at coords.
    ### Output [0] : Kd490 Average over chosen region
    ### Output [1] : Number of empty pixels in chosen region
    ### Output [2] : Time Data Capture started
    ### Output [3] : Time Data Capture end
    ### Output [4] : Period of data capture
    from os import listdir
    from os.path import isfile, join
    mypath = folderpath
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
    x = np.array([])
    y = np.array([])
    t0 = np.array([])
    tf = np.array([])
    period = np.array([])
    Date = np.array([])
    DataNum = np.array([])
    inst = np.array([])
    for i in range(len(files)):
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        x = np.append(x,KD490_Region_Avg(r"{}".format(loc),coords)[0]) ## Avg Value
        y = np.append(y,KD490_Region_Avg(r"{}".format(loc),coords)[1]) ## No. of NaN
        t0 = np.append(t0,KD490_Region_Avg(r"{}".format(loc),coords)[2]) ## Time Begun
        tf = np.append(tf,KD490_Region_Avg(r"{}".format(loc),coords)[3]) ## Time Ended
        period = np.append(period,KD490_Region_Avg(r"{}".format(loc),coords)[4]) ## Data Capture length
        Date = np.append(Date,KD490_Region_Avg(r"{}".format(loc),coords)[2][:-14]) ## Date of capture start
        DataNum = np.append(DataNum,KD490_Region_Avg(r"{}".format(loc),coords)[5]) ## Number of Data Points in region
        inst = np.append(inst,KD490_Region_Avg(r"{}".format(loc),coords)[6]) ## instrument
    return np.array([x,y,t0,tf,period,Date,DataNum,inst])




### Read the list of .nc files in a given local file path
#### #### #### ####
def MultiFile_FileNames (folderpath):
    mypath = folderpath
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
    file_names = []
    for i in range(len(files)):
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        file_names = np.append(file_names,loc)
    return file_names




### Plot the average value of the data over a certain region over an extended period of time (multiple files)
#### #### #### ####
def MutiFile_Data_plotting (folderpath, coords=[],**kwargs):
    ## Call Data generating Function
    many_avg = MultiFile_Reg_avg (folderpath, coords)
    num_files = len(MultiFile_FileNames(folderpath))
    if num_files <= 30:
        spacing = 1
    elif num_files <= 230:
        spacing = 8
    elif num_files > 230:
        spacing = 14
    ## Plot data
    avg = many_avg[0].astype('f8')
    Nans = (many_avg[1].astype('f8')/many_avg[6].astype('f8'))*100
    
    fig, axs = plt.subplots(2, figsize=(8,10),layout="constrained")
    fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n'.format(many_avg[7][0],many_avg[5][0],many_avg[5][-1],many_avg[4][0],coords[0],coords[1],coords[2],coords[3]),size=16)

    axs[0].set_title(r'$\mathrm{K}_\mathrm{d}(490)$: Averaged over target region'+'\n Variance = {:.3f}'.format(np.nanvar(avg)*10**3)+r'$\cdot 10^{-3}$')
    axs[0].plot(many_avg[5],avg)
    axs[0].set_xticks(many_avg[5][::spacing])
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
    axs[0].set_ylabel('$\mathrm{K}_\mathrm{d}(490) [\mathrm{m}^{1}]$')
    
    axs[1].set_title('\n Percentage of viable data in region of interest')
    axs[1].bar(many_avg[5],100 - Nans)
    axs[1].set_xticks(many_avg[5][::spacing])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')   

### Compute the variance in the data over multiple regions of intrest through time
#### #### #### ####
def MultiPos_DataVar(CoordsList, folderpath):
    var = np.array([])
    for i in range(len(CoordsList)):
        coords = CoordsList[i]
        var = np.append(var,np.nanvar(MultiFile_Reg_avg (folderpath, coords)[0].astype('f8')))
    return var

# def MultiPos_DataVar_MultiCoords(CoordsList, folderpath):
#     var = np.array([])
#     for i in range(len(CoordsList[0])):
#         for j in range(len(CoordsList[0][0])):
#             coords = CoordsList[i][j]
#             var = np.append(var,np.nanvar(MultiFile_Reg_avg (folderpath, coords)[0].astype('f8')))
#     return var

### Plot the variance in the data over multiple regions of intrest through time

def MultiPos_DataVar_Plot(CoordsList,PosList, folderpath,**kwargs):
    Var = MultiPos_DataVar(CoordsList, folderpath)
    many_avg = MultiFile_Reg_avg (folderpath, CoordsList[0])
    fig, ax = plt.subplots()
    Pos_String = []
    for i in range (len(PosList)):
        Pos_String = np.append(Pos_String,'{}'.format(PosList[i]))
    ax.plot(Pos_String ,Var*10**3)
    fig.suptitle('{} data variance\n From {} to {} ({})\n'.format(many_avg[7][0],many_avg[5][0],many_avg[5][-1],many_avg[4][0]))
    ax.set_xlabel('Position Coordinates [Lon, Lat]')
    ax.set_ylabel(r'Variance [$10^{-3}$]')
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')    
    
### Plot the data for multiple data sets(instruments/satelites) over a period of time (multiple files).
#### #### #### ####
def MutiFile_MultiInstr_Data_plotting (folderpaths=[], coords=[], **kwargs):
    fig, axs = plt.subplots(2, figsize=(14,10),layout="constrained")
    Total_dates = []
    for i in range(len(folderpaths)):
        first_file = MultiFile_FileNames(folderpaths[i])[0]
        last_file = MultiFile_FileNames(folderpaths[i])[-1]
        Total_dates = np.append(Total_dates, KD490_Region_Avg(first_file, coords)[2][:-14])
        Total_dates = np.append(Total_dates, KD490_Region_Avg(last_file, coords)[2][:-14])

    Total_dates_dtype = np.sort(np.array(Total_dates,dtype='datetime64'))
    Total_dates_list = np.arange(Total_dates_dtype[0], Total_dates_dtype[-1], dtype='datetime64')
    
    if len(Total_dates_list) <= 30:
        spacing = 1
    elif len(Total_dates_list) <= 230:
        spacing = 20
    elif len(Total_dates_list) > 230:
        spacing = 100
        
    for i in range(len(folderpaths)):
        folderpath = folderpaths[i]
        ## Call Data generating Function
        many_avg = MultiFile_Reg_avg (folderpath, coords)

        ## Plot data
        avg = many_avg[0].astype('f8')
        Nans = (many_avg[1].astype('f8')/many_avg[6].astype('f8'))*100
        Dates = np.array(many_avg[5],dtype='datetime64')
        axs[0].set_title(r'$\mathrm{K}_\mathrm{d}(490)$: Averaged over target region'+'\n Multi Instrument')
        axs[0].plot(Dates,avg, label = ('{} data\n {} to {}\n ({})'.format(many_avg[7][0],many_avg[5][0],
                                                                                    many_avg[5][-1],many_avg[4][0])))
        axs[0].set_xticks(Total_dates_list[::spacing])
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right')
        axs[0].set_ylabel('$\mathrm{K}_\mathrm{d}(490) [\mathrm{m}^{1}]$')
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=10)
        
        
        axs[1].set_title('\n Percentage of empty data points captured')
        axs[1].bar(Dates,Nans, label = ('{} \n {} to {}\n ({})'.format(many_avg[7][0],many_avg[5][0],
                                                                                    many_avg[5][-1],many_avg[4][0])))
        axs[1].set_xticks(Total_dates_list[::spacing])
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=10)
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')
        
### Plot the variance in the data over multiple regions of intrest from multiple instruments
#### #### #### ####
def MultiPos_MultiInstr_DataVar_Plot(CoordsList,PosList, folderpaths=[],**kwargs):
    fig, ax = plt.subplots(1, figsize=(14,10),layout="constrained")
    for i in range(len(folderpaths)):
        Var = MultiPos_DataVar(CoordsList, folderpaths[i])
        many_avg = MultiFile_Reg_avg (folderpaths[i], CoordsList[0])
        Pos_String = []
        for i in range (len(PosList)):
            Pos_String = np.append(Pos_String,'{}'.format(PosList[i]))
        ax.plot(Pos_String ,Var*10**3,label=('{}'.format(many_avg[7][0])))
       
    fig.suptitle('Variance vs Position \n Multi Instrument \n {} to {} ({})\n'.format(many_avg[5][0], many_avg[5][-1],many_avg[4][0]))
    ax.set_xlabel('Position Coordinates [Lon, Lat]')
    ax.set_ylabel(r'Variance [$10^{-3}$]')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=10)
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')
    
### Plot the data through time at multiple regions of interest
#### #### #### ####
def MultiPos_Data_plot(CoordsList,PosList,folderpath, **kwargs):
    fig, ax = plt.subplots(figsize=(14,10),layout="constrained")
    for i in range(len(CoordsList)):
        coords = CoordsList[i]
        many_avg = MultiFile_Reg_avg (folderpath, coords)
        num_files = len(MultiFile_FileNames(folderpath))
        if num_files <= 30:
            spacing = 1
        elif num_files <= 230:
            spacing = 8
        elif num_files > 230:
            spacing = 14
        ## Plot data
        avg = many_avg[0].astype('f8')
        ax.plot(many_avg[5],avg,label=('{}').format(PosList[i]))
    fig.suptitle('{} data\n From {} to {} ({})\n'.format(many_avg[7][0],many_avg[5][0],many_avg[5][-1],many_avg[4][0]),size=16)
    ax.set_xticks(many_avg[5][::spacing])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('$\mathrm{K}_\mathrm{d}(490) [\mathrm{m}^{1}]$')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=10)
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')


#########################################################    
### Select files only within a certain window of time ###
#########################################################
############# Only works with two folders ###############
#########################################################    


def DateSelection(FolderPath_DataFolder1,FolderPath_DataFolder2,EarliestDate,LatestDate):
    def find_nearest(array, value):
#         array = np.asarray(array)
        idx = (np.abs(array - value))
        return np.argmin(idx)
    
    Folder1_StartDates = np.datetime64()
    Folder1_EndDates = np.datetime64()
    for i in range(len(MultiFile_FileNames(FolderPath_DataFolder1))):
        mypath = FolderPath_DataFolder1
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        Kd_data = Dataset(r"{}".format(loc), format="NETCDF4")
        Folder1_StartDates = np.append(Folder1_StartDates,np.datetime64(Kd_data.time_coverage_start[:10]))
        Folder1_EndDates = np.append(Folder1_EndDates,np.datetime64(Kd_data.time_coverage_end[:10]))
    
    F1_delta = np.abs(np.datetime64(Kd_data.time_coverage_end[:10]) - np.datetime64(Kd_data.time_coverage_start[:10]))
    
    Folder2_StartDates = np.datetime64()
    Folder2_EndDates = np.datetime64()
    for i in range(len(MultiFile_FileNames(FolderPath_DataFolder2))):
        mypath = FolderPath_DataFolder2
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        Kd_data = Dataset(r"{}".format(loc), format="NETCDF4")
        Folder2_StartDates = np.append(Folder2_StartDates,np.datetime64(Kd_data.time_coverage_start[:10]))
        Folder2_EndDates = np.append(Folder2_EndDates,np.datetime64(Kd_data.time_coverage_end[:10]))
    
    F2_delta = np.abs(np.datetime64(Kd_data.time_coverage_end[:10]) - np.datetime64(Kd_data.time_coverage_start[:10]))

    
    
    F1_D0 = find_nearest(Folder1_StartDates[1:],np.datetime64(EarliestDate)) ## First date for folder one
    F1_Df = find_nearest(Folder1_StartDates[1:],np.datetime64(LatestDate)) ## Final date for folder one
    F2_D0 = find_nearest(Folder2_StartDates[1:],np.datetime64(EarliestDate)) ## First date for folder two
    F2_DF = find_nearest(Folder2_StartDates[1:],np.datetime64(LatestDate)) ## Final date for folder two
    
    
    return [F1_D0,F1_Df,F2_D0,F2_DF,F1_delta.astype('i8'), F2_delta.astype('i8')]

################################################################
################################################################

def MultiFile_Reg_avg_reduced (start,stop,folderpath, coords=[]):
    ### Function to take an average of the Kd490 data at coords.
    ### Output [0] : Kd490 Average over chosen region
    ### Output [1] : Number of empty pixels in chosen region
    ### Output [2] : Time Data Capture started
    ### Output [3] : Time Data Capture end
    ### Output [4] : Period of data capture
    from os import listdir
    from os.path import isfile, join
    mypath = folderpath
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
    files = files[start:stop]
    x = np.array([])
    y = np.array([])
    t0 = np.array([])
    tf = np.array([])
    period = np.array([])
    Date = np.array([])
    DataNum = np.array([])
    inst = np.array([])
    Date_end = np.array([])
    for i in range(len(files)):
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        x = np.append(x,KD490_Region_Avg(r"{}".format(loc),coords)[0]) ## Avg Value
        y = np.append(y,KD490_Region_Avg(r"{}".format(loc),coords)[1]) ## No. of NaN
        t0 = np.append(t0,KD490_Region_Avg(r"{}".format(loc),coords)[2]) ## Time Begun
        tf = np.append(tf,KD490_Region_Avg(r"{}".format(loc),coords)[3]) ## Time Ended
        period = np.append(period,KD490_Region_Avg(r"{}".format(loc),coords)[4]) ## Data Capture length
        Date = np.append(Date,KD490_Region_Avg(r"{}".format(loc),coords)[2][:-14]) ## Date of capture start
        Date_end = np.append(Date_end,KD490_Region_Avg(r"{}".format(loc),coords)[3][:-14]) ## Date of capture end
        DataNum = np.append(DataNum,KD490_Region_Avg(r"{}".format(loc),coords)[5]) ## Number of Data Points in region
        inst = np.append(inst,KD490_Region_Avg(r"{}".format(loc),coords)[6]) ## instrument
    return np.array([x,y,t0,tf,period,Date,DataNum,inst,Date_end])

def Compare_dataBinning_plot (EarliestDate,LatestDate, folderpaths=[], coords=[], **kwargs):
    fig, axs = plt.subplots(2, figsize=(14,10),layout="constrained")
    Dates_Reduc = DateSelection(folderpaths[0],folderpaths[1],EarliestDate,LatestDate)
    
    
    if Dates_Reduc[4] < Dates_Reduc[5]:
        
        many_avg_1 = MultiFile_Reg_avg_reduced (Dates_Reduc[0],Dates_Reduc[1],folderpaths[0], coords)
        many_avg_2 = MultiFile_Reg_avg_reduced (Dates_Reduc[2],Dates_Reduc[3],folderpaths[1], coords)
        
        Dates_minor = np.array(many_avg_1[5],dtype='datetime64')
        Dates_major = np.array(many_avg_2[5],dtype='datetime64')
        Dates_major_end = np.array(many_avg_2[8],dtype='datetime64')

        avg_1 = many_avg_1[0].astype('f8')
        Nans_1 = (many_avg_1[1].astype('f8')/many_avg_1[6].astype('f8'))*100

        avg_2 = many_avg_2[0].astype('f8')
        Nans_2 = (many_avg_2[1].astype('f8')/many_avg_2[6].astype('f8'))*100
        
    else :
        many_avg_1 = MultiFile_Reg_avg_reduced (Dates_Reduc[2],Dates_Reduc[3],folderpaths[1], coords)
        many_avg_2 = MultiFile_Reg_avg_reduced (Dates_Reduc[0],Dates_Reduc[1],folderpaths[0], coords)
    
        Dates_minor = np.array(many_avg_1[5], dtype='datetime64')
        Dates_major = np.array(many_avg_2[5], dtype='datetime64')
        Dates_major_end = np.array(many_avg_2[8], dtype='datetime64')

        avg_1 = many_avg_1[0].astype('f8')
        Nans_1 = (many_avg_1[1].astype('f8')/many_avg_1[6].astype('f8'))*100

        avg_2 = many_avg_2[0].astype('f8')
        Nans_2 = (many_avg_2[1].astype('f8')/many_avg_2[6].astype('f8'))*100

    h = np.round(coords[3] - coords[2],4)
    w = np.round(coords[1] - coords[0],4)
    c_lon = (coords[1] + coords[0])/2
    c_lat = (coords[3] + coords[2])/2
    h_km = np.round(111*np.abs(h),0)
    w_km = np.round(111*np.abs(w),0)
    
    axs[0].set_title('Data capture binning: {} vs {} \n Pos: [{}, {}] \n Area: {} x {} [km] \n'.format(many_avg_1[4][0], many_avg_2[4][0],c_lon,c_lat,h_km,w_km))
    
    axs[0].scatter(Dates_minor, avg_1, label=('{} data\n {} \n {}\n ({})'.format(many_avg_1[7][0], many_avg_1[5][0],
                                                          many_avg_1[5][-1], many_avg_1[4][0])), zorder=10,s=12,alpha=0.9)
#     adj = np.array([])
#     for i in range(len(Dates_major)-1):
#         adj = np.append(adj,(Dates_major[i+1]-Dates_major[i])/2,0)
    Cent = (Dates_major[1]-Dates_major[0])/2
    Centered_Dates_major = Dates_major+Cent
                        
    axs[0].scatter(Dates_major+Cent,avg_2, label = ('{} data\n {} \n {}\n ({})'.format(many_avg_2[7][0],many_avg_2[5][0],
                                                                                many_avg_2[5][-1],many_avg_2[4][0])),color='tab:orange',marker='s')
    axs[0].plot(Dates_major+Cent,avg_2 ,color='grey', zorder=0)
#     for i in range(len(Dates_major)+1):
#         axs[0].axvline(Dates_major[0] + i*Cent*2, linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
#     axs[0].set_xlim(Dates_minor[0], Dates_minor[-1])  
    axs[0].xaxis.set_major_locator(mdates.MonthLocator())
    axs[0].xaxis.set_minor_locator(mdates.DayLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("\n\n%b - %Y"))
    axs[0].xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
    
    axs[0].tick_params('x', length=4, width=1, which='both')
    axs[0].tick_params(axis='x', which="minor", rotation=90, labelsize=8)

    axs[0].set_ylabel('$\mathrm{K}_\mathrm{d}(490) [\mathrm{m}^{-1}]$')
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5),
      fancybox=False,frameon=True, ncol=1,fontsize=10)
    axs[0].text(0.7,1.2,'Total Var ({}) = {} '.format(many_avg_1[4][0],np.round(np.nanvar(avg_1)*10**3,5))+r'$\cdot~10^{-3}~[\mathrm{m}^{-1}]$'+'\nTotal Var ({}) = {} '.format(many_avg_2[4][0],np.round(np.nanvar(avg_2)*10**3,5))+r'$\cdot~10^{-3}~[\mathrm{m}^{-1}]$', ha='left', va='top', transform=axs[0].transAxes,fontsize=12, backgroundcolor='lightgrey')

    axs[1].set_title('\n Percentage of viable data in region of interest')
    axs[1].bar(Dates_minor,100 - Nans_1, label = ('{} \n {} \n {}\n ({})'.format(many_avg_1[7][0],many_avg_1[5][0],
                                                                                many_avg_1[5][-1],many_avg_1[4][0])),alpha=0.7)
    axs[1].bar(Dates_major+Cent,100 - Nans_2, label = ('{} \n {} \n {}\n ({})'.format(many_avg_2[7][0],many_avg_2[5][0],
                                                                                many_avg_2[5][-1],many_avg_2[4][0])),alpha=0.7)
   
    axs[1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[1].xaxis.set_minor_locator(mdates.DayLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("\n\n%b - %Y"))
    axs[1].xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
#     axs[1].set_xlim(Dates_minor[0], Dates_minor[-1])

#     for i in range(len(Dates_major)+1):
#         axs[0].axvline(Dates_major[0] + i*Cent*2, linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
#         axs[1].axvline(Dates_major[0] + i*Cent*2, linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
        
#     for i in range(len(Dates_major)):
#         axs[0].axvline(Dates_major[i], linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
#         axs[0].axvline(Dates_major_end[i], linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
#         axs[1].axvline(Dates_major[i], linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
#         axs[1].axvline(Dates_major_end[i], linestyle=(0,(4,4)), linewidth=1, alpha=0.3, color='grey')
    
    for i in range(len(Dates_major)-2):
        k = i+1
        up = np.argwhere(Dates_major_end[k] == Dates_minor)[0][0] 
        lw = np.argwhere(Dates_major[k] == Dates_minor)[0][0] 
        var = np.round(np.nanvar(avg_1[lw:up])*10**6,3)
        pRMS = np.round(np.nanstd(avg_1[lw:up])/np.nanmean(avg_1[lw:up]),3)
        axs[0].text(Dates_major[k]+ Cent/4, np.nanmax(avg_1)*1.1, 'Variance:\n{}'.format(var)+r'$~\cdot~10^{-6}$'+'\n\n% RMS:\n{}'.format(pRMS), fontsize=7)

    v1up = np.argwhere(Dates_major_end[0] == Dates_minor)[0][0]   
    var1 = np.round(np.nanvar(avg_1[:v1up])*10**6,3)
    pRMS1 = np.round(np.nanstd(avg_1[:v1up])/np.nanmean(avg_1[:v1up]),3)
    axs[0].text(Dates_major[0]+ Cent/4, np.nanmax(avg_1)*1.1, 'Variance:\n{}'.format(var1)+r'$~\cdot~10^{-6}$'+'\n\n% RMS:\n{}'.format(pRMS1), fontsize=7)
    
    v2lp = np.argwhere(Dates_major[-1] == Dates_minor)[0][0]  
    var2 = np.round(np.nanvar(avg_1[v2lp:])*10**6,3)
    pRMS2 = np.round(np.nanstd(avg_1[:v1up])/np.nanmean(avg_1[:v1up]),3)
    axs[0].text(Dates_major[-1]+ Cent/4, np.nanmax(avg_1)*1.1, 'Variance:\n{}'.format(var2)+r'$~\cdot~10^{-6}$'+'\n\n% RMS:\n{}'.format(pRMS2), fontsize=7)
    axs[0].set_ylim(np.nanmin(avg_1)*0.95, np.nanmax(avg_1)*1.25 )

    
    axs[1].tick_params('x', length=4, width=1, which='both')
    axs[1].tick_params(axis='x', which="minor", rotation=90, labelsize=8)
    
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),
      fancybox=False,frameon=True, ncol=1,fontsize=10)
    if len(Dates_minor) > 100 :
        axs[0].minorticks_off()
        axs[1].minorticks_off()
#     axs[0].set_xlim(Dates_major[0]-Cent,Dates_major[-1]+Cent)
#     axs[1].set_xlim(Dates_major[0]-Cent,Dates_major[-1]+Cent)
    

#     for key, value in kwargs.items():
#         if key =='var' :
#             for i in range(len(Dates_major)-2):
#                 k = i+1
#                 up = np.argwhere(Dates_major_end[k] == Dates_minor)[0][0] 
#                 lw = np.argwhere(Dates_major[k] == Dates_minor)[0][0] 
#                 var = np.round(np.nanvar(avg_1[lw:up])*10**6,3)
#                 axs[0].text(Dates_major[k]+ Cent/4, np.nanmax(avg_1)*1.12, 'Week Var\n{}'.format(var)+r'$~\cdot~10^{-6}$', fontsize=7)
#         v1up = np.argwhere(Dates_major_end[0] == Dates_minor)[0][0]   
#         var1 = np.round(np.nanvar(avg_1[:v1up])*10**6,3)
#         axs[0].text(Dates_major[0]+ Cent/4, np.nanmax(avg_1)*1.12, 'Week Var\n{}'.format(var1)+r'$~\cdot~10^{-6}$', fontsize=7)
#         v2lp = np.argwhere(Dates_major[-1] == Dates_minor)[0][0]  
#         var2 = np.round(np.nanvar(avg_1[v2lp:])*10**6,3)
#         axs[0].text(Dates_major[-1]+ Cent/4, np.nanmax(avg_1)*1.12, 'Week Var\n{}'.format(var2)+r'$~\cdot~10^{-6}$', fontsize=7)
#         axs[0].set_ylim(np.nanmin(avg_1)*0.95, np.nanmax(avg_1)*1.2 )
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value),dpi=600)
            
######################################################  
######################################################
######################################################

def KD490_Plot_Data_Region_ani(file, coords=[], **kwargs):
    # Read and store .nc file using netCDF4 Package function 'Dataset'
    Kd_data = Dataset(r"{}".format(file), format="NETCDF4")

    # Optional: Print Data Information
    # print(Kd_data)

    # Assuming standard NASA variable naming convention
    # Store variables as np.arrays for easy plotting
    Kd_490, lat, lon = np.array(Kd_data.variables['Kd_490']), np.array(Kd_data.variables['lat']), np.array(Kd_data.variables['lon'])

    # Overwrite int16 to NaN values
    Kd_490 = np.where(Kd_490 == -32767, np.nan, Kd_490)

    # Slicing Data for specified region
    Left, Right, Upper, Lower = coords

    lat_uslice = np.min([i for i in range(len(lat)) if lat[i] < Upper])
    lat_lslice = np.max([i for i in range(len(lat)) if lat[i] >= Lower])
    lon_wslice = np.min([i for i in range(len(lon)) if lon[i] >= Left])
    lon_eslice = np.max([i for i in range(len(lon)) if lon[i] < Right])

    Kd_490_reduced = Kd_490[lat_uslice:lat_lslice, lon_wslice:lon_eslice]
    lat_reduced = lat[lat_uslice:lat_lslice]
    lon_reduced = lon[lon_wslice:lon_eslice]

    t0 = Kd_data.time_coverage_start
    tf = Kd_data.time_coverage_end
    period = Kd_data.temporal_range
    DateStart = t0[:-14]
    Dateend = tf[:-14]
    inst = '{} - {}'.format(Kd_data.instrument, Kd_data.platform)

    # Plotting
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.27)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[0].stock_img()
    axs[1].stock_img()
    axs[2].stock_img()
    pad_lon1 = (np.max(lon_reduced) - np.min(lon_reduced)) * 8
    pad_lat1 = (np.max(lat_reduced) - np.min(lat_reduced)) * 8
    pad_lon2 = (np.max(lon_reduced) - np.min(lon_reduced)) * 0.2
    pad_lat2 = (np.max(lat_reduced) - np.min(lat_reduced)) * 0.2
    lat1, lon1 = np.meshgrid(lon, lat)
    
#     from numba import njit
#     NaNs = KD490_Region_Avg(file, coords)[1]
#     @njit
#     def get_first_index_nb(A):
#         for i in range(len( MultiFile_FileNames (A))):
#             if KD490_Region_Avg(MultiFile_FileNames (A)[i], coords)[1]/KD490_Region_Avg(MultiFile_FileNames (A)[i], coords)[5] < 10:
#                 return i
#         return -1
    
#     idx = get_first_index_nb(folder)
    
    if np.count_nonzero(~np.isnan(Kd_490_reduced)) == 0:
        cmap_adj = plt.colormaps.get_cmap("rainbow").copy()
        lev_exp = np.arange(np.log10(0.01), np.log10(5), 0.01)
        levs = np.power(10, lev_exp)
        cs1 = axs[0].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
        cs1.cmap.set_under('none')
        formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(5,1000))
        
        axins = axs[0].inset_axes([0,-0.2,2.2,0.05])
        
        cbar1 = fig.colorbar(cs1, cax=axins ,orientation='horizontal', ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)

        cmap_adj1 = plt.colormaps.get_cmap("inferno").copy()
        lat1_reduced, lon1_reduced = np.meshgrid(lon_reduced, lat_reduced)
        cs1 = axs[0].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='neither', norm='log')
        axs[0].set_extent([np.min(lon), np.max(lon), np.max(lat), np.min(lat)])
        cs1.cmap.set_under('none')

        cs2 = axs[1].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
        cs2 = axs[1].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='neither', norm='log')
        axs[1].set_extent([np.min(lon_reduced) - pad_lon1, np.max(lon_reduced) + pad_lon1, np.min(lat_reduced) - pad_lat1, np.max(lat_reduced) + pad_lat1])

        cs3 = axs[2].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='neither', norm='log')
        cs3 = axs[2].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
        axs[2].set_extent([np.min(lon_reduced) - pad_lon2, np.max(lon_reduced) + pad_lon2, np.min(lat_reduced) - pad_lat2, np.max(lat_reduced) + pad_lat2])

        h = np.max(lat_reduced) - np.min(lat_reduced)
        w = np.max(lon_reduced) - np.min(lon_reduced)

        ext_h = (np.max(lat_reduced) + pad_lat1) - (np.min(lat_reduced) - pad_lat1)
        ext_w = (np.max(lon_reduced) + pad_lon1) - (np.min(lon_reduced) - pad_lon1)
#         rect = patches.Rectangle((np.min(np.min(lon_reduced) - pad_lon1), np.min(np.min(lat_reduced) - pad_lat1)), ext_h, ext_w, linewidth=0.5, edgecolor='red', facecolor='none')
#         axs[0].add_patch(rect)
        rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=1, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)
        rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=2.5, edgecolor='black', facecolor='none')
        axs[2].add_patch(rect)
        cbar = fig.colorbar(cs3, fraction=0.046, pad=0.05, ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)





        Avg = KD490_Region_Avg(file, coords)[0]
        NaNs = KD490_Region_Avg(file, coords)[1]
        NanAvg = (NaNs / KD490_Region_Avg(file, coords)[5]) * 100
        axs[2].set_title(r'Avg $\mathrm{K_d}(490)$ = '+'{:.5f}\n Empty data points = {} ({:.1f}%)'.format(Avg, NaNs, NanAvg))
        cbar.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')
        cbar1.ax.set_xlabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')

        cs1.changed()
        cs2.changed()
        cs3.changed()


        for key, value in kwargs.items():
            if key == 'frame':
                fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n Frame: {}\n'.format(inst, DateStart, Dateend, period, coords[0], coords[1], coords[2], coords[3],value), size=16)
        for key, value in kwargs.items():
            if key == 'save':
                fig.savefig('{}'.format(value), dpi=300)
        plt.close()

        
    else:
        cmap_adj = plt.colormaps.get_cmap("rainbow").copy()
        lev_exp = np.arange(np.log10(0.01), np.log10(5), 0.01)
        levs = np.power(10, lev_exp)
        cs1 = axs[0].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj, extend='neither')
        cs1.cmap.set_under('none')
        formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(5,1000))
        
        axins = axs[0].inset_axes([0,-0.2,2.2,0.05])
        
        cbar1 = fig.colorbar(cs1, cax=axins ,orientation='horizontal', ticks=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], format=formatter)

        cmap_adj1 = plt.colormaps.get_cmap("inferno").copy()
        lat1_reduced, lon1_reduced = np.meshgrid(lon_reduced, lat_reduced)
        cs1 = axs[0].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
        axs[0].set_extent([np.min(lon), np.max(lon), np.max(lat), np.min(lat)])
        cs1.cmap.set_under('none')

        cs2 = axs[1].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
        cs2 = axs[1].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
        axs[1].set_extent([np.min(lon_reduced) - pad_lon1, np.max(lon_reduced) + pad_lon1, np.min(lat_reduced) - pad_lat1, np.max(lat_reduced) + pad_lat1])


        cs3 = axs[2].contourf(lat1, lon1, Kd_490, levs, norm=LogNorm(), cmap=cmap_adj)
        cs3 = axs[2].contourf(lat1_reduced, lon1_reduced, Kd_490_reduced, 700, cmap=cmap_adj1, extend='both', norm='log')
        axs[2].set_extent([np.min(lon_reduced) - pad_lon2, np.max(lon_reduced) + pad_lon2, np.min(lat_reduced) - pad_lat2, np.max(lat_reduced) + pad_lat2])

        h = np.max(lat_reduced) - np.min(lat_reduced)
        w = np.max(lon_reduced) - np.min(lon_reduced)

        ext_h = (np.max(lat_reduced) + pad_lat1) - (np.min(lat_reduced) - pad_lat1)
        ext_w = (np.max(lon_reduced) + pad_lon1) - (np.min(lon_reduced) - pad_lon1)
#         rect = patches.Rectangle((np.min(np.min(lon_reduced) - pad_lon1), np.min(np.min(lat_reduced) - pad_lat1)), ext_h, ext_w, linewidth=0.5, edgecolor='red', facecolor='none')
#         axs[0].add_patch(rect)
        rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=1, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)
        rect = patches.Rectangle((np.min(lon_reduced), np.min(lat_reduced)), w, h, linewidth=2.5, edgecolor='black', facecolor='none')
        axs[2].add_patch(rect)
        ticks_finer = np.logspace(np.log10(0.01),np.log10(5),40,endpoint=True)
        cbar = fig.colorbar(cs3, fraction=0.046, pad=0.05, ticks=ticks_finer, format=formatter)





        Avg = KD490_Region_Avg(file, coords)[0]
        NaNs = KD490_Region_Avg(file, coords)[1]
        NanAvg = (NaNs / KD490_Region_Avg(file, coords)[5]) * 100
        axs[2].set_title(r'Avg $\mathrm{K_d}(490)$ = '+'{:.5f}\n Empty data points = {} ({:.1f}%)'.format(Avg, NaNs, NanAvg))
        cbar.ax.set_ylabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')
        cbar1.ax.set_xlabel('$\mathrm{K_d}(490) ~ [\mathrm{m}^{-1}]$')

        cs1.changed()
        cs2.changed()
        cs3.changed()

        for key, value in kwargs.items():
            if key == 'frame':
                fig.suptitle('{} data\n From {} to {} ({})\n Lon:[{} - {}] Lat:[{} - {}]\n Frame: {}\n'.format(inst, DateStart, Dateend, period, coords[0], coords[1], coords[2], coords[3],value), size=16)
        for key, value in kwargs.items():
            if key == 'save':
                fig.savefig('{}'.format(value), dpi=300)
                
        plt.close()

def Animate_Kd490_Region(Folder,location,name,coords=[],**kwargs):
    st = time.time()
    for key, value in kwargs.items():
        if key == 'range':
            for i in range(value):
                idx = -(i+1)
                File = MultiFile_FileNames (Folder)[idx]
                KD490_Plot_Data_Region_ani(File, coords, save='{}\{}_({}).png'.format(location,name,i),frame='{}'.format(i+1))
                t = value
                p = np.round(i/t * 100,1)
                print('Progress: {}%'.format(p), end="\r")
            with imageio.get_writer('{}\{}.gif'.format(location,name), mode='I',duration=700,disposal=2, loop=0) as writer:
                for i in range(value):
                    image = imageio.imread('{}\{}_({}).png'.format(location,name,i))
                    writer.append_data(image)
                    os.remove('{}\{}_({}).png'.format(location,name,i))
            et = time.time()
            tt = np.round(et - st,0)
            print('Completed: File saved as {}.gif! (time elapsed: {}s)'.format(name,tt))
            writer.close()
    if len(kwargs.items()) == 0:
        for i in range(len(MultiFile_FileNames (Folder))):
            idx = -(i+1)
            File = MultiFile_FileNames (Folder)[idx]
            KD490_Plot_Data_Region_ani(File, coords, save='{}\{}_({}).png'.format(location,name,i),frame='{}'.format(i+1))
            t = len(MultiFile_FileNames (Folder))
            p = np.round(i/t * 100,1)
            print('Progress: {}% ({}/{})'.format(p,i,t), end="\r")
        with imageio.get_writer('{}\{}.gif'.format(location,name), mode='I',duration=700,disposal=2, loop=0) as writer:
            for i in range(len(MultiFile_FileNames (Folder))):
                image = imageio.imread('{}\{}_({}).png'.format(location,name,i))
                writer.append_data(image)
                os.remove('{}\{}_({}).png'.format(location,name,i))
        et = time.time()
        tt = np.round(et - st,0)
        print('Completed: File saved as {}.gif! (time elapsed: {}s)'.format(name,tt))
        writer.close()


def Animate_Kd490_Region_mp4(Folder,location,name,coords=[],**kwargs):
    st = time.time()
    for key, value in kwargs.items():
        if key == 'range':
            for i in range(value):
                idx = -(i+1)
                File = MultiFile_FileNames (Folder)[idx]
                KD490_Plot_Data_Region_ani(File, coords, save='{}\{}_({}).png'.format(location,name,i),frame='{}'.format(i+1))
                t = value
                p = np.round(i/t * 100,1)
                print('Progress: {}%'.format(p), end="\r")
            with imageio.get_writer('{}\{}.mp4'.format(location,name), mode='I', fps=2, macro_block_size = 1) as writer:
                for i in range(value):
                    image = imageio.imread('{}\{}_({}).png'.format(location,name,i))
                    writer.append_data(image)
                    os.remove('{}\{}_({}).png'.format(location,name,i))
            et = time.time()
            tt = np.round(et - st,0)
            print('Completed: File saved as {}.mp4! (time elapsed: {}s)'.format(name,tt))
            writer.close()
    if len(kwargs.items()) == 0:
        for i in range(len(MultiFile_FileNames (Folder))):
            idx = -(i+1)
            File = MultiFile_FileNames (Folder)[idx]
            KD490_Plot_Data_Region_ani(File, coords, save='{}\{}_({}).png'.format(location,name,i),frame='{}'.format(i+1))
            t = len(MultiFile_FileNames (Folder))
            p = np.round(i/t * 100,1)
            print('Progress: {}% ({}/{})'.format(p,i,t), end="\r")
        with imageio.get_writer('{}\{}.mp4'.format(location,name), mode='I', fps=2, macro_block_size = 1) as writer:
            for i in range(len(MultiFile_FileNames (Folder))):
                image = imageio.imread('{}\{}_({}).png'.format(location,name,i))
                writer.append_data(image)
                os.remove('{}\{}_({}).png'.format(location,name,i))
        et = time.time()
        tt = np.round(et - st,0)
        print('Completed: File saved as {}.mp4! (time elapsed: {}s)'.format(name,tt))
        writer.close()
        
        
def MultiPos_MultiROI_DataVar_Plot(SizeList,PosList, folderpath, date_start,date_end,**kwargs):
    CoordsList = []
    for i in range(len(SizeList)):
        List = CoordsCalcList(SizeList[i], PosList)
        CoordsList.insert(i,List)
    fig, ax = plt.subplots(1, figsize=(14,10),layout="constrained")
    for i in range(len(SizeList)):
#         Var = MultiPos_DataVar(CoordsList[i], folderpath)
#         std = np.sqrt(Var)
        avg = np.array([])
        Var = np.array([])
        SubList = CoordsList[i]
        for j in range(len(SubList)):
            many_avg = MultiFile_Reg_avg_adj (folderpath, date_start,date_end, SubList[j])
            calcAVG = np.nanmean(many_avg[0].astype('float64'))
            calcVar = np.nanvar(many_avg[0].astype('float64'))
            avg = np.append(avg,calcAVG)
            Var = np.append(Var,calcVar)
        std = np.sqrt(Var)
        Pos_String = []
        for k in range (len(PosList)):
            Pos_String = np.append(Pos_String,'{}'.format(PosList[k]))
        s = int(np.round(4+(np.emath.logn(2,SizeList[i])),0))
        ax.errorbar(Pos_String , avg, yerr = std, marker='s', capsize=s, markersize = s, label='{} km ROI'.format(SizeList[i]), linestyle='',elinewidth=3,alpha=0.7)  
        # ax.scatter(Pos_String ,avg,label='{}'.format(SizeList[i]))  
    window =  np.datetime64(many_avg[5][-1]) - np.datetime64(many_avg[5][0])
    fig.suptitle('Comparison of position and ROI size (Including standard deviation)\n{} to {} ({})\nSatellite data captured period ({}) [Files: {}]\n{}\n'.format(many_avg[5][0], many_avg[5][-1],window,many_avg[4][0],many_avg[8][0],many_avg[7][0] ),fontsize=22)
    ax.set_xlabel('Position Coordinates [Lon, Lat]',fontsize=22)
    ax.set_ylabel(r'Kd(490)[m$^{-1}$]',fontsize=22)
    
    handles, labels = ax.get_legend_handles_labels()

    new_handles = []

    for h in handles:
        #only need to edit the errorbar legend entries
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)

    ax.legend(new_handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=10)
    for key, value in kwargs.items():
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))
        else:
            print('FIGURE NOT SAVED! Check kwarg is correct')

            
            

def MultiFile_Reg_avg_adj (folderpath,date_start,date_end, coords=[]):

    mypath = folderpath
    Folder1_StartDates = np.datetime64()
    Folder1_EndDates = np.datetime64()
    for i in range(len(MultiFile_FileNames(folderpath))):
        mypath = folderpath
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        Kd_data = Dataset(r"{}".format(loc), format="NETCDF4")
        Folder1_StartDates = np.append(Folder1_StartDates,np.datetime64(Kd_data.time_coverage_start[:10]).astype('datetime64[M]'))
        Folder1_EndDates = np.append(Folder1_EndDates,np.datetime64(Kd_data.time_coverage_end[:10]).astype('datetime64[M]'))
            
    index = np.where(np.logical_and(np.datetime64(date_start,'M')<=Folder1_StartDates[:], np.datetime64(date_end, 'M')>=Folder1_EndDates[:]))
    a = index[0][0] -1 
    b = index[0][-1]
    
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
    files = files[a:b]
    x = np.array([])
    y = np.array([])
    t0 = np.array([])
    tf = np.array([])
    period = np.array([])
    Date = np.array([])
    DataNum = np.array([])
    inst = np.array([])
    for i in range(len(files)):
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        x = np.append(x,KD490_Region_Avg(r"{}".format(loc),coords)[0]) ## Avg Value
        y = np.append(y,KD490_Region_Avg(r"{}".format(loc),coords)[1]) ## No. of NaN
        t0 = np.append(t0,KD490_Region_Avg(r"{}".format(loc),coords)[2]) ## Time Begun
        tf = np.append(tf,KD490_Region_Avg(r"{}".format(loc),coords)[3]) ## Time Ended
        period = np.append(period,KD490_Region_Avg(r"{}".format(loc),coords)[4]) ## Data Capture length
        Date = np.append(Date,KD490_Region_Avg(r"{}".format(loc),coords)[2][:-14]) ## Date of capture start
        DataNum = np.append(DataNum,KD490_Region_Avg(r"{}".format(loc),coords)[5]) ## Number of Data Points in region
        inst = np.append(inst,KD490_Region_Avg(r"{}".format(loc),coords)[6]) ## instrument
    numfiles = np.array([])
    for l in range(len(x)):
        numfiles= np.append(numfiles,np.round((len(files)),0))
    return np.array([x,y,t0,tf,period,Date,DataNum,inst,numfiles])

        
def MultiPos_MonthlyVar_Plot(Size,PosList, folderpath, date_start,date_end,**kwargs):

    CoordsList = CoordsCalcList(Size, PosList)
    fig, ax = plt.subplots(1, figsize=(16,10),layout="constrained")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b - %Y"))
    RealEnd = np.array(date_end, dtype='datetime64[M]')
    Months = np.arange(date_start, RealEnd+1 ,1, dtype='datetime64[M]')
    for i in range(len(CoordsList)):
        avg = np.array([])
        Var = np.array([])
        Num = np.array([])
        Date = np.array([])
        for j in range(len(Months)):
            many_avg = MultiFile_Reg_avg_monthly(folderpath, Months[j],Months[j], CoordsList[i])
            calcAVG = np.nanmean(many_avg[0].astype('float64'))
            calcVar = np.nanvar(many_avg[0].astype('float64'))
            avg = np.append(avg,calcAVG)
            Var = np.append(Var,calcVar)
            Num = np.append(Num,many_avg[8][0].astype('float64'))
            Date = np.append(Date,many_avg[5][0])
        std = np.sqrt(Var)
        NumT = np.sum(Num)
        Mon = np.array(Date,dtype='datetime64[M]')
        ax.errorbar(Mon , avg, yerr = std, marker='s', capsize=20, markersize = 10, label='{}'.format(PosList[i]), linestyle='-',elinewidth=3,alpha=0.6, markeredgewidth=2)  
    end = np.array(Months[-1]+1,dtype='datetime64[D]')
    start = np.array(Months[0],dtype='datetime64[D]')
    window =  end-start
    fig.suptitle('Monthly Average (Including standard deviation) at Multiple Position\n{} to {} ({})\nSatellite data captured period ({}) [Files: {}]\n{}\n'.format(Months[0],  Months[-1],window,many_avg[4][0],NumT,many_avg[7][0] ),fontsize=22)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.xaxis.set_tick_params(labelsize=24, rotation=90)

    ax.set_ylabel(r'Kd(490)[m$^{-1}$]',fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    new_handles = []

    for h in handles:
        if isinstance(h, container.ErrorbarContainer):
            new_handles.append(h[0])
        else:
            new_handles.append(h)

    ax.legend(new_handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=False,frameon=True, ncol=1,fontsize=18, labelspacing=2, title='ROI Size:\n{}X{} km'.format(Size,Size), title_fontsize=18)
    for key, value in kwargs.items():
        if key == 'Map'  and value == True :
            Coords_Check_funcAdju( CoordsList)
        if key == 'save' :
            print('Figure Saved Successfully')
            fig.savefig('{}'.format(value))

            
            
def MultiFile_Reg_avg_monthly (folderpath,date_start,date_end, coords=[]):

    mypath = folderpath
    Folder1_StartDates = np.datetime64()
    Folder1_EndDates = np.datetime64()
    for i in range(len(MultiFile_FileNames(folderpath))):
        mypath = folderpath
        files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        Kd_data = Dataset(r"{}".format(loc), format="NETCDF4")
        Folder1_StartDates = np.append(Folder1_StartDates,np.datetime64(Kd_data.time_coverage_start[:10]).astype('datetime64[M]'))
        Folder1_EndDates = np.append(Folder1_EndDates,np.datetime64(Kd_data.time_coverage_end[:10]).astype('datetime64[M]'))
            
    index = np.where(np.logical_and(np.datetime64(date_start,'M')<=Folder1_StartDates[:], np.datetime64(date_end, 'M')>=Folder1_EndDates[:]))
    a = index[0][0] -1 
    b = index[0][-1]
    
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".nc")]
    files = files[a:b]
    x = np.array([])
    y = np.array([])
    t0 = np.array([])
    tf = np.array([])
    period = np.array([])
    Date = np.array([])
    DataNum = np.array([])
    inst = np.array([])
    for i in range(len(files)):
        f = files[i]
        loc = '{}\{}'.format(mypath, f)
        x = np.append(x,KD490_Region_Avg(r"{}".format(loc),coords)[0]) ## Avg Value
        y = np.append(y,KD490_Region_Avg(r"{}".format(loc),coords)[1]) ## No. of NaN
        t0 = np.append(t0,KD490_Region_Avg(r"{}".format(loc),coords)[2]) ## Time Begun
        tf = np.append(tf,KD490_Region_Avg(r"{}".format(loc),coords)[3]) ## Time Ended
        period = np.append(period,KD490_Region_Avg(r"{}".format(loc),coords)[4]) ## Data Capture length
        Date = np.append(Date,KD490_Region_Avg(r"{}".format(loc),coords)[2][:-14]) ## Date of capture start
        DataNum = np.append(DataNum,KD490_Region_Avg(r"{}".format(loc),coords)[5]) ## Number of Data Points in region
        inst = np.append(inst,KD490_Region_Avg(r"{}".format(loc),coords)[6]) ## instrument
    numfiles = np.array([])
    for l in range(len(x)):
        numfiles= np.append(numfiles,np.round((len(files)),0))
    return np.array([x,y,t0,tf,period,Date,DataNum,inst,numfiles])



def Coords_Check_funcAdju(coords=[]):
    
    ### Plotting ###
    fig,axs= plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,7),layout="constrained")
    fig.suptitle('Plotting Given Coordinates',fontsize=25)
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].stock_img()
    axs[1].stock_img()
    
    lat_mins = []
    lat_maxs = []
    lon_mins = []
    lon_maxs = []

    for i in range(len(coords)):
        lat_mins = np.append(lat_mins,coords[i][2])
        lat_maxs = np.append(lat_maxs,coords[i][3])
        lon_mins = np.append(lon_mins,coords[i][0])
        lon_maxs = np.append(lon_maxs,coords[i][1])
        
    pad_lon_left = np.min(lon_mins)
    pad_lon_right = np.max(lon_maxs)
    pad_lat_upper = np.max(lat_maxs)
    pad_lat_lower = np.min(lat_mins)
    
    axs[0].set_extent([110,165,-5,-45])
    
    axs[1].set_extent([pad_lon_left*0.95,pad_lon_right*1.05,pad_lat_upper*0.9,pad_lat_lower*1.1])
    
    for i in range(len(coords)):
        h = np.round(coords[i][3] - coords[i][2],4)
        w = np.round(coords[i][1] - coords[i][0],4)
        c_lat = (coords[i][1] + coords[i][0])/2
        c_lon = (coords[i][3] + coords[i][2])/2

        axs[0].scatter(c_lat,c_lon,s=80,marker='x',color='r')
        axs[1].annotate('[{}, {}]'.format(c_lat,c_lon),(c_lat+2,c_lon-0.3),fontsize=10)
        
        rect = patches.Rectangle((coords[i][0],coords[i][2]), w, h, linewidth=2, edgecolor='black', facecolor='none')
        axs[1].add_patch(rect)    
        axs[1].scatter(c_lat,c_lon,s=20,marker='x',color='r')


    
