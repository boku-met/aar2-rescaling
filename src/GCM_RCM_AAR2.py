#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:56:26 2021

@author: fabianl
"""

import xarray as xr
import numpy as np
import regionmask
import geopandas as gp
import matplotlib.pyplot as plt
import glob
import pandas as pd

''' emission scenarios are completely ignored'''
### compare temperature and precipitation for all ÖKS 15 models an CMIP 6 GCMs

#############   enter your data here   ############################
years_for_running_mean=20
area='Austria'
path_out =  '/nas8/Fabian/FORSITE_II/Szenarienauswahl/Abbildungen/'
endname = '' # 



# path for monthly ÖKS15 and STARC impact data 
# path_gcm = '/metstor_nfs/projects/BIOCLIM/GCMs/'
path_gcm = '/metstor_nfs/projects/VTreasures/Austria_AllModels/'
###################################################################

    
if area=='Austria':
    latmin=46.5
    latmax=49
    lonmin=9.5
    lonmax=17
elif area=='Tirol': 
    latmin=46.6
    latmax=48.9
    lonmin=10
    lonmax=12.8
elif area=='FORSITE2': 
    latmin=47
    latmax=49
    lonmin=13
    lonmax=17
    
#%% tas 
parameter='tas' # ## 'tas' or 'pr'
unit='°C'
months='' #   , '' [10,11,12,1,2,3]   [4,5,6,7,8,9]  ,[6,7,8]#share of summer
# allfiles=sorted(glob.glob(path_gcm+'Tas_Mon/*'+parameter+'*ssp*.nc'))
allfiles=sorted(glob.glob(path_gcm+parameter+'_*ssp*.nc'))
# allfiles=sorted(glob.glob(path_gcm+'*/Monthly/'+parameter+'_*ssp126*.nc'))

data_tas_preind = pd.DataFrame()
data_tas = pd.DataFrame()
data_tas_global = pd.DataFrame()
for SSP in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
    allfiles_ssp = [line for line in allfiles if SSP in line]

    
    for file in allfiles_ssp: 
        file_list = [*glob.glob(file.rsplit('ssp')[0]+'*historical*'), *glob.glob(file.rsplit('ssp')[0]+'ssp'+file.rsplit('ssp')[1][0:3]+'*')]
        print(file_list)
        DS=xr.open_mfdataset(file_list).load()
   
        weights = np.cos(np.deg2rad(DS.lat))
        weights.name = "weights"
        var_weighted = DS[parameter].weighted(weights)
        series_xr = DS[parameter].sel(lat=slice(latmin,latmax), lon=slice(lonmin,lonmax)).mean(dim=('lat', 'lon'))
        series_xr_global = var_weighted.mean(dim=('lat', 'lon'))
        # if len(months)>0:
        #     series_xr = series_xr[series_xr.time.dt.month.isin(months)]       

        series_xr = series_xr.groupby('time.year').mean().load()
        series_xr_global = series_xr_global.groupby('time.year').mean().load()
        series = pd.Series(series_xr.values, index=series_xr.year.values, name=file.split('tas_Amon_')[1].rsplit('_', 1)[0])
        series_global = pd.Series(series_xr_global.values, index=series_xr_global.year.values, name=file.split('tas_Amon_')[1].rsplit('_', 1)[0])
        
        # subtract reference period  
        series_1875 = series - series.loc[slice(1851,1900)].mean()
        series = series - series.loc[slice(1991,2020)].mean()        
        series_global = series_global - series_global.loc[slice(1851,1900)].mean()
    
        # add to dataframe
        data_tas_preind= data_tas_preind.join(series_1875, how='outer')
        data_tas_preind = data_tas_preind.sort_index()
        
        data_tas= data_tas.join(series, how='outer')
        data_tas = data_tas.sort_index()
        
        data_tas_global= data_tas_global.join(series_global, how='outer')
        data_tas_global = data_tas_global.sort_index()
        
    # smoothing 
data_tas_preind= data_tas_preind.rolling(years_for_running_mean,min_periods=1, win_type='hamming', center=True).mean()   
data_tas= data_tas.rolling(years_for_running_mean,min_periods=1, win_type='hamming', center=True).mean() 
data_tas_global= data_tas_global.rolling(int(years_for_running_mean/2),min_periods=1, win_type='hamming', center=True).mean() 
                     


      

#%%   
data_tas_global_2085 = data_tas_global.loc[slice(2081,2100)].mean()
data_tas_2085 = data_tas.loc[slice(2081,2100)].mean()
data_tas_2085_preind = data_tas_preind.loc[slice(2081,2100)].mean()

data_tas_2085_preind = data_tas_preind.loc[slice(2081,2100)].mean()


          
#%% 
# select 1.5°C global goal 
temp_mask_15 = ((data_tas_global_2085<=1.5))
tas15_50 = 0.8


# select <2°C global goal 
temp_mask_2 = ((data_tas_global_2085<2) & (data_tas_global_2085>1.5))
tas2_50 = (data_tas_2085)[temp_mask_2].quantile(0.5)

#% select 3°C goal (which is 2.5-3.5°C)
temp_mask_3 = ((data_tas_global_2085<3.5) & (data_tas_global_2085>2.5))
tas3_50 = (data_tas_2085)[temp_mask_3].quantile(0.5)

# select 4°C goal (which is 3.5-4.5°C)
temp_mask_4 = ((data_tas_global_2085<4.5) & (data_tas_global_2085>3.5))
tas4_50 = (data_tas_2085)[temp_mask_4].quantile(0.5)

# select 4°C goal (which is 4.5-5.5°C)
temp_mask_5 = ((data_tas_global_2085<5.5) & (data_tas_global_2085>4.5))
tas5_50 = (data_tas_2085)[temp_mask_5].quantile(0.5)





#%%   Plots
plt.close('all')
count=0
colors=['tab:blue', 'tab:orange', 'tab:red', 'maroon']
plot_labels=['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
fig, ax = plt.subplots(figsize=(1.5*6/2, 1.36*6/2))
for SSP in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
    chosen = data_tas_global_2085.index.str.find(SSP)>0
    plt.scatter(data_tas_global_2085[chosen], data_tas_2085[chosen], s=80, c=colors[count], label=plot_labels[count], alpha=0.7)    
    count=count+1  
    
# plt.scatter((data_tas_global_2085)[temp_mask_15].quantile(0.5), 0.8, c='k', marker='x', s=150)
# plt.text((data_tas_global_2085)[temp_mask_15].quantile(0.5)+0.15, 0.8-0.1, '<1.5°C')
# plt.scatter((data_tas_global_2085)[temp_mask_2].quantile(0.5), (data_tas_2085)[temp_mask_2].quantile(0.5), c='k', marker='x', s=150)
# plt.text((data_tas_global_2085)[temp_mask_2].quantile(0.5)+0.15, (data_tas_2085)[temp_mask_2].quantile(0.5)-0.15, '<2°C')
# plt.scatter((data_tas_global_2085)[temp_mask_3].quantile(0.5), (data_tas_2085)[temp_mask_3].quantile(0.5), c='k', marker='x', s=150)
# plt.text((data_tas_global_2085)[temp_mask_3].quantile(0.5)+0.15, (data_tas_2085)[temp_mask_3].quantile(0.5)-0.15, '3°C')
# plt.scatter((data_tas_global_2085)[temp_mask_4].quantile(0.5), (data_tas_2085)[temp_mask_4].quantile(0.5), c='k', marker='x', s=150, label='°C worlds (median)')
# plt.text((data_tas_global_2085)[temp_mask_4].quantile(0.5)+0.15, (data_tas_2085)[temp_mask_4].quantile(0.5)-0.15, '4°C')
    
plt.plot(np.linspace(0,8), np.linspace(0,8), c='Gray', linewidth=0.5)
plt.xlim(0,8)
plt.ylim(0,8)
plt.xlabel('Worldwide (2081-2100 \N{MINUS SIGN} 1851-1900)')  
plt.ylabel('Austria (2081-2100 \N{MINUS SIGN} 1991-2020)')  
plt.title('CMIP6: temperature change (°C)', size='medium')    
plt.legend(loc='best')
plt.grid(alpha=0.7)
plt.tight_layout() 
plt.savefig('/nas8/Fabian/Skripte/plots/tas_tas_global_vs_austria_scatterplot.png' , dpi=300)

#%% ÖKS 15 tas
# path for monthly ÖKS15 and STARC impact data 
parameter = 'tas'
path_rcm = '/hp2/OKS15/Monthly-Data-All-Models/'

allfiles=sorted(glob.glob(path_rcm+parameter+'*rcp*.nc'))

data_OKS15_tas = pd.DataFrame()

for file in allfiles: 
    if 'SDM_CNRM-CERFACS-CNRM-CM5_rcp26_r1i1p1_CNRM-ALADIN53' in file: 
        print('Skip '+file)
        continue        
    print(file)
    DS=xr.open_dataset(file)
    DA_cut = DS[parameter].where((DS.lat>latmin) & (DS.lat<latmax) & (DS.lon>lonmin) & (DS.lon<lonmax), drop=True)
    series_xr = DA_cut.sel(time=slice('1981','2100')).mean(dim=('y', 'x'))
    if parameter=='tas':
        series_xr = series_xr.groupby('time.year').mean().load()
    elif parameter=='pr':
        series_xr = (series_xr.groupby('time.year').sum()).load()
    series = pd.Series(series_xr.values, index=series_xr.year.values, name=file.split('SDM_')[1].rsplit('_', 1)[0])
    # subtract reference period       
    series = series - series.loc[slice(1991,2020)].mean() 
    series = series.rolling(years_for_running_mean,min_periods=1, win_type='hamming', center=True).mean()
    # add to dataframe
    data_OKS15_tas = data_OKS15_tas.join(series, how='outer')
    data_OKS15_tas = data_OKS15_tas.sort_index()
    


#%% 1.5, 2,3,4 degree years
year15 = data_OKS15_tas[data_OKS15_tas>tas15_50].idxmin()
year2 = data_OKS15_tas[data_OKS15_tas>tas2_50].idxmin()
year3 = data_OKS15_tas[data_OKS15_tas>tas3_50].idxmin()
year4 = data_OKS15_tas[data_OKS15_tas>tas4_50].idxmin()
# dictionary_year = {'<1.5°C':year15, '<2°C':year2, '3°C':year3, '4°C':year4}
df_year = pd.concat([year15,year2, year3, year4], axis=1, keys=['<1.5°C','<2°C','3°C','4°C'])

#%% plot line scatterplot 
# FORSITE
plt.close('all')

count=0
colors=['tab:blue', 'tab:orange', 'tab:red']
plot_labels=['RCP2.6 ('+str((df_year.index.str.find('rcp26')>0).sum())+')', 
             'RCP4.5 ('+str((df_year.index.str.find('rcp45')>0).sum())+')',
             'RCP8.5 ('+str((df_year.index.str.find('rcp45')>0).sum())+')']
fig, ax = plt.subplots(figsize=(1.5*6/2, 1.36*6/2))
for RCP in ['rcp26', 'rcp45', 'rcp85']:
    for i, column in enumerate(df_year.columns):
        chosen = df_year.index.str.find(RCP)>0
        if i==len(df_year.columns)-1:
            plt.scatter(np.ones(sum(chosen))*i, df_year[chosen][column], s=400/(count*0.5+1), linewidths=2+count*0.2, c=colors[count], label=plot_labels[count], marker='_', alpha=1)  
        else:                
            plt.scatter(np.ones(sum(chosen))*i, df_year[chosen][column], s=400/(count*0.5+1), linewidths=2+count*0.2, c=colors[count], marker='_',alpha=1)    
    count=count+1
    
    
plt.ylabel('year') 
plt.xticks(np.arange(4), ['<1.5°C','<2°C','3°C','4°C']) 
plt.title(area+', ÖKS15: year of reaching a certain global warming', size='small')    
plt.legend(loc='best')
plt.grid(alpha=0.7)
plt.tight_layout() 
plt.savefig('/nas8/Fabian/Skripte/plots/Years_of_reaching_temperature.png',dpi=300) 

#%% ÖKS 15 kysely
# path for monthly ÖKS15 and STARC impact data 
path_rcm = '/metstor_nfs/home/bennib/Bennib/HotSpotKlim/HotSpotKlim_Salzburg/Data/Indicators/'
allfiles=sorted(glob.glob(path_rcm+'kysely_periods_*rcp*.nc'))
parameter = 'kysely_periods_noofdays'

data_OKS15_kysely = pd.DataFrame()

for file in allfiles: 
    if 'SDM_CNRM-CERFACS-CNRM-CM5_rcp26_r1i1p1_CNRM-ALADIN53' in file: 
        print('Skip '+file)
        continue        
    print(file)
    DS=xr.open_dataset(file)
    DA_cut = DS[parameter].sel(x=626500, y=482500)
    series_xr = DA_cut.sel(time=slice('1981','2100'))

    series_xr = series_xr.groupby('time.year').mean().load()

    series = pd.Series(series_xr.values, index=series_xr.year.values, name=file.split('SDM_')[1].rsplit('_', 1)[0].rsplit('_all', 1)[0])
    # add to dataframe
    data_OKS15_kysely = data_OKS15_kysely.join(series, how='outer')
    data_OKS15_kysely = data_OKS15_kysely.sort_index()


#%% index 
df = pd.DataFrame(data=None, index=np.arange(21), columns=data_OKS15_tas.columns)
dictionary_indicator = {'<1.5°C':df.copy(), '<2°C':df.copy(), '3°C':df.copy(), '4°C':df.copy()}

for degree in dictionary_indicator: 
    print(degree)
    for model in data_OKS15_tas.columns:
        if ~np.isnan(df_year[degree].loc[model]):
            year_begin = max(int(df_year[degree].loc[model])-10,1981)
            year_end = min(int(df_year[degree].loc[model])+10,2100)
            dictionary_indicator[degree][model][0:year_end-year_begin+1]=data_OKS15_kysely[model].loc[year_begin:year_end].values
            print(np.mean(data_OKS15_kysely[model].loc[year_begin:year_end].values))
    

    
indicator_flat = pd.concat([pd.Series(dictionary_indicator['<1.5°C'].values.flatten(), dtype='f4'),
                            pd.Series(dictionary_indicator['<2°C'].values.flatten(), dtype='f4'), 
                            pd.Series(dictionary_indicator['3°C'].values.flatten(), dtype='f4'), 
                            pd.Series(dictionary_indicator['4°C'].values.flatten(), dtype='f4')], axis=1, 
                           keys=['<1.5°C','<2°C','3°C','4°C'])
                      
#%% plots for degree goals
fig= plt.subplots(figsize=(5, 4))
indicator_flat.boxplot()
plt.ylabel('Kysely days')
plt.title('Vienna')
plt.tight_layout()
plt.savefig('/nas8/Fabian/Skripte/plots/Kysely_days.png',dpi=300) 

#%% plots for RCP
years= 2050, 2090

for year in years:
    df_rcp = pd.DataFrame(data=None, index=np.arange(1,13*20+1), columns=['rcp26', 'rcp45', 'rcp85'], dtype='f4')
    
    for rcp in df_rcp.columns: 
        print(rcp)
        dummy = data_OKS15_kysely.loc[year-9:year+10]
        dummy = dummy.iloc[:,dummy.columns.str.find(rcp)>1].values.flatten()
        df_rcp[rcp].iloc[0:len(dummy)] =dummy
    
    fig= plt.subplots(figsize=(5, 4))  
    df_rcp.boxplot()
    plt.ylabel('Kysely days')
    plt.title('ÖKS15: Vienna ('+str(year-9)+' - '+str(year+10)+')')
    plt.xticks(np.arange(1,4), ['RCP2.6', 'RCP4.5', 'RCP8.5']) 
    plt.tight_layout()
    plt.savefig('/nas8/Fabian/Skripte/plots/Kysely_days_rcp_'+str(year-9)+'-'+str(year+10)+'.png',dpi=300) 
    
    
#%%% SECURES
path = '/metstor_nfs/projects/Secrues/Final_Data/Temperature/NUTS0_Europe/'
dataframe=pd.DataFrame(dtype='f4', columns=['ICHEC_rcp45', 'ICHEC_rcp85'], index=np.arange(1951,2101))
for model in ['ICHEC_rcp45', 'ICHEC_rcp85']:
    print(model)
    for period in ['1951-2000', '2001-2050', '2051-2100']:
        print(period)
        df = pd.read_csv(path+model+'/T2M_NUTS0_Europe_mean_'+model.replace('_','-')+'_hourly_'+period+'.csv', index_col=0, parse_dates=True)
        df_current = df.AT
        df_current.name = model
        df_current_annual = df_current.resample('Y').mean()
        df_current_annual.index= df_current_annual.index.year
        dataframe.loc[df_current_annual.index, model]= df_current_annual
dataframe = dataframe -dataframe.loc[slice(1991,2020)].mean()  
dataframe = dataframe.rolling(int(years_for_running_mean/2),min_periods=1, win_type='hamming', center=True).mean() 
year15 = dataframe[dataframe>0.8].idxmin()
year2 = dataframe[dataframe>1.2].idxmin()
year3 = dataframe[dataframe>2.6].idxmin()
year4 = dataframe[dataframe>4.0].idxmin()
