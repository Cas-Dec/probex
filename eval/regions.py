
# define regions, as in ECMWF scorecard (https://sites.ecmwf.int/ifs/scorecards/scorecards-49r1HRES.html)
Global = {'lat': slice(-90, 90), 'lon': slice(-180, 180)}
NHem_extratropics = {'lat': slice(20, 90), 'lon': slice(-180, 180)}
SHem_extratropics = {'lat': slice(-90, -20), 'lon': slice(-180, 180)}
Tropics = {'lat': slice(-20, 20), 'lon': slice(-180, 180)}
Europe = {'lat': slice(35, 75), 'lon': slice(-12.5, 42.5)}
Arctic = {'lat': slice(60, 90), 'lon': slice(-180, 180)}
Antarctic = {'lat': slice(-90, -60), 'lon': slice(-180, 180)}
North_America = {'lat': slice(25, 60), 'lon': slice(-120, -75)}
East_Asia = {'lat': slice(25, 60), 'lon': slice(102.5, 150)}
no_Antarctic = {'lat': slice(-60, 90), 'lon': slice(-180, 180)} # this is the region used in the paper

regions = [Global, NHem_extratropics, SHem_extratropics, Tropics, Europe, 
           Arctic, Antarctic, North_America, East_Asia, 
           no_Antarctic]
regions_names = ['Global', 'NHem Extratropics', 'SHem Extratropics', 'Tropics', 
                 'Europe', 'Arctic', 'Antarctic', 'North America', 'East Asia', 
                 'no_Antarctic']


