import fiona
import os
import numpy as np
import shapely
import shapely.geometry
import casadi as ca
import matplotlib.pyplot as plt

def polygontotxt():
    # script_dir = os.path.dirname(__file__)
    
    # read obstacle file
    # obstacles=[]
    obstacle_file = os.path.join('/home/acsr/Documents/extended_polygon.shp')
    file_path = os.path.join('/home/acsr/Documents/polygon.txt')
    
    f = open(file_path, "w")
    with fiona.open(obstacle_file) as shapefile:
        for record in shapefile:
            geometry = shapely.geometry.shape(record['geometry'])
            x5,y5 = geometry.exterior.xy
            for x,y in zip(x5,y5):
                f.write('{:.2f} {:.2f} '.format(x,y))
            f.write('\n')
            # obstacles.append(np.vstack([x5,y5]).transpose())                
    f.close()       

if __name__ == "__main__":

    polygontotxt()
