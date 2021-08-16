
import os
import matplotlib.pyplot as plt

from PlanktonImageAnalysis  import *

suffix='.pklz'

dest_dir='/home/dg/ZoopOA/Python4Plankton/Test2'

base_dir='/home/dg/ZoopOA/Images/AssociatedFullFrames/CompletedArchives/'

source_dirs=['18Sept2018_CompletedArchives/5143_18Sept2018_Completed Archives',
             '18Sept2018_CompletedArchives/6055_18Sept2018_Completed Archives',
             '18Sept2018_CompletedArchives/6162_18Sept2018_Completed Archives',
             '17Sept2018_CompletedArchives/4032_17Sept2018_Completed Archives',
             '17Sept2018_CompletedArchives/4043_17Sept2018_Completed Archives',
             '17Sept2018_CompletedArchives/5041_17Sept2018_Completed Archives',
             '17Sept2018_CompletedArchives/5062_17Sept2018_Completed Archives',
             '17Sept2018_CompletedArchives/6071_17Sept2018_Completed Archives']


for s_d in source_dirs:
    dir_path=os.path.join(base_dir,s_d)
    archive_list=[s for s in os.listdir(dir_path) if s[-5:]=='.pklz']
    for a_l in archive_list:
        a_path=os.path.join(dir_path,a_l)
        print('Exporting archive: ',a_path)
        a=load_snapshot(a_path)
        # Check if a spurious plotting window was opened
        last_window_name=plt.gcf().canvas.manager.get_window_title()
        print('last_window_name = ',last_window_name)
        if last_window_name[:6]=='Figure':
            # close spurious platting window
            plt.close(plt.gcf())

        a.export_all_frames(dest_dir,verbose=True,plotting=True,delay=0)
