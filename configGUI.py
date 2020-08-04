#
#  A configuration file for the GUI interface for plankton image analysis
#
#============================================================================
# Collage figure parameters
nrows=6 # Number of ROI rows and columns
ncols=8
figsize = [12, 8]     # ROI collage figure size, inches
figpos=[50,500]       # ROI collage figure position (x,y)
padsize=[90,90]   # padded size (w,h) of images, in pixels
edge=5            # width of a color indicator of category
bg_color=(16,16,16)
fill_color=(128,128,128)

plt_pad=0.05

# Classification window parameters:
#     1: implemented
#     0: not implemented
#     2: default
#
classifications=[[1,'Centric_Diatom','cendt','blue'],
                 [1,'Barnacle_nauplii','barnp','red'],
                 [1,'Oithona','oith','green'],
                 [1,'Calanoid_copepod','calcpd','yellow'],
                 [1,'Euphausiid/Mysid','euphmys', 'purple'],
		 [1,'Filament_Filaments','flmts','darkorange'],
                 [1,'Fish_larvae','fshlv','cadetblue'],
                 [1,'Krill/Decapod_larva','krdclv','darkviolet'],
                 [1,'Bivalve','bivl','deeppink'],
                 [1,'Limacina_pteropod','limptr','wheat'],
		 [1,'Gastropod','gast','olivedrab'],
                 [1,'Larvacean','lrvcn','darkkhaki'],
                 [1,'Nauplius','naupl','darkcyan'],
                 [1,'Amphipod','amph','burlywood'],
                 [1,'Chaetognath','chaet','goldenrod'],
		 [1,'Clione','cliptr','chartreuse'], 
                 [1,'Ctenophora','cten','crimson'], 
                 [1,'Ostracod','ostr','darkorchid'],
                 [1,'MarineSnow','marsn','deepskyblue'], 
                 [2,'Unknown','unknw','white'],
		 [1,'Crab_zoea','crbzo','seagreen'], 
                 [1,'Polychaete','polcht','maroon'], 
                 [1,'Siphonophore','siph','teal'], 
                 [1,'Tentacle','tent','coral'], 
                 [1,'Medusa-bell','medbl','bisque'],
		 [1,'Medusa-flat','medft','orangered'],
                 [1,'Blobs','blob','gold'],
                 [1,'Noctiluca','noct','gray'],
                 [1,'Error','error','lightgray']]
# construct classification lists
labels=[]
codes=[]
colors=[]
for icl,cl in enumerate(classifications):
    if cl[0]>0:
        labels.append(cl[1])
        codes.append(cl[2])
        colors.append(cl[3])
    if cl[0]==2:
        default_category=icl

# Labels window parameters                                              %
lbl_figsize = [1.5, 7.85]     # Labels figure size, inches
lbl_alpha=1.
lbl_figpos=[1500,500]       # Labels figure position (x,y)
lbl_x0 = 0
lbl_y0 = 784                # Tabs position within Labels figure
lbl_fontsize=12

#============================================================================
# Controls window parameters                                              %
ctrl_figsize = [1.5, 2.75]     # Controls figure size, inches
ctrl_alpha=1.
ctrl_figpos=[1200,500]       # Controls figure position (x,y)
ctrl_x0 = 0
#ctrl_y0 = 265                # Tabs position within Controls figure
ctrl_y0 = 272                # Tabs position within Controls figure
ctrl_fontsize=12

ctrl_labels= ('Next frame/ROI set','Previous frame/ROI set',
              'Define group','Dissolve group',
              'Randomize ROIsets','Reorder ROIsets',
              'Save snapshot','Load snapshot',
              'Archive all frames','Archive current frame')
ctrl_colors= ['lightgray','lightgray','gray','gray','darkgray','darkgray','gray',
		'gray','gainsboro','gainsboro']

