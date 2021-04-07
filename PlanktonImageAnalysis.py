#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# A module constructing a collage of plankton images for classification using a graphical user interface
# Requires Python 3.6 or later.
# Danny Grunbaum, PublicSensors.org and University of Washington 20200420-20200706
import os
import pickle
import gzip
import copy
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import csv
import time
import random
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.mathtext as mathtext
import matplotlib.artist as mpl_artist
import matplotlib.image as mpl_image
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

# Specify the backend for matplotlib -- default does not allow interactive mode
from matplotlib import use
use('TkAgg')   # this needs to happen before pyplot is loaded
import matplotlib.pyplot as plt
plt.ion()  # set interactive mode
from matplotlib.backend_bases import MouseButton
from matplotlib import get_backend

from configGUI import *
#from config23 import *

from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
#root = Tk()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

import imageio

global load_dir
load_dir=os.getcwd()

global selector

#============================================================================
version = 'PIA_20210407'
print('PlanktonImageAnalysis version',version)
#============================================================================
# Default thresholds for segmenting blobs from a image or binary image
#minThreshold = 5
minThreshold = 20    # Can/should be reset during creation of Analysis or Frame objects
#minThreshold = 50    # Can/should be reset during creation of Analysis or Frame objects
#minThreshold = 10
maxThreshold = 255   # Should genereally be 255
minArea=10           # Obsolete, currently used only in depreciated "simple" blob segmentation

#============================================================================
# Lasso selection infrastructure from the matplotlib demo, lasso_selector_demo_sgskip.py
class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

# Menu bar infrastructure for choosing classes, modified from an example on the matplotlib site;
# Uses new version of menu construction in revised menu.py from matplotlib gallery, downloaded 20210407
class ItemProperties:
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha


class MenuItem(mpl_artist.Artist):
    padx = 5
    pady = 5

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None,number=0):
        super().__init__()

        self.set_figure(fig)
        self.labelstr = labelstr

        self.props = props if props is not None else ItemProperties()
        self.hoverprops = (
            hoverprops if hoverprops is not None else ItemProperties())
        if self.props.fontsize != self.hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')

        self.on_select = on_select

        # Setting the transform to IdentityTransform() lets us specify
        # coordinates directly in pixels.
        self.label = fig.text(0, 0, labelstr, transform=IdentityTransform(),
                              size=props.fontsize)
        self.text_bbox = self.label.get_window_extent(
            fig.canvas.get_renderer())

        self.rect = mpl_patches.Rectangle((0, 0), 1, 1)  # Will be updated later.

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, _ = self.rect.contains(event)
        if not over:
            return
        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h, depth):
        self.rect.set(x=x, y=y, width=w, height=h)
        self.label.set(position=(x + self.padx, y + depth + self.pady/2))
        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        props = self.hoverprops if b else self.props
        self.label.set(color=props.labelcolor)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        """
        Update the hover status of event and return whether it was changed.
        """
        b, _ = self.rect.contains(event)
        changed = (b != self.hover)
        if changed:
            self.set_hover_props(b)
        self.hover = b
        return changed


class Menu:
    def __init__(self, fig, x0, y0, menuitems):
        self.figure = fig

        self.menuitems = menuitems

        maxw = max(item.text_bbox.width for item in menuitems)
        maxh = max(item.text_bbox.height for item in menuitems)
        depth = max(-item.text_bbox.y0 for item in menuitems)

        totalh = self.numitems*maxh + (self.numitems + 1)*2*MenuItem.pady

        #x0 = 100
        #y0 = 400

        width = maxw + 2*MenuItem.padx
        height = maxh + MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height, depth)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        if any(item.set_hover(event) for item in self.menuitems):
            self.figure.canvas.draw()


'''
# Disable depreciated version of memu construction infrastructure;
# to be removed after confirmation new code is working.

class ItemProperties(object):
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha

        self.labelcolor_rgb = mpl_colors.to_rgba(labelcolor)[:3]
        self.bgcolor_rgb = mpl_colors.to_rgba(bgcolor)[:3]


class MenuItem(mpl_artist.Artist):
    parser = mathtext.MathTextParser("Bitmap")
    padx = 10
    pady = 5

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None,number=0):
        mpl_artist.Artist.__init__(self)

        self.set_figure(fig)
        self.labelstr = labelstr

        if props is None:
            props = ItemProperties()

        if hoverprops is None:
            hoverprops = ItemProperties()

        self.props = props
        self.hoverprops = hoverprops

        self.on_select = on_select

        self.number=number

        x, self.depth = self.parser.to_mask(
            labelstr, fontsize=props.fontsize, dpi=fig.dpi)

        if props.fontsize != hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')

        self.labelwidth = x.shape[1]
        self.labelheight = x.shape[0]

        self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
        self.labelArray[:, :, -1] = x/255.

        self.label = mpl_image.FigureImage(fig, origin='upper')
        self.label.set_array(self.labelArray)

        # we'll update these later
        self.rect = mpl_patches.Rectangle((0, 0), 1, 1)

        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        over, junk = self.rect.contains(event)
        if not over:
            return

        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h):
        #print(x, y, w, h)
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w)
        self.rect.set_height(h)

        self.label.ox = x + self.padx
        self.label.oy = y - self.depth + self.pady/2.

        self.hover = False

    def draw(self, renderer):
        self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        r, g, b = props.labelcolor_rgb
        self.labelArray[:, :, 0] = r
        self.labelArray[:, :, 1] = g
        self.labelArray[:, :, 2] = b
        self.label.set_array(self.labelArray)
        self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b, junk = self.rect.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)

        self.hover = b
        return changed


class Menu(object):
    def __init__(self, fig, x0, y0,menuitems):
        self.figure = fig
        fig.suppressComposite = True

        self.menuitems = menuitems
        self.numitems = len(menuitems)

        maxw = max(item.labelwidth for item in menuitems)
        maxh = max(item.labelheight for item in menuitems)

        totalh = self.numitems*maxh + (self.numitems + 1)*2*MenuItem.pady

        width = maxw + 2*MenuItem.padx
        height = maxh + MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height)

            fig.artists.append(item)
            y0 -= maxh + MenuItem.pady

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        draw = False
        for item in self.menuitems:
            draw = item.set_hover(event)
            if draw:
                self.figure.canvas.draw()
                break
'''
#============================================================================
# A function to move windows on the desktop, following
# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

#============================================================================
# Create the classification GUI
lbl_fig = plt.figure(figsize=lbl_figsize,facecolor='k')
lbl_fig.subplots_adjust(left=0.01)
lbl_fig.canvas.set_window_title('Category Selection')
move_figure(lbl_fig,lbl_figpos[0],lbl_figpos[1])

menuitems = []
for ilabel in range(len(labels)):
    label=labels[ilabel]
    props = ItemProperties(labelcolor='black', bgcolor=colors[ilabel],fontsize=lbl_fontsize, alpha=lbl_alpha)
    hoverprops = ItemProperties(labelcolor='white', bgcolor=colors[ilabel],
                                fontsize=lbl_fontsize, alpha=lbl_alpha)
    def on_select(item):
        global cat_number, cat_color, cat_label
        print('you selected %d, %s' % (item.number, item.labelstr))
        cat_number=item.number
        cat_color=colors[item.number]
        cat_label=item.labelstr
    item = MenuItem(lbl_fig, label, props=props, hoverprops=hoverprops,
                    on_select=on_select, number=ilabel)
    menuitems.append(item)

menu = Menu(lbl_fig, lbl_x0, lbl_y0, menuitems)


#============================================================================
# Define a class to facilitate assignment and handling of ROI image classification

class Frame():
    """A class to contain and analyze full images ("frames") from ZooCAM profiles
    """
    global selector
    def __init__(self,frame_dir=None,frame_file=None,frame_image=None,ROIlist=[],ROIgroup=[],counter=None,display=False):
        self.frame_dir=frame_dir
        self.frame_file=frame_file
        if frame_image is not None:
            self.frame_image=frame_image
        elif self.frame_file is not None:
            try:
                self.read_frame()
            except:
                self.frame_image=None
                #print('ERROR: Failed to load frame_ file %s' % self.frame_file)
        if display:
            try:
                self.show_frame()
            except:
                print('ERROR: Failed to show frame_ file %s' % self.frame_file)
        self.counter=counter
        self.ROIlist=ROIlist
        self.ROIgroup=ROIgroup
        print('creating Frame, len(self.ROIlist)=',len(self.ROIlist))


    def read_frame(self,frame_dir=None,frame_file=None):
        print('reading frame...')
        if frame_dir is not None:
            self.frame_dir=frame_dir
        if frame_file is not None:
            self.frame_file=frame_file
        frame_path=os.path.join(self.frame_dir,self.frame_file)
        print('path is %s' % frame_path)
        try:
            self.frame_image=cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            self.color_frame_image=cv2.imread(frame_path)#,cv2.COLOR_BAYER_RG2RGB)
            self.color_frame_image=imageio.imread(frame_path)
            #self.color_frame_image=cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR)
            print('Loaded frame from path %s' % frame_path)
        except:
            self.frame_image=None
            print('ERROR: Failed to load frame from path %s' % frame_path)

    def show_frame(self,fig_num=100):
        try:
            plt.figure(fig_num,facecolor=tuple([i/255 for i in bg_color]))
            plt.imshow(self.frame_image, cmap='gray', interpolation='bicubic')
            plt.tight_layout(pad=plt_pad)
        except:
            print('ERROR: Failed to show frame image...')
        
    def binary_frame(self, min_val=minThreshold, max_val=maxThreshold,display=False,fill_holes=True,fig_num=101):
    #def binary_frame(self, min_val=thr_min_val, max_val=thr_max_val,display=False,fill_holes=True,fig_num=101):
        self.binary_image=cv2.threshold(self.frame_image, min_val, max_val, cv2.THRESH_BINARY)[1]
        #print('before:',self.binary_image)
        if fill_holes: # Fill holes within thresholded blobs
            # after example at https://www.programcreek.com/python/example/89425/cv2.floodFill
            frame_floodfill=self.binary_image.copy()
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = self.binary_image.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            
            # Floodfill from point (0, 0)
            cv2.floodFill(frame_floodfill, mask, (0,0), 255);
            
            # Invert floodfilled image
            frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)
            
            # Combine the two images to get the foreground.
            self.binary_image = self.binary_image.astype(np.uint8) | frame_floodfill_inv.astype(np.uint8)
            #print('after:',self.binary_image)
            self.binary_fig_num=fig_num
            
        if display:
            self.show_binary_frame()
            
    def show_binary_frame(self):
        plt.figure(self.binary_fig_num,facecolor=tuple([i/255 for i in bg_color]))
        plt.imshow(self.binary_image, cmap='gray')
        plt.tight_layout(pad=plt_pad)
        title_str='Figure '+str(self.binary_fig_num)+ \
            ',   Filename: '+self.frame_file
        plt.gcf().canvas.set_window_title(title_str)

    def show_blobs_frame(self):
        print('plotting contours...')
        cnt_fig=plt.figure(self.blobs_fig_num,facecolor=tuple([i/255 for i in bg_color]))
        cnt_fig.clf()
        title_str='Figure '+str(self.blobs_fig_num)+ \
            ',   Filename: '+self.frame_file
        plt.gcf().canvas.set_window_title(title_str)
        if self.use_binary:
            plt.imshow(self.binary_image, cmap='gray')
        else:
            plt.imshow(self.frame_image, cmap='gray')
        for ctr in self.contours:
            polygon = Polygon(np.squeeze(ctr, axis=1),True,linewidth=1,edgecolor='m',facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(polygon)
            bbox=cv2.boundingRect(ctr)
            rect = Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],
                             linewidth=1,edgecolor='c',facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(rect)
        plt.tight_layout(pad=plt_pad)
        cnt_fig.canvas.draw()
            
    def show_ROIs_frame(self,color_classes=True):
        roi_fig=plt.figure(self.ROI_fig_num,facecolor=tuple([i/255 for i in bg_color]))
        roi_fig.clf()
        title_str='Figure '+str(self.ROI_fig_num)+ \
            ',   Filename: '+self.frame_file
        plt.gcf().canvas.set_window_title(title_str)
        if self.use_binary:
            plt.imshow(self.binary_image, cmap='gray')
        else:
            plt.imshow(self.color_frame_image, cmap='gray')
            #plt.imshow(self.frame_image, cmap='gray')
        edge_color='r'
        for roi in self.ROIlist:
            if color_classes:
                edge_color=roi.fill_color
            rect = Rectangle((roi.j_beg,roi.i_beg),roi.j_end-roi.j_beg,roi.i_end-roi.i_beg,
                             linewidth=1,edgecolor=edge_color,facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(rect)
        plt.tight_layout(pad=plt_pad)
        cfm = plt.get_current_fig_manager()
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        roi_fig.canvas.draw()

    def show_all_frames(self):
        self.show_binary_frame()
        self.show_blobs_frame()
        self.show_ROIs_frame()

    def select_group(self):
        # bring ROI figure to the front
        plt.figure(self.ROI_fig_num)
        cfm = plt.get_current_fig_manager()
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        # Add reference points to all ROIs
        ref_pts=np.zeros([len(self.ROIlist),2])
        for i,roi in enumerate(self.ROIlist):
            ref_pts[i,0]=roi.j_beg
            ref_pts[i,1]=roi.i_beg
        pts=plt.scatter(ref_pts[:, 0], ref_pts[:, 1], color='r',s=20)

        print("Entering group selection: select ROIs with lasso, then <cr> to accept, c or q to cancel...")
        selector = SelectFromCollection(plt.gca(),pts)
        
        def accept(event):
            if event.key == "enter":
                print("New group formed with ROIs:")
                print('selector.ind=',selector.ind)
                # Check for empty group selections:
                if len(selector.ind)==0:
                    print('Empty grouping selected; skipping group definition...')
                else:
                    print('selector.xys[selector.ind]=',selector.xys[selector.ind])
                    new_group_num=self.create_group(selector.ind)
                    self.plot_group(new_group_num)
                selector.ind=[]
                selector.xys=[]
                selector.disconnect()
                plt.disconnect(self.binding_id)
                self.show_ROIs_frame()
            elif event.key == "c" or event.key == "q":
                print('cancelled...')
                selector.ind=[]
                selector.xys=[]
                selector.disconnect()
                plt.disconnect(self.binding_id)
                self.show_ROIs_frame()

        self.binding_id=plt.gcf().canvas.mpl_connect("key_press_event", accept)
        
    def create_group(self,grp_ind,reset_current=True,reset_previous=True):
        # Create a group out of the indices in the list grp_ind,
        # and any ROIs these are already grouped with.
        # "reset_" flags determine whether respective categories are reset.
        # Default behavior is to reset both, to avoid divergence in categories
        # within a group (e.g. with "undo").
        grps = [*{*[self.ROIgroup[ir] for ir in grp_ind]}] # unique list of represented groups
        print('grps=',grps)
        extended_grp_ind=[]
        for ir,roi in enumerate(self.ROIlist): # set all ROIs with specified group numbers 
            if self.ROIgroup[ir] in grps:                     # to lowest group number 
                self.ROIgroup[ir]=grps[0]
                roi.group=grps[0]
                if reset_current:
                    roi.category=default_category
                if reset_previous:
                    roi.prev_category=default_category
                roi.fill_color=colors[roi.category]
                roi.label=labels[roi.category]
                roi.code=codes[roi.category]
        print('Forming new group #',grps[0], ' with members ',self.ROIgroup)
        return grps[0]

    def plot_group(self,igrp,grp_fig_num=106):
        # Plot ROIs in specified group to a new image window
        self.grp_fig_num=grp_fig_num
        grp_fig=plt.figure(self.grp_fig_num,facecolor=tuple([i/255 for i in bg_color]))
        grp_fig.clf()
        title_str='Group #'+str(igrp)+'   Figure '+str(self.grp_fig_num)+ \
            ',   Filename: '+self.frame_file
        plt.gcf().canvas.set_window_title(title_str)
        if self.use_binary:
            plt.imshow(self.binary_image, cmap='gray')
        else:
            plt.imshow(self.frame_image, cmap='gray')
        for ir,roi in enumerate(self.ROIlist):
            if roi.group == igrp:    # indicate selected group with a cyan ROI box
            #if self.ROIgroup[ir] == igrp:    # indicate selected group with a cyan ROI box
                colr='c'
            else:
                colr='r'
            rect = Rectangle((roi.j_beg,roi.i_beg),roi.j_end-roi.j_beg,roi.i_end-roi.i_beg,
                                 linewidth=1,edgecolor=colr,facecolor='none')
            # Add the patch to the Axes
            plt.gca().add_patch(rect)
        plt.tight_layout(pad=plt_pad)
        cfm = plt.get_current_fig_manager()
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        grp_fig.canvas.draw()
                   
    def get_group_members(self,igrp):
        # Return a list of all members of the indicated group
        print('Getting members of group ',igrp)
        group_members=[]
        for ir,roi in enumerate(self.ROIlist): # set all ROIs with specified group numbers
            #print('ir,roi.group = ',ir,roi.group)
            if roi.group == igrp:
                group_members.append(ir)
        return group_members
                
    def classify_group(self,igrp,next_category,reset_previous=False):
        # Set new classification of all ROIs in group igrp.
        # If reset_previous is True, the previous category is also reset.
        # This should be done whenever a new group is formed, so that members
        # of groups cannot diverge with "undo" clicks..
        print('Classifying group ',igrp)
        for ir,roi in enumerate(self.ROIlist): # set all ROIs with specified group numbers
            if roi.group == igrp:                     # to lowest group number 
                print('ir,roi.group = ',ir,roi.group)
                print('prev.,current, new category = ',roi.prev_category,roi.category,next_category)
                roi.prev_category=roi.category  # shift current category into previous
                roi.category=next_category      # replace current category with submitted next category
                roi.fill_color=colors[roi.category]
                roi.label=labels[roi.category]
                roi.code=codes[roi.category]
    
    def dissolve_group(self,igrp):
        print('dissolving group ',igrp)
        for ir,roi in enumerate(self.ROIlist): # dissolve group, by setting all ROI group
            if self.ROIgroup[ir] == igrp:      # numbers back to ROOI index
                self.ROIgroup[ir]=ir
                roi.group=ir
                roi.category=default_category
                roi.prev_category=default_category
                roi.fill_color=colors[roi.category]
                roi.label=labels[roi.category]
                roi.code=codes[roi.category]
        self.plot_group(igrp)
        self.show_ROIs_frame()
        
    def segment_frame(self, method='contour',min_val=minThreshold, max_val=maxThreshold,min_area=minArea,
                      display_ROIs=False,display_blobs=False,display_blobsCV=False,use_binary=True,
                      fig_numROI=102,fig_numBLOB=103,fig_numCTR=104,category=None,ROIpad=15):

        if method == 'simple':
            self.segment_frameSIMPLE(min_val=min_val, max_val=max_val,min_area=min_area,
                      display_ROIs=display_ROIs,display_blobs=display_blobs,display_blobsCV=display_blobsCV,
                      use_binary=use_binary,fig_numROI=fig_numROI,fig_numBLOB=fig_numBLOB,category=category)
        elif method == 'contour':
            self.segment_frameCONTOUR(min_val=min_val, max_val=max_val,min_area=min_area,
                      display_ROIs=display_ROIs,display_blobs=display_blobs,display_blobsCV=display_blobsCV,
                      use_binary=use_binary,fig_numROI=fig_numROI,fig_numBLOB=fig_numBLOB,category=category,
                      fig_numCTR=fig_numCTR,ROIpad=ROIpad)
        else:
            print("Unknown method, '%s', for Frame segmentation; valid choices are 'simple' and 'contour'")
            return

        self.ROIgroup=[]
        for i in range(len(self.ROIlist)):  # Parse ROIs into groups; initially each group contains only
            self.ROIgroup.append(i)         # a single ROI, to be aggregated subsequently e.g. using lasso
            self.ROIlist[i].group=i         # Revised group infrastructure with group residing in the ROI object
        
    def segment_frameCONTOUR(self, min_val=minThreshold, max_val=maxThreshold,min_area=minArea,
                      display_ROIs=False,display_blobs=False,display_blobsCV=False,use_binary=True,
                             fig_numROI=102,fig_numBLOB=103,fig_numCTR=104,category=None,ROIpad=5):

        print('starting segment_frameCONTOUR: len(self.ROIlist)=',len(self.ROIlist))
        self.ROIlist=[]
        self.blob_keypoints = []

        self.blobs_fig_num = fig_numBLOB
        self.ROI_fig_num = fig_numROI
        self.use_binary=use_binary

        self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if display_blobs:
            self.show_blobs_frame()
            
        # dimensions of image in pixels
        nx=len(self.frame_image)
        ny=len(self.frame_image[0])

        # Parse ROIs from contours
        for ctr in self.contours:
            #print(ctr[:,0,0])
            area=cv2.contourArea(ctr)
            bbox=cv2.boundingRect(ctr)
            try:
                ellbox = cv2.fitEllipse(ctr)
                ell = Ellipse((ellbox[0][0],ellbox[0][1]),ellbox[1][0],ellbox[1][1],angle=ellbox[2],
                              linewidth=1,edgecolor='y',facecolor='none')
                if display_ROIs:
                    plt.gca().add_patch(ell)
            except:
                ellbox=None
            i_beg=np.max([np.min(ctr[:,0,1])-ROIpad,0])
            i_end=np.min([np.max(ctr[:,0,1])+ROIpad,nx-1])
            j_beg=np.max([np.min(ctr[:,0,0])-ROIpad,0])
            j_end=np.min([np.max(ctr[:,0,0])+ROIpad,ny-1])
            
            # get blob subimage
            blob_img = Image.fromarray( self.color_frame_image[i_beg:i_end, j_beg:j_end])
            #blob_img = Image.fromarray( cv2.cvtColor(self.color_frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2RGB))
            #blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2RGB))
            #blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2GRAY))
            
            self.ROIlist.append(ROI(ROIimage=blob_img,edge=np.squeeze(ctr,axis=1),
                                    area=area,bbox=bbox,ellbox=ellbox,
                                    i_beg=i_beg,i_end=i_end,j_beg=j_beg,j_end=j_end,
                                    category=category))    
        if display_ROIs:
            self.show_ROIs_frame()

    def segment_frameSIMPLE(self, min_val=minThreshold, max_val=maxThreshold,min_area=minArea,
                      display_ROIs=False,display_blobs=False,display_blobsCV=False,use_binary=True,
                      fig_numROI=102,fig_numBLOB=103,category=None):

        print('starting segment_frameSIMPLE: len(self.ROIlist)=',len(self.ROIlist))
        self.ROIlist=[]
        
        # set up blob detection parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByCircularity = False
        params.filterByColor = False
        params.minThreshold = min_val
        params.maxThreshold = max_val
        params.minArea = min_area  # only detect blobs that have at least 10 pixels
        detector = cv2.SimpleBlobDetector_create(params)

        # grab blobs from binary image
        # note: a blob is represented as a center point and a radius;
        #       an ROI is the minimal rectangular that encloses a blob
        # list of keypoints in image corresponding to blobs
        if use_binary:
            print('segmenting using binary image...')
            self.blob_keypoints = detector.detect(self.binary_image)
        else:
            print('segmenting using original image...')
            self.blob_keypoints = detector.detect(self.frame_image)
        print('found %d blobs: ' % len(self.blob_keypoints))
        #print(dir(self.blob_keypoints[0]))

        # dimensions of image in pixels
        nx=len(self.frame_image)
        ny=len(self.frame_image[0])
        # Parse ROIs from blobs
        for kp in self.blob_keypoints:
            #print(kp.pt[0],kp.pt[1],kp.size)
            kp_i = int(kp.pt[1])
            kp_j = int(kp.pt[0])
            kp_sz = int(kp.size)        # kp.size is the diameter of the widest part of the blob

            # get boundaries on blob subimage
            if (kp_i - kp_sz) > 0:
                i_beg = kp_i - kp_sz
            else:
                i_beg = 0

            if kp_j - kp_sz > 0:
                j_beg = kp_j - kp_sz
            else:
                j_beg = 0

            if kp_i + kp_sz < len(self.frame_image):
                i_end = kp_i + kp_sz
            else:
                i_end = len(img)

            if kp_j + kp_sz < len(self.frame_image[0]):
                j_end = kp_j + kp_sz
            else:
                j_end = len(self.frame_image[0])

            # get blob subimage
            blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2RGB))
            #blob_img = Image.fromarray( cv2.cvtColor(self.frame_image[i_beg:i_end, j_beg:j_end], cv2.COLOR_BGR2GRAY))
            
            self.ROIlist.append(ROI(ROIimage=blob_img,keypoints=kp,i_beg=i_beg,i_end=i_end,j_beg=j_beg,j_end=j_end,
                                    category=category))    
            # write subimage to new file
            #cv2.imwrite(output_name, blob_img)
            
        if display_ROIs:
            roi_fig=plt.figure(fig_numROI,facecolor=tuple([i/255 for i in bg_color]))
            roi_fig.clf()
            if use_binary:
                plt.imshow(self.binary_image, cmap='gray')
            else:
                plt.imshow(self.frame_image, cmap='gray')
            for roi in self.ROIlist:
                rect = Rectangle((roi.j_beg,roi.i_beg),roi.j_end-roi.j_beg,roi.i_end-roi.i_beg,
                                         linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                plt.gca().add_patch(rect)
            plt.tight_layout(pad=plt_pad)
            
        if display_blobs:
            blob_fig=plt.figure(fig_numBLOB,facecolor=tuple([i/255 for i in bg_color]))
            blob_fig.clf()
            if use_binary:
                plt.imshow(self.binary_image, cmap='gray')
            else:
                plt.imshow(self.frame_image, cmap='gray')
            for roi in self.ROIlist:
                circ = Circle((roi.keypoints.pt[0],roi.keypoints.pt[1]),roi.keypoints.size,
                              linewidth=1,edgecolor='r',facecolor='none')
                # Add the patch to the Axes
                plt.gca().add_patch(circ)
            plt.tight_layout(pad=plt_pad)
            
        # Plotting segmented circles is proving problematic;
        # This flag turns it on and off for comparison with the plotting in pyplot above
        # Note that kp.size--> radius (as above) corresponds more closely with the ROIs
        if display_blobsCV:
            if True: #use_binary:
                frame_blobs=cv2.drawKeypoints(self.binary_image, self.blob_keypoints, np.array([]),
                                              (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else: # This does not display, for reasons unclear to me...
                frame_blobs=cv2.drawKeypoints(self.frame_image, self.blob_keypoints, np.array([]),
                                              (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('blobs',frame_blobs)
            #print('frame_blobs.shape = ',frame_blobs.shape)
            cv2.waitKey(1) 
        print('exiting segment_frameSIMPLE: len(self.ROIlist)=',len(self.ROIlist))

            
class ROI():
    """A class to contain and analyze ROIs extracted from ZooCAM frames
    """
    def __init__(self,ROIfile=None,ROIimage=None,counter=None,category=None,label=None,code=None,bg_color=None,fill_color=None,
                 keypoints=None,i_beg=None,i_end=None,j_beg=None,j_end=None,edge=None,area=None,bbox=None,ellbox=None):
        self.ROIfile=ROIfile
        if ROIimage is not None:
            self.ROIimage=ROIimage
        elif self.ROIfile is not None:
            try:
                self.read_ROI()
            except:
                self.ROIimage=None
                print('ERROR: Failed to load ROI file %s' % self.ROIfile)
        self.counter=counter
        if category is not None:      # Initialize category, label, color, code
            self.category=category
        else:
            self.category=default_category
        self.prev_category=category
        if label is not None:
            self.label=label
        else:
            self.label=labels[self.category]
        if code is not None:
            self.code=code
        else:
            self.code=codes[self.category]
        if fill_color is not None:
            self.fill_color=fill_color
        else:
            self.fill_color=colors[self.category]
        self.bg_color=bg_color
        self.image_with_border=None
        self.image_with_edge=None
        self.keypoints=keypoints
        self.edge=edge
        self.area=area
        self.bbox=bbox
        self.ellbox=ellbox
        self.i_beg=i_beg
        self.i_end=i_end
        self.j_beg=j_beg
        self.j_end=j_end
        self.group=None
        
    def add_border(self,padsize=[90,90],bg_color=(16,16,16)):
        # calculate the border sizes to make the padded image
        ow,oh=self.ROIimage.size
        #print('self.ROIimage.size=',self.ROIimage.size)
        nw=padsize[0]
        nh=padsize[1]
        delta_w = nw-ow
        delta_h = nh-oh
        ltrb_border=(delta_w//2,delta_h//2,delta_w-(delta_w//2),delta_h-(delta_h//2))
        # create the padded image
        self.image_with_border=ImageOps.expand(self.ROIimage,border=ltrb_border,fill=bg_color)

    def add_edge(self,edge=5):#,category=None,reset_previous=None):
        self.edge=edge
        self.image_with_edge=ImageOps.expand(self.image_with_border,border=self.edge,fill=self.fill_color)
            
    def show_image(self,axi):
        axi.imshow(self.image_with_edge)
        axi.axis('off')   # turn off axes rendering

    def read_ROI(self,ROIfile=None):
        if ROIfile is not None:
            self.ROIfile=ROIfile
        if self.ROIfile is not None:
            try:
                self.ROIimage=Image.open(self.ROIfile)
            except:
                self.ROIimage=None
                print('ERROR: Failed to load ROI file %s' % self.ROIfile)

def save_snapshot(analysis,compression='gzip',directory=None,prefix="snapshot",timestamp=True,frame_num=None):
    # Save a snapshot of the submitted analysis object
    if timestamp:
        prefix+='_'+str(time.time())
    if directory is not None:
        prefix=os.path.join(directory,prefix)
    if compression is None:
        savefile=prefix+'.pkl'
        outfile = open(savefile,'wb')
    elif compression == 'gzip':
        savefile=prefix+'.pklz'
        outfile = gzip.open(savefile,'wb')
    else:
        print('Warning: unsupported file format in save_snapshot(). Using uncompressed format...')
        savefile=prefix+'.pkl'
        outfile = open(savefile,'wb')
    if frame_num is None: # save entire analysis object
        pickle.dump(analysis,outfile)
    else:  # save only specified frame
        anal=analysis # create a copy 
        anal.Frames=[analysis.Frames[frame_num]]
        anal.Frame_num=0
        pickle.dump(anal,outfile)
    outfile.close()
    print('Done saving ',savefile,'...')

def load_snapshot(loadfile,compression='gzip'):
    # Load a snapshot of an analysis object from the specified file
    print('Unpickling analysis snapshot...')
    if compression is None:
        infile = open(loadfile,'rb')
    elif compression == 'gzip':
        infile = gzip.open(loadfile,'rb')
    else:
        print('Warning: unsupported file format in load_snapshot(). Using uncompressed format...')
        infile = open(loadfile,'rb')
        #infile = open(loadfile,'r')
    analysis = pickle.load(infile)
    infile.close()
    print('Done loading ',loadfile,'...')
    return analysis

def resume_analysis(loadfile=None,compression='gzip',old_analysis=None):
    global load_dir
    # If an old analysis is passed, delete it to avoid duplication
    if old_analysis != None:
        print('Purging and deleting old analysis...')
        #try:
        #    old_analysis.close_analysis_window()
        #except:
        #    pass
        old_analysis.purge()
        del old_analysis
    # Resume an analysis in progress
    if loadfile==None:
        loadfile = askopenfilename(initialdir = load_dir,title = "Open snapshot or archive:",
                                   filetypes = (("pklz files","*.pklz"),("all files","*.*")))
        if loadfile==None:
            print('User canceled load...')
            return
    # Record the load directory to be reused next reload
    load_dir=os.path.split(loadfile)[0]
    print('Loading snapshot/archive ',loadfile)
    analysis=load_snapshot(loadfile)
    analysis.create_controlsGUI()
    try:
        analysis.close_analysis_window()
    except:
        pass
    analysis.open_analysis_window()
    analysis.Frames[analysis.Frame_num].show_all_frames()
    analysis.load_ROIset(Frame_num=analysis.Frame_num)
    #analysis.load_ROIset(Frame_num=0)

    # For backwards compatibility, check for analyst, comment and log fields.
    # Create them if they're not already present
    if not hasattr(analysis,'analyst'):
        analysis.analyst='nobody'
        analysis.analyst=simpledialog.askstring(title = "Analyst", prompt = "Specify analyst:",
                                                            initialvalue='')
    print('\nComments:')
    if not hasattr(analysis,'comments'):
        print('comment field not found; creating one...')
        cmt_str='{}; {}; Initiated comment field in existing Analysis object'.format(time.time(),analysis.analyst)
        analysis.comments=[cmt_str]
    try:
        for c in analysis.comments:
            print(c)
    except:
        print('Error in printing comments...') 
    print('\nLog:')
    if not hasattr(analysis,'log'):
        print('log field not found; creating one...')
        log_str='{}; {}; Initiated log field in existing Analysis object with {} Frames'.format(time.time(),analysis.analyst,len(analysis.Frames))
        analysis.log=[log_str]
    try:
        for l in analysis.log:
            print(l)
    except:
        print('Error in printing log...') 
    return analysis

def parse_input_file(input_file,minThreshold=minThreshold,maxThreshold=maxThreshold):
    """Parse an input file to generate the specified FrameSetArray
    """
    analyst = input('\nEnter analyst name/initials: ')
    print('Logs will cite analyst: ',analyst)

    print('\n\nCurrent threshold for resolving blobs is: ',minThreshold)
    inpt = input('Enter new value or hit <cr> to continue: ')
    if inpt!='':
        try:
            minThreshold=int(inpt)
            print('Threshold for resolving blobs set to: ',minThreshold)
        except:
            print('Error in parsing suggested threshold; threshold is still ',minThreshold)
            
    # Parse lines from input file, to determie FrameSetArray structure
    print('\n\nParsing file ',input_file,' to create FrameSetArray...')
    with open(input_file, 'r') as infile:
        for line in infile:
            #line=line.strip('\r')  # drop newline character
            line=line.strip('\n')  # drop newline character
            if len(line)==0: # skip empty lines
                continue
            print(line)
            if line[0]=='#':  # skip comment lines
                #print('skipped comment line')
                continue
            #try:
            if True:
                lbl,info=line.split(': ')
                print('|'+lbl+'|'+info+'|')
                if lbl=='snapshot_dir':
                    snapshot_dir=info
                    print('snapshot_dir = ',snapshot_dir)
                elif lbl=='snapshot_prefix':
                    snapshot_prefix=info
                    print('snapshot_prefix = ',snapshot_prefix)
                elif lbl=='append_timestamp':
                    append_timestamp=eval(info)
                    print('append_timestamp = ',append_timestamp)
                elif lbl=='archive_dir':
                    archive_dir=info
                    print('archive_dir = ',archive_dir)
                elif lbl=='archive_prefix':
                    archive_prefix=info
                    print('archive_prefix = ',archive_prefix)
                elif lbl=='newFrameSet':
                    frame_dir_list=[]
                    frame_file_list=[]
                    archive_label=''
                    name=info
                    comment=''
                elif lbl=='comment':
                    comment=info
                    print('comment = ',comment)
                elif lbl=='frame_dir':
                    frame_dir_list.append(info)
                elif lbl=='frame':
                    frame_file_list.append(info)
                elif lbl=='create_archive':
                    archive_label=info
                    analysis=Analysis(frame_dir_list=frame_dir_list,frame_file_list=frame_file_list,
                                      create_GUI=True,comment=comment,name=name,analyst=analyst)
                                      #create_GUI=False,comment=comment,name=name,analyst=analyst)
                    
                    for i in range(len(analysis.Frames)):
                        analysis.Frames[i].binary_frame(display=False, min_val=minThreshold, max_val=maxThreshold)
                        analysis.Frames[i].segment_frame(method='contour',display_blobs=True,display_ROIs=True,
                                                       use_binary=False,min_val=minThreshold, max_val=maxThreshold)
                        analysis.ROIset_size=nrows*ncols
                        analysis.parse_ROIsets(Frame_num=i)
                    archive_name=archive_prefix
                    if len(archive_label)>0:
                        archive_name+='_'+archive_label
                    analysis.snapshot_dir=snapshot_dir
                    analysis.snapshot_prefix=snapshot_prefix
                    analysis.append_timestamp=append_timestamp
                    analysis.archive_dir=archive_dir
                    analysis.archive_prefix=archive_name
                    print('Creating new archive as: ',archive_dir,'/',archive_name)
                    save_snapshot(analysis,compression='gzip',directory=archive_dir,prefix=archive_name,
                                  timestamp=append_timestamp,frame_num=None)
                else:
                    print('unknown label, skipping entry...')
            #except:
            #    print('error parsing line, skipping entry...')

class Analysis():
    """A class to design and execute analysis of frames from ZooCAM profiles
    """
    global cat_number
    global frm_dir_
    def __init__(self,frame_dir_list=[],frame_file_list=[],scan_dir=True,Frames=[],frame_image_list=[],
                 current_frame=0,create_GUI=True,min_val=minThreshold,comment=None,name=None,analyst=None):
        self.current_frame=current_frame
        self.Frames=[]  # an empty list to contain frames
        # Order of precedence in initializing analysis:
        for Frm in Frames: # 1.  frame_list is a list of Frames
            self.Frames.append(Frm)
        for img in frame_image_list:  # 2.  image_list is a list of tif images
            self.Frames.append(Frame(frame_image=img))
        # 3. Files to be read in and converted to Frames
        if not frame_dir_list and frame_file_list:
            # (a) if dir_list is empty and frame_lst is not, assume frame_list has full paths
            for frm_file in frame_file_list:  
                self.Frames.append(Frame(frame_dir="",frame_file=frm_file))
        if frame_dir_list and not frame_file_list and scan_dir:
            # (b) if dir_list not empty and frame_lst is, assume we should scan directories for frames
            for frm_dir in frame_dir_list:
                tif_list=[f for f in os.listdir(frm_dir) if  \
                          (os.path.isfile(os.path.join(frm_dir, f)) and f.endswith('.tif'))]
                for frm_file in tif_list:  
                    self.Frames.append(Frame(frame_dir=frm_dir,frame_file=frm_file))
        elif len(frame_dir_list)==1 and frame_file_list:
            # (c) if only one directory, assume it contains all frames
            for frm_file in frame_file_list:     # assume it's a str containing a directory name
                self.Frames.append(Frame(frame_dir=frame_dir_list[0],frame_file=frm_file))
        elif len(frame_dir_list)==len(frame_file_list):
            # (d) if a directory for each file, associate crresponding items
            for iframe in range(len(frame_file_list)): 
                self.Frames.append(Frame(frame_dir=frame_dir_list[iframe],frame_file=frame_file_list[iframe]))
        else:
            print('Error in initializing Analysis object: incompatible dir and file list lengths:')
            print('frame_dir_list=',frame_dir_list)
            print('frame_file_list=',frame_file_list)

        self.Frame_num=0
        self.name=name
        self.analyst=analyst
        self.comments=[]
        if comment != None:
            cmt_str='{}; {}; {}'.format(time.time(),analyst,comment)
            self.comments.append(cmt_str)
        log_str='{}; {}; Initiated Analysis object with {} Frames'.format(time.time(),analyst,len(self.Frames))
        self.log=[log_str]
        print(log_str)

        if create_GUI:
            self.create_controlsGUI()
        
        #============================================================================
    def create_controlsGUI(self,ctrl_fig_num=107):
        # Create the controls GUI
        self.ctrl_fig_num=ctrl_fig_num
        #self.ctrl_fig = plt.figure(self.ctrl_fig_num,figsize=ctrl_figsize,facecolor='k')
        ctrl_fig = plt.figure(ctrl_fig_num,figsize=ctrl_figsize,facecolor='k')
        #ctrl_fig = plt.figure(108,figsize=ctrl_figsize,facecolor='k')
        #self.ctrl_fig = plt.figure(figsize=ctrl_figsize,facecolor='k')
        ctrl_fig.subplots_adjust(left=0.01)
        ctrl_fig.canvas.set_window_title('Navigation')
        move_figure(ctrl_fig,ctrl_figpos[0],ctrl_figpos[1])
        #self.ctrl_fig.subplots_adjust(left=0.01)
        #self.ctrl_fig.canvas.set_window_title('Navigation')
        #move_figure(self.ctrl_fig,ctrl_figpos[0],ctrl_figpos[1])

        ctrl_menuitems = []
        for ilabel in range(len(ctrl_labels)):
            print(ilabel)
            label=ctrl_labels[ilabel]
            print(ctrl_labels[ilabel])
            props = ItemProperties(labelcolor='black', bgcolor=ctrl_colors[ilabel],fontsize=ctrl_fontsize, alpha=ctrl_alpha)
            hoverprops = ItemProperties(labelcolor='white', bgcolor=ctrl_colors[ilabel],
                                fontsize=ctrl_fontsize, alpha=ctrl_alpha)

            def ctrl_on_select(item):
                global ctr_number, ctr_color, ctr_label
                #global from_dir_,default_prefix_
                print('Navigator command %d, %s' % (item.number, item.labelstr))
                ctr_number=item.number
                ctr_color=ctrl_colors[item.number]
                ctr_label=item.labelstr
                if ctr_label=='Next frame/ROI set':   # Advance to next set of images, whether in frame or ROI set
                    if self.ROIset==self.Frames[self.Frame_num].num_ROIsets-1: # Already at the last ROI set, advance frame
                        self.Frame_num=min(self.Frame_num+1,len(self.Frames)-1)
                        #self.Frame_num=min(self.Frame_num+1,len(self.Frames))
                        self.Frames[self.Frame_num].show_all_frames()
                        self.load_ROIset(Frame_num=self.Frame_num,ROIset=0)
                    elif self.ROIset<self.Frames[self.Frame_num].num_ROIsets-1: # Advance ROI set
                        self.ROIset+=1
                        self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                elif ctr_label=='Previous frame/ROI set':   # Move back to previous set of images, whether in frame or ROI set
                    if self.ROIset==0 and self.Frame_num>0: # Already at the first ROI set, previous frame exists
                        self.Frame_num-=1                                      # Select previous frame,
                        self.ROIset=self.Frames[self.Frame_num].num_ROIsets-1  # and the last ROI set in that frame
                        #self.ROIset==0   # an alternative, always go to zeroeth frame...
                        self.Frames[self.Frame_num].show_all_frames()
                        self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                    elif self.ROIset>0: # Move back to previous ROI set
                        self.ROIset-=1
                        self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                        #self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                # Separate frame and ROI shifts are currently commented out in the GUI construction 
                elif ctr_label=='Next frame':
                    self.Frame_num=min(self.Frame_num+1,len(self.Frames))
                    self.Frames[self.Frame_num].show_all_frames()
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=0)
                elif ctr_label=='Previous frame':
                    self.Frame_num=max(self.Frame_num-1,0)
                    self.Frames[self.Frame_num].show_all_frames()
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=0)
                elif ctr_label=='Next ROI set':
                    self.ROIset=min(self.ROIset+1,self.Frames[self.Frame_num].num_ROIsets-1)
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                elif ctr_label=='Previous ROI set':
                    self.ROIset=max(self.ROIset-1,0)
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
                elif ctr_label=='Randomize ROIsets':
                    self.parse_ROIsets(Frame_num=self.Frame_num,randomize=True)
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=0)
                elif ctr_label=='Reorder ROIsets':
                    self.parse_ROIsets(Frame_num=self.Frame_num,randomize=False)
                    self.load_ROIset(Frame_num=self.Frame_num,ROIset=0)
                    
                elif ctr_label=='Define group':
                    self.Frames[self.Frame_num].select_group()
                elif ctr_label=='Dissolve group':
                    dis_group=simpledialog.askstring(title = "Dissolve group", prompt = "Enter group #:",
                                                     initialvalue='')
                    if dis_group==None:
                        print('User canceled dissolution...')
                    else:
                        print('dissolving group ',int(dis_group))
                        self.Frames[self.Frame_num].dissolve_group(int(dis_group))

                elif ctr_label=='Save snapshot':
                    #print('frm_dir = ',frm_dir_)
                    #filename = asksaveasfilename(initialdir = "/",title = "Save snapshot:",
                    #                                          filetypes = (("pklz files","*.pklz"),("all files","*.*")),
                    #                                          initialfile =    )
                    prefix=simpledialog.askstring(title = "Save snapshop", prompt = "Enter path/prefix:",
                                                  initialvalue=os.path.join(self.snapshot_dir,self.snapshot_prefix+'_'+self.name))
                    #                              initialvalue=os.path.join(snapshot_dir,snapshot_prefix))
                    if prefix==None:
                        print('User canceled save...')
                    else:
                        print('Saving snapshot with prefix ',prefix)
                        analyst=simpledialog.askstring(title = "Analyst", prompt = "Change or confirm analyst:",
                                                            initialvalue=self.analyst)
                        if analyst!=None:
                            self.analyst=analyst
                        comment=simpledialog.askstring(title = "Comment", prompt = "Add comment:",
                                                       initialvalue='')
                        if comment!=None:
                            if len(comment)>0:
                                cmt_str='{}; {}; {}'.format(time.time(),self.analyst,comment)
                                self.comments.append(cmt_str)
                                #self.comments.append(comment)
                        #log_str=str(time.time())+'; '+self.analyst+'; Saving snapshot'
                        log_str='{}; {}; Saving snapshot'.format(time.time(),self.analyst)
                        self.log.append(log_str)
                        save_snapshot(self,compression='gzip',prefix=prefix,timestamp=False) # changed to never include timestamp
                        #save_snapshot(self,compression='gzip',prefix=prefix,timestamp=self.append_timestamp)
                elif ctr_label=='Load snapshot':
                    #filename =  askopenfilename(initialdir = self.Frames[self.Frame_num].frame_dir,title = "Open snapshot:",
                    #                                       filetypes = (("pklz files","*.pklz"),("all files","*.*")))
                    #if filename==None:
                    #    print('User canceled load...')
                    #else:
                    #    print('Load snapshot ',filename)
                        #self.purge()

                    resume_analysis(old_analysis=self)
                        
                    #    print('Loading from the GUI currently disabled...')
                    #    print('Instead load from the command line, with a command like "B=resume_analysis(filename)"')
                    #    print('or "B=resume_analysis()" for the GUI interface.')
                        #print('Instead load from the command line, with a command like "B=load_snapshot(filename)"')
                        #B=load_snapshot(filename)
                elif ctr_label=='Archive all frames':
                    prefix=simpledialog.askstring(title = "Archive all frames ", prompt = "Enter path/prefix:",
                                                  initialvalue=os.path.join(self.archive_dir,self.archive_prefix))
                    if prefix==None:
                        print('User canceled save...')
                    else:
                        print('Archiving all frames with prefix ',prefix)
                        analyst=simpledialog.askstring(title = "Analyst", prompt = "Change or confirm analyst:",
                                                            initialvalue=self.analyst)
                        if analyst!=None:
                            self.analyst=analyst
                        comment=simpledialog.askstring(title = "Comment", prompt = "Add comment:",
                                                       initialvalue='')
                        if comment!=None:
                            if len(comment)>0:
                                cmt_str='{}; {}; {}'.format(time.time(),analyst,comment)
                                self.comments.append(cmt_str)
                                #self.comments.append(comment)
                        log_str='{}; {}; Saving archive'.format(time.time(),analyst)
                        #log_str=str(time.time())+'; '+self.analyst+'; Saving archive'
                        self.log.append(log_str)
                        save_snapshot(self,compression='gzip',prefix=prefix,timestamp=True,frame_num=self.Frame_num)
                elif ctr_label=='Archive current frame':
                    prefix=simpledialog.askstring(title = "Archive frame "+str(self.Frame_num), prompt = "Enter path/prefix:",
                                                  initialvalue=os.path.join(self.Frames[self.Frame_num].frame_dir,
                                                                            self.Frames[self.Frame_num].frame_file[:-4]))
                    if prefix==None:
                        print('User canceled save...')
                    else:
                        print('Archiving frame %d with prefix %s' % (self.Frame_num,prefix))
                        analyst=simpledialog.askstring(title = "Analyst", prompt = "Change or confirm analyst:",
                                                            initialvalue=self.analyst)
                        if analyst!=None:
                            self.analyst=analyst
                        comment=simpledialog.askstring(title = "Comment", prompt = "Add comment:",
                                                       initialvalue='')
                        if comment!=None:
                            if len(comment)>0:
                                cmt_str='{}; {}; {}'.format(time.time(),analyst,comment)
                                self.comments.append(cmt_str)
                                #self.comments.append(comment)
                        #log_str=str(time.time())+'; '+self.analyst+'; Saving single frame archive'
                        log_str='{}; {}; Saving single frame archive'.format(time.time(),analyst)
                        self.log.append(log_str)
                        save_snapshot(self,compression='gzip',prefix=prefix,timestamp=True,frame_num=self.Frame_num)
            #item = MenuItem(self.ctrl_fig, label, props=props, hoverprops=hoverprops,
            item = MenuItem(ctrl_fig, label, props=props, hoverprops=hoverprops,
                    on_select=ctrl_on_select, number=ilabel)
            ctrl_menuitems.append(item)

        ctrl_menu = Menu(ctrl_fig, ctrl_x0, ctrl_y0, ctrl_menuitems)
        #ctrl_menu = Menu(self.ctrl_fig, ctrl_x0, ctrl_y0, ctrl_menuitems)


    #def prepare_frame(self,Frame_num=0,ROIset=0,randomize=False,segment=False):

    def __copy__(self,filename):
        Acopy=load_snapshot(filename)
        Acopy.open_analysis_window()
        for i in range(len(A.Frames)):
            Acopy.Frames[i].binary_frame(display=False,min_val=5)
            Acopy.Frames[i].segment_frame(method='contour',display_blobs=False,display_ROIs=False,use_binary=False)
            Acopy.parse_ROIsets(Frame_num=i,randomize=False)
        return Acopy

    def purge(self):
        """This function deletes the fields in an Analysis object specific to an existing analysis,
           so that a new analysis can be loaded without duplication."""
        #plt.figure(self.ctrl_fig_num)
        plt.close(self.ctrl_fig_num)
        #plt.figure(self.anal_fig_num)
        #plt.clf()
        #plt.close(self.fig)
        self.close_analysis_window()
        #self.open_analysis_window()
        
        for fld in [self.Frame_num,self.Frames,self.ROIset,self.roi_counter]:
            try:
                del fld
            except:
                pass


    def open_analysis_window(self,nrows=nrows,ncols=ncols,figsize=figsize,figpos=figpos,facecolor=bg_color,fig_numAnal=105):
        # Open a window for displaying and classifying sets of ROIs
        self.anal_fig_num=fig_numAnal
        self.fig,self.axs = plt.subplots(num=fig_numAnal,nrows=nrows, ncols=ncols,
                                         figsize=figsize,facecolor=tuple([i/255 for i in bg_color]))
        plt.tight_layout(pad=plt_pad)
        move_figure(self.fig,figpos[0],figpos[1])
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        # Prepare for parsing ROIs into multiple analysis collages
        self.ROIset_size=nrows*ncols

    def close_analysis_window(self):
        # This function is used primarily to close orphaned windows appearing
        # when a snapshot or archive is reloaded (having an incorrect figure number).
        plt.close(self.fig)
         
    def parse_ROIsets(self,Frame_num=0,randomize=False,ROIset_size=None):
        print('Parsing ROIsets...')
        self.Frame_num=Frame_num
        if ROIset_size is not None:
            self.ROIset_size=ROIset_size
        self.Frames[Frame_num].num_ROIs=len(self.Frames[Frame_num].ROIlist)
        self.Frames[Frame_num].ROIindices=list(range(self.Frames[Frame_num].num_ROIs)) # Create a list of ROI indices
        if randomize is True: # If requested, randomize the order of ROI indices
             self.Frames[Frame_num].ROIindices=random.sample(self.Frames[Frame_num].ROIindices,
                                                            k=len(self.Frames[Frame_num].ROIindices))
        print('self.Frames[Frame_num].ROIindices = ',self.Frames[Frame_num].ROIindices)

        self.Frames[Frame_num].num_ROIsets=self.Frames[Frame_num].num_ROIs // self.ROIset_size
        if self.Frames[Frame_num].num_ROIs % self.ROIset_size > 0:
            self.Frames[Frame_num].num_ROIsets+=1
        self.Frames[Frame_num].ROIset_list=[[] for i in range(self.Frames[Frame_num].num_ROIsets)]
        
        for i in range(self.Frames[Frame_num].num_ROIs):
            iROIset=i // self.ROIset_size
            self.Frames[Frame_num].ROIset_list[iROIset].append(self.Frames[Frame_num].ROIindices[i])
        print('ROIset_list = ',self.Frames[Frame_num].ROIset_list)
            
    def load_ROIset(self,Frame_num=0,ROIset=0):
        print('loading ROIset %d for Frame_num = %d' % (ROIset,Frame_num))
        self.Frame_num=Frame_num  # Update frame and ROIset numbers
        self.ROIset=ROIset
        plt.figure(self.anal_fig_num)
        if len(self.Frames)>1:
            frame_range_str=' (0-'+str(len(self.Frames)-1)+'), '
        else:
            frame_range_str=' (0), '
        if self.Frames[Frame_num].num_ROIsets>1:
            ROIset_range_str=' (0-'+str(self.Frames[Frame_num].num_ROIsets-1)+'), '
        else:
            ROIset_range_str=' (0), '
        title_str='Figure '+str(self.anal_fig_num)+ \
            ',   Frame_num '+str(Frame_num)+frame_range_str+ \
            'ROIset '+str(ROIset)+ROIset_range_str+ \
            'Filename: '+self.Frames[Frame_num].frame_file
        plt.gcf().canvas.set_window_title(title_str)
        
        # bring window to front
        #self.fig.canvas.manager.window.tkraise() # this does not seem to work 
        cfm = plt.get_current_fig_manager()       # but this seems to
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        self.roi_counter=0
        # Clear pre-existing ROIs, if any
        for i, axi in enumerate(self.axs.flat):
            axi.clear()
            axi.set_facecolor(tuple([i/255 for i in bg_color]))
            axi.axis('off')
            #axi.set_facecolor((16,16,16))
        for i, axi in enumerate(self.axs.flat):
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            if i==len(self.Frames[Frame_num].ROIset_list[ROIset]):
                print('breaking at end of ROIset, after %d ROIS...' % len(self.Frames[Frame_num].ROIset_list[ROIset]))
                break
            ROIindex=self.Frames[Frame_num].ROIset_list[ROIset][i]
            self.Frames[Frame_num].ROIlist[ROIindex].counter=self.roi_counter
            # create the padded image
            self.Frames[Frame_num].ROIlist[ROIindex].add_border()
            self.Frames[Frame_num].ROIlist[ROIindex].add_edge()
            self.Frames[Frame_num].ROIlist[ROIindex].show_image(axi)
            self.roi_counter+=1
                
    def load_ROIs(self,Frame_num=0):
        print('loading ROIs for Frame_num = %d' % (Frame_num))
        # bring window to front
        plt.figure(self.anal_fig_num)
        cfm = plt.get_current_fig_manager()       # but this seems to
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        self.Frame_num=Frame_num
        self.roi_counter=0
        # Clear pre-existing ROIs, if any
        for i, axi in enumerate(self.axs.flat):
            axi.clear()
        for i, axi in enumerate(self.axs.flat):
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            print(i,rowid,colid)
            if i==len(self.Frames[Frame_num].ROIlist):
                print('breaking at end of ROIlist, after %d ROIS...' % len(self.Frames[Frame_num].ROIlist))
                break
            self.Frames[Frame_num].ROIlist[i].counter=self.roi_counter
            # create the padded image
            self.Frames[Frame_num].ROIlist[i].add_border()
            self.Frames[Frame_num].ROIlist[i].add_edge()
            self.Frames[Frame_num].ROIlist[i].show_image(axi)
            self.roi_counter+=1
                
    def onClick(self,event):
        # Detect clicks on specific ROIs, determine group number, and change
        # all group members to the currently selected category
        global cat_number, ctrl_number
        print('button is ',event.button)
        # Update classification of all members of the selected ROI's group.
        # We loop through the displayed ROIs twice. In the first pass, we determine
        # the selected ROI, its group number, and the intended new categor. Tehn, we call
        # a function to update the category of all group members.
        # In the second pass, we replot all members of the group to reflect the new category.
        i = 0
        axisNr = None
        for axis in self.fig.axes:
            if axis == event.inaxes:  
                axisNr = i
                print('axisNr=',axisNr)
                print('self.ROIset = =',self.ROIset)
                print('self.Frames[self.Frame_num].ROIset_list = ',self.Frames[self.Frame_num].ROIset_list)
                ROIindex=self.Frames[self.Frame_num].ROIset_list[self.ROIset][axisNr] # the index of the ROI clicked on
                #group_num=self.Frames[self.Frame_num].ROIgroup[ROIindex]              # the group containing that ROI
                group_num=self.Frames[self.Frame_num].ROIlist[ROIindex].group         # the group containing that ROI
                group_members=self.Frames[self.Frame_num].get_group_members(group_num)
                print(ROIindex,self.Frames[self.Frame_num].ROIlist[ROIindex].prev_category,
                      self.Frames[self.Frame_num].ROIlist[ROIindex].category,group_num)
                #print(event.button==MouseButton.LEFT)
                if event.button==MouseButton.RIGHT:  # highlight group containing selected ROI
                    print('highlighting group #',group_num,' with members ',group_members)
                    self.Frames[self.Frame_num].plot_group(group_num)
                    break
                if event.button==MouseButton.LEFT: # Tag image with new category
                    next_category=cat_number
                elif event.button==MouseButton.MIDDLE:
                    next_category=self.Frames[self.Frame_num].ROIlist[ROIindex].prev_category
                    #next_category=self.Frames[self.Frame_num].ROIlist[axisNr].prev_category
                #elif event.button==MouseButton.RIGHT:
                #    next_category=default_category
                self.Frames[self.Frame_num].classify_group(group_num,next_category)
                #self.Frames[self.Frame_num].plot_group(group_num)
                self.Frames[self.Frame_num].show_ROIs_frame()
                break
            i += 1

        # For unclear reasons, the loop approach fails to update the figure. load_ROIset does...
        #print('updating collage for group members ',group_members)
        #plt.figure(self.anal_fig_num)
        #plt.gcf().canvas.draw()
        #cfm = plt.get_current_fig_manager()       # but this seems to
        #cfm.window.attributes('-topmost', True)
        #cfm.window.attributes('-topmost', False)
        self.load_ROIset(Frame_num=self.Frame_num,ROIset=self.ROIset)
        
#============================================================================

