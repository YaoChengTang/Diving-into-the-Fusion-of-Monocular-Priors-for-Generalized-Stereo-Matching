import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.frame_utils import writeDispMiddlebury, writeDispKITTI, write_gen



def show_imgs(param, sv_img=False, save2where=None, fontsize=20, szWidth=10, szHeight=5, group=3, if_inter=False):
    """function: visualize the input data
    args:
        paras: [(img, title, colormap), ... ] or
               [{"img":..., "title":..., "cmap":..., "point_x":..., "point_y":..., "point_s":..., "point_c":..., "point_m":..., "colorbar":...}, ... ]
        sv_img: whether to save the visualization
        fontsize : the size of font in title
        szWidth, szHeight: width and height of each subfigure
        group: the columns of the whole figure
    """
    img_num = len(param)
    cols = int(group)
    rows = int(np.ceil(img_num/group))
    sv_title = ""
    color_map = None
    plt_par_list = []
#     plt.clf()
    fig = plt.figure(figsize=(szWidth*cols, szHeight*rows))
    
    for i in np.arange(img_num) :
        if len(param[i])<2 :
            raise Exception("note, each element should be (img, title, ...)")
        
        if isinstance(param[i], list) or isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) :
            name_list = ["img", "title", "cmap", "point_x", "point_y", "point_s", "point_c", "point_m", "point_alpha"]
            plt_par = {}
            for key_id, ele in enumerate(param[i]) :
                plt_par[name_list[key_id]] = ele
        elif isinstance(param[i], dict) :
            plt_par = param[i]
        else :
            raise Exception("unrecognized type: {}, only recept element with type list, np.ndarray, tuple or dict".format(type(param[i])))
        plt_par_list.append(plt_par)
        
        plt.subplot(rows,cols,i+1)
#         plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
        plt.title(plt_par.get("title").replace("\t","   "), fontsize=fontsize)
        im = plt.imshow(plt_par.get("img"), cmap=plt_par.get("cmap"))
        
        if plt_par.get("colorbar") == True :
            plt.colorbar(im, orientation='horizontal', fraction=0.02, pad=0.0004)
        
        if plt_par.get("point_x") is not None and plt_par.get("point_y") is not None :
            plt.scatter(plt_par.get("point_x"), plt_par.get("point_y"), s=plt_par.get("point_s"), c=plt_par.get("point_c"), marker=plt_par.get("point_m"), alpha=plt_par.get("point_alpha"))
        plt.axis("off")
        
#         plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
#         plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
#         plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
#         plt.margins(0,0)
        fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)
        
        if sv_img is True :
            if i>0 :
                sv_title += "-"
            sv_title += plt_par.get("title")
    
    if if_inter :
        from ipywidgets import Output
        output = Output()
        display(output)
        
        @output.capture()
        def onclick(event):
            if event.button == 3 and event.ydata is not None and event.xdata is not None :
                print_info = ""
                for i in np.arange(img_num) :
                    img = plt_par_list[i].get("img")
                    title = plt_par_list[i].get("title")
                    print_info += "{}:\t({},{})-{}\r\n".format(title, int(np.round(event.ydata)), int(np.round(event.xdata)), img[int(np.round(event.ydata)),int(np.round(event.xdata))])
                print(print_info)
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    
    if sv_img is True and save2where is not None :
        plt.savefig(os.path.join(save2where),dpi=600)
    # plt.show(block=False)


def show_dis(param, sv_img=False, fontsize=20, szWidth=10, szHeight=5, group=3):
    """function: visualize the input data
    args:
        paras: [([(x,y,label),(x,y,label),...], title), ... ] or
               [{"x":...shape(num_type,inter), "y":...shape(num_type,inter), "label":...shape(batch,), "title":...}, ... ]
        sv_img: whether to save the visualization
        fontsize : the size of font in title
        szWidth, szHeight: width and height of each subfigure
        group: the columns of the whole figure
    """
    fig_num = len(param)
    cols = group
    rows = np.ceil(fig_num/group)
    sv_title = ""
    color_map = None
    plt.figure(figsize=(szWidth*cols, szHeight*rows))
    
    for i in np.arange(fig_num) :
        if len(param[i])<3 :
            raise Exception("note, each element should be (x, y, title, ...)")
        
        if isinstance(param[i], list) or isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) :
            name_list = ["x", "y", "title", "cmap", "point_x", "point_y", "point_s", "point_c", "point_m"]
            plt_par = {}
            for key_id, ele in enumerate(param[i]) :
                plt_par[name_list[key_id]] = ele
        elif isinstance(param[i], dict) :
            plt_par = param[i]
        else :
            raise Exception("unrecognized type: {}, only recept element with type list, np.ndarray, tuple or dict".format(type(param[i])))
        
        plt.subplot(rows,cols,i+1)
        plt.title(plt_par.get("title"), fontsize=fontsize)
        plt.bar(plt_par.get("x"), plt_par.get("y"), color=plt_par.get("cmap"))
#         plt.legend()
        
        if plt_par.get("point_x") is not None and plt_par.get("point_y") is not None :
            plt.scatter(plt_par.get("point_x"), plt_par.get("point_y"), s=plt_par.get("point_s"), c=plt_par.get("point_c"), marker=plt_par.get("point_m"))
#         plt.axis("off")
        if sv_img is True :
            if i>0 :
                sv_title += "-"
            sv_title += plt_par.get("title")
    
    if sv_img is True :
        plt.savefig(os.path.join(args.save2where,sv_title+".png"))
    # plt.show(block=False)


class Visualizer:
    def __init__(self, root, sv_root, dataset=None, scratch=True):
        self.root = root.rstrip("/")
        self.sv_root = sv_root.rstrip("/")
        self.dataset = dataset
        self.scratch = scratch
        self.sv_root = self.sv_root if self.sv_root[-len(self.dataset):]==self.dataset else os.path.join(self.sv_root, self.dataset)
        self.vis_root = os.path.join(self.sv_root, "analysis")
        print("saving to {}".format(self.sv_root))

    def save_pred_vis(self, flow_pr, imageGT_file):
        assert self.root in imageGT_file, "{} not in {}".format(self.root, imageGT_file)

        # create saving path, /xxx/disp0GT.pfm -> /xxx/disp0GT-pred.pfm
        sv_path = imageGT_file.replace(self.root, self.sv_root)
        pre,lat = os.path.splitext(sv_path)
        sv_path = pre + "-pred" + lat
        if not self.scratch and os.path.exists(sv_path):
            print("{} exists".format(sv_path))
            return True

        # build directory
        sv_dir = os.path.dirname(sv_path)
        if not os.path.exists(sv_dir) :
            os.makedirs(sv_dir)

        # write prediction
        if self.dataset.lower()=="middlebury" :
            writeDispMiddlebury(sv_path, flow_pr)
        elif self.dataset.lower()=="kitti2015" :
            writeDispKITTI(sv_path, flow_pr)
        elif self.dataset.lower()=="eth3d" :
            write_gen(sv_path, flow_pr)
        else:
            raise Exception("such daatset is not supported: {}".format(dataset))
        return True
        
    def analyze(self, flow_pr, image1, image2, flow_gt, valid_gt, imageGT_file, info):
        # print(" ".join(["{}, {}, {}, {}\r\n".format(ele.shape, ele.dtype, ele.min(), ele.max()) \
        #                 for ele in [flow_pr, image1, image2, flow_gt, valid_gt]]))
        
        # create saving path
        file_name = "-".join(imageGT_file.replace(self.root, "").split("/"))
        pre,lat = os.path.splitext(file_name)
        file_name = pre+".png"
        sv_path = os.path.join(self.vis_root, file_name)

        # build directory
        sv_dir = os.path.dirname(sv_path)
        if not os.path.exists(sv_dir) :
            os.makedirs(sv_dir)

        image1 = np.transpose(image1, (1,2,0)).astype(np.uint8)
        image2 = np.transpose(image2, (1,2,0)).astype(np.uint8)
        error_map = np.abs(flow_pr-flow_gt)
        colored_error_map = colorize_error_map(error_map)
        show_imgs([{"img":image1, "title":"Left Image", },
                   {"img":image2, "title":"Right Image", },
                   {"img":flow_pr, "title":"Predicted Disparity", "cmap":'jet', },
                   {"img":valid_gt, "title":"Mask", "cmap":"gray", },
                   {"img":flow_gt, "title":"GT Disparity", "cmap":'jet', },
                   {"img":colored_error_map, "title":"Error Map"+": "+info, "cmap":None, },
                   ], 
                  sv_img=True, save2where=sv_path, if_inter=False, 
                  fontsize=20, szWidth=10, szHeight=5, group=2)
        pass


def colorize_error_map(error_map):
    # Define a custom colormap for errors within 10 (shades of red)
    num_colors_within_10 = 10
    colors_within_10 = [
        (255, 255, 255),  # White
        (255, 248, 220),  # Brown
        (255, 192, 203),  # Pink
        (128, 128, 128),  # Gray
        (128, 0, 128),    # Purple
        (64, 224, 208),   # Turquoise
        (255, 165, 0),    # Orange
        (255, 255, 0),    # Yellow
        (0, 128, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 0, 0),      # Red
    ]

    # Define the color for errors larger than 10 (dark red)
    color_over_10 = (128, 0, 0)  # Dark red color for errors larger than 10

    # Create a blank colored map with the same dimensions as the error map
    colored_map = np.zeros((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8)

    # Map error values within 10 to custom colors
    for i in range(1, num_colors_within_10 + 1):
        colored_map[(error_map<i) & (error_map>=i-1)] = colors_within_10[i - 1]
    colored_map[error_map>=i] = colors_within_10[i - 1]

    # create corlor bar
    color_bar = np.ones((15, error_map.shape[1], 3))*255
    step = error_map.shape[1]//(num_colors_within_10+1)
    for i in range(1+num_colors_within_10):
        color_bar[5:, i*step:(i+1)*step] = colors_within_10[i]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_color = (0, 0, 0)  # Black
    font_thickness = 1
    for i in range(1+num_colors_within_10):
        x = i * step + step // 8
        y = 11
        cv2.putText(color_bar, str(i), (x, y), font, font_scale, font_color, font_thickness)
    
    colored_map = np.vstack((colored_map, color_bar))
    return colored_map.astype(np.uint8)