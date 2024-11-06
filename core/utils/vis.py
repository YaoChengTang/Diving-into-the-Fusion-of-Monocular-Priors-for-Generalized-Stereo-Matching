import os
import re
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import frame_utils as frame_utils



def show_imgs(param, sv_img=False, save2where=None, 
              fontsize=20, szWidth=10, szHeight=5, group=3, 
              if_inter=False, dpi=600):
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
        im = plt.imshow(plt_par.get("img"), cmap=plt_par.get("cmap"), alpha=plt_par.get("alpha"))
        
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
        plt.savefig(os.path.join(save2where), dpi=dpi)
    # plt.show(block=False)
    plt.close()


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


def compute_confidence(movement_cur, movement_pre):
    # mask_forward = ((movement_cur<-1) & (movement_cur>=movement_pre-3)) | (movement_cur>=-1)
    mask_forward = np.ones_like(movement_cur)
    mask_direction = ((np.abs(movement_cur)>1) & (np.abs(movement_pre)>1) & (movement_cur*movement_pre>0)) | (np.abs(movement_cur)<=1) | (np.abs(movement_pre)<=1)
    return mask_forward * mask_direction


class Visualizer:
    def __init__(self, root, sv_root, dataset=None, scratch=True, args=None, logger=None):
        self.root    = root.rstrip("/")
        self.sv_root = sv_root.rstrip("/")
        self.dataset = dataset
        self.scratch = scratch
        self.args    = args
        self.sv_root = self.sv_root if self.sv_root[-(1+len(self.dataset)):]=="/"+self.dataset \
                       else os.path.join(self.sv_root, self.dataset)
        self.vis_root = os.path.dirname(self.sv_root) + os.path.join("analysis", self.dataset)

        self.my_print = print if logger is None else logger.info
        self.my_print("saving prediction to {}, visualization to {}".format(self.sv_root, self.vis_root))

    def save_pred_vis(self, flow_pr, imageGT_file):
        assert self.root in imageGT_file, "{} not in {}".format(self.root, imageGT_file)

        # create saving path, /xxx/disp0GT.pfm -> /xxx/disp0GT-pred.pfm
        sv_path = imageGT_file.replace(self.root, self.sv_root)
        pre,lat = os.path.splitext(sv_path)
        sv_path = pre + "-pred" + lat
        if not self.scratch and os.path.exists(sv_path):
            self.my_print("{} exists".format(sv_path))
            return True

        # build directory
        sv_dir = os.path.dirname(sv_path)
        os.makedirs(sv_dir, exist_ok=True)

        # write prediction
        if self.dataset.lower()=="middlebury" :
            frame_utils.writeDispMiddlebury(sv_path, flow_pr)
        elif self.dataset.lower()=="kitti2015" :
            frame_utils.writeDispKITTI(sv_path, flow_pr)
        elif self.dataset.lower()=="eth3d" :
            frame_utils.write_gen(sv_path, flow_pr)
        elif self.dataset.lower()=="booster" :
            frame_utils.writeDispBooster(sv_path, flow_pr)
        else:
            raise Exception("such daatset is not supported: {}".format(dataset))
        return True
    
    def get_xpx(self, key_list):
        pattern = re.compile(r'^\d+(\.\d+)?px_list$')
        px_keys = [key for key in key_list if pattern.match(key)]
        assert len(px_keys) <= 1, f"too many xpx in {key_list} ~ {px_keys}"
        if len(px_keys)==0:
            return "0px_list"
        return px_keys[0]

    def get_error_map(self, pr_list, gt_list, stop_idx=-1):
        error_map_list = []
        colored_error_map_list = []
        for idx in np.arange( len(pr_list) ):
            if stop_idx>0 and idx>=stop_idx:
                break
                
            gt = gt_list[0] if len(gt_list)==1 else gt_list[idx]
            error_map = np.abs(pr_list[idx] - gt)
            error_map[np.isinf(gt) | np.isnan(gt) | (gt==0)] = 0
            error_map_list.append(error_map)

            # colored_error_map = colorize_error_map(error_map, ver_hor="hor")
            colored_error_map = colorize_error_map(error_map, ver_hor="ver")
            colored_error_map_list.append(colored_error_map)
        
        return error_map_list, colored_error_map_list
    
    def get_imp_map(self, error_map_list, stop_idx=-1):
        imp_map_list = []
        colored_imp_map_list = []
        for idx in np.arange( len(error_map_list) ):
            if stop_idx>0 and idx>=stop_idx:
                break
                
            imp_map = np.zeros_like(error_map_list[0]) if idx==0 else error_map_list[idx] - error_map_list[idx-1]
            imp_map_list.append(imp_map)

            # colored_imp_map = colorize_improvement_map(imp_map, ver_hor="hor")
            colored_imp_map = colorize_improvement_map(imp_map, ver_hor="ver")
            colored_imp_map_list.append(colored_imp_map)
        return imp_map_list, colored_imp_map_list

    def get_movement_map(self, pr_list, stop_idx=-1):
        move_map_list = []
        colored_move_map_list = []
        for idx in range(0, len(pr_list)):
            if stop_idx>0 and idx>=stop_idx:
                break
                
            move_map = np.zeros_like(pr_list[idx]) if idx<1 else pr_list[idx] - pr_list[idx-1]
            move_map_list.append(move_map)

            # colored_move_map = colorize_improvement_map(move_map, ver_hor="hor")
            colored_move_map = colorize_improvement_map(move_map, ver_hor="ver")
            colored_move_map_list.append(colored_move_map)
        return move_map_list, colored_move_map_list

    def get_acceleration_map(self, move_map_list, stop_idx=-1):
        # get the difference between movement vector
        colored_acc_map_list =[]
        for idx in range(0, len(move_map_list)):
            if stop_idx>0 and idx>=stop_idx:
                break
                
            acc_map = np.zeros_like(move_map_list[idx]) if idx<2 else move_map_list[idx] - move_map_list[idx-1]

            # colored_acc_map = colorize_improvement_map(acc_map, ver_hor="hor")
            colored_acc_map = colorize_improvement_map(acc_map, ver_hor="ver")
            colored_acc_map_list.append(colored_acc_map)
        return colored_acc_map_list

    def get_mask(self, mask_list, binary_thold, stop_idx=-1):
        colored_mask_list = []
        mask_binary_list = []
        for idx in range(0, len(mask_list)):
            if stop_idx>0 and idx>=stop_idx:
                break
                
            # colored_mask = colorize_confidence(mask, ver_hor="hor")
            colored_mask = colorize_confidence(mask_list[idx], ver_hor="ver")
            colored_mask_list.append(colored_mask)

            mask_binary = mask_list[idx] < binary_thold
            mask_binary_list.append(mask_binary)
        
        return colored_mask_list, mask_binary_list

    def analyze(self, dict_list, imageGT_file, in_one_fig=False):
        """
            dict_list:
                [{"name": "disp",
                  "img_list": [...],
                  "cmap": "jet",
                  "epe_list": [...],
                  "xpx_list": [...],
                  "GT": [tensor],
                  "stop_idx": 20,
                  "improvement": False,
                  "movement": False,
                  "error_map": True,
                  "acceleration": False,
                  "mask": False,
                  "binary_thold": 0.5},
                ]
        """
        # create saving path
        file_name = "-".join(imageGT_file.replace(self.root, "").split("/"))[1:]
        pre,lat = os.path.splitext(file_name)
        file_name = pre+".png"
        sv_path = os.path.join(self.vis_root, file_name)

        # build directory
        sv_dir = os.path.dirname(sv_path)
        os.makedirs(sv_dir, exist_ok=True)

        fig_data_list = []
        for vis_dict in dict_list :
            vis_name = vis_dict.get("name", None)
            assert vis_name is not None, "missing 'name' in vis_dict"
            
            GT       = vis_dict.get("GT", None)
            img_list = vis_dict.get("img_list", [])
            cmap     = vis_dict.get("cmap", None)
            stop_idx = vis_dict.get("stop_idx", -1)

            epe_list = vis_dict.get("epe_list", None)
            xpx_name = self.get_xpx(vis_dict.keys())
            xpx_list = vis_dict.get(xpx_name, None)

            error_map_req    = vis_dict.get("error_map", False)
            movement_req     = vis_dict.get("movement", False)
            improvement_req  = vis_dict.get("improvement", False)
            acceleration_req = vis_dict.get("acceleration", False)
            
            binary_thold = vis_dict.get("binary_thold", 0.5)
            mask_req     = vis_dict.get("mask", False)

            if img_list is None or len(img_list)==0 :
                continue

            # get the colored error maps for the prediction sequence
            if error_map_req :
                error_map_list, colored_error_map_list = self.get_error_map(img_list, GT, stop_idx)
            
            # get the colored improvement map between adjacent iterations,
            # the improvement map of the first iteration is empty.
            if error_map_req and improvement_req :
                imp_map_list, colored_imp_map_list = self.get_imp_map(error_map_list, stop_idx)

            # get the movement vector at each step
            if movement_req :
                move_map_list, colored_move_map_list = self.get_movement_map(img_list, stop_idx)

            # get the difference between movement vector
            if acceleration_req :
                colored_acc_map_list = self.get_acceleration_map(move_map_list, stop_idx)

            # get the colorized mask and binary mask
            if mask_req :
                colored_mask_list, mask_binary_list = self.get_mask(img_list, binary_thold, stop_idx)

            cnt = 0
            for idx in np.arange( len(img_list) ) :
                if stop_idx>0 and idx>=stop_idx:
                    break
                
                info = ""
                if epe_list is not None and len(epe_list) > 0 :
                    info = ": epe~{:.2f}".format(epe_list[idx]) + ", " + \
                            "{}~{:.1f}".format(xpx_name[:-5], epe_list[idx]*100)
                
                if cmap is None or cmap.find("private") == -1 :
                    cnt += 1
                    title = f"{vis_name}-{idx}"
                    fig_data_list += [{"img"  : img_list[idx], 
                                      "title": title, 
                                      "cmap" : cmap, },]
                
                if error_map_req :
                    cnt += 1
                    title = f"Error Map-{idx}" + info
                    fig_data_list += [{"img"  : colored_error_map_list[idx], 
                                       "title": title, 
                                       "cmap" : None, },]
                
                if error_map_req and improvement_req :
                    cnt += 1
                    title = f"Improvement (err[i]-err[i-1])-{idx}"
                    fig_data_list += [{"img"  : colored_imp_map_list[idx], 
                                       "title": title, 
                                       "cmap" : None, },]
                
                if movement_req :
                    cnt += 1
                    title = f"Movement (disp[i]-disp[i-1])-{idx}"
                    fig_data_list += [{"img"  : colored_move_map_list[idx], 
                                       "title": title, 
                                       "cmap" : None, },]

                if acceleration_req :
                    cnt += 1
                    title = f"Acceleration (Move[i]-Move[i-1])-{idx}"
                    fig_data_list += [{"img"  : colored_acc_map_list[idx], 
                                       "title": title, 
                                       "cmap" : None, },]
                
                if mask_req:
                    cnt += 1
                    title = f"Mask-{idx}"
                    fig_data_list += [{"img"  : colored_mask_list[idx], 
                                       "title": title, 
                                       "cmap" : None, },]

                    cnt += 1
                    title = f"Binary Mask-{idx}"
                    fig_data_list += [{"img"  : mask_binary_list[idx], 
                                       "title": title, 
                                       "cmap" : "gray", },]
            if not in_one_fig:
                group = cnt // (stop_idx if stop_idx>0 else len(img_list))
                H,W = img_list[0].shape
                pre,lat = os.path.splitext(sv_path)
                sv_path = pre + f"-sequence-{vis_name}" + lat
                show_imgs(fig_data_list, 
                        sv_img=True, save2where=sv_path, if_inter=False, 
                        fontsize=20, szWidth=np.ceil(W/H)*5, szHeight=5, 
                        group=group, dpi=300)
                fig_data_list = []

        if in_one_fig:
            show_imgs(fig_data_list, 
                    sv_img=True, save2where=sv_path, if_inter=False, 
                    fontsize=20, szWidth=10, szHeight=5, group=2, dpi=300)
        

def colorize_error_map(error_map, ver_hor="hor"):
    # Define a custom colormap for errors within 10 (shades of red)
    num_colors = 10
    colors_map = [
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

    # Create a blank colored map with the same dimensions as the error map
    colored_map = np.zeros((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8)

    # Map error values within 10 to custom colors
    for i in range(1, num_colors + 1):
        colored_map[(error_map<i) & (error_map>=i-1)] = colors_map[i - 1]
    colored_map[error_map>=i] = colors_map[i - 1]

    # create corlor bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)  # Black
    if ver_hor=="hor":
        bar_size = 15
        font_scale = 0.45
        font_thickness = 1
        color_bar = np.ones((bar_size, error_map.shape[1], 3))*255
        step = error_map.shape[1]//(num_colors+1)
        for i in range(1+num_colors):
            color_bar[bar_size//3:, i*step:(i+1)*step] = colors_map[i]
        for i in range(1+num_colors):
            x = i * step + step // 8
            y = bar_size//3*2
            cv2.putText(color_bar, str(i), (x, y), font, font_scale, font_color, font_thickness)
        colored_map = np.vstack((colored_map, color_bar))

    elif ver_hor=="ver":
        bar_size = error_map.shape[1] // 10
        font_scale = 0.9
        font_thickness = 2
        color_bar = np.ones((error_map.shape[0], bar_size, 3))*255
        step = error_map.shape[0]//(num_colors+1)
        for i in range(1+num_colors):
            color_bar[i*step:(i+1)*step, bar_size//3:] = colors_map[i]
        for i in range(1+num_colors):
            y = i * step + step // 4
            x = bar_size//3*2
            cv2.putText(color_bar, str(i), (x, y), font, font_scale, font_color, font_thickness)
        colored_map = np.hstack((colored_map, color_bar))

    return colored_map.astype(np.uint8)


def colorize_confidence(confidence, ver_hor="hor"):
    # Define a custom colormap for errors within 10 (shades of red)
    colors_map = [
        (255, 219, 172),  # Navajo White
        (241, 194, 125),  # Mellow Apricot
        (233, 159, 51 ),
        (224, 172, 105),  # Fawn
        (198, 134, 66 ),  # Peru
        (168, 112, 50 ),
        (141, 85 , 36 ),  # Russet
        (121, 81 , 37 ),
        (103, 63 , 27 ),
        (53 , 32 , 13 ),
    ]
    num_colors = len(colors_map)

    # Create a blank colored map with the same dimensions as the error map
    colored_map = np.zeros((confidence.shape[0], confidence.shape[1], 3), dtype=np.uint8)

    # Map error values within 10 to custom colors
    for i in range(1, num_colors+1):
        colored_map[(confidence>=(i-1)/num_colors) & (confidence<i/num_colors)] = colors_map[i-1]
    colored_map[confidence>=i/num_colors] = colors_map[i-1]

    # create corlor bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)  # Black
    if ver_hor=="hor":
        bar_size = 8
        font_scale = 0.35
        font_thickness = 1
        color_bar = np.ones((bar_size, confidence.shape[1], 3))*255
        step = confidence.shape[1]//num_colors
        for i in range(1,1+num_colors):
            color_bar[bar_size//3:, (i-1)*step:i*step] = colors_map[i-1]
        for i in range(1+num_colors):
            x = i * step
            x = x + step // 8 if i<num_colors else x - step // 8
            y = bar_size//3*2
            cv2.putText(color_bar, "{:.1f}".format(i/num_colors), (x, y), 
                        font, font_scale, font_color, font_thickness)
        colored_map = np.vstack((colored_map, color_bar))

    elif ver_hor=="ver":
        bar_size = confidence.shape[1] // 10
        font_scale = 0.25
        font_thickness = 1
        color_bar = np.ones((confidence.shape[0], bar_size, 3))*255
        step = confidence.shape[0]//num_colors
        for i in range(1,1+num_colors):
            color_bar[(i-1)*step:i*step, bar_size//3:] = colors_map[i-1]
        for i in range(1+num_colors):
            y = i * step
            y = y + step // 4 if i<num_colors else y - step // 4
            x = int(bar_size//3*1.5)
            cv2.putText(color_bar, "{:.1f}".format(i/num_colors), (x, y), 
                        font, font_scale, font_color, font_thickness)
        colored_map = np.hstack((colored_map, color_bar))

    return colored_map.astype(np.uint8)


def colorize_improvement_map(improvement_map, ver_hor="hor"):
    # Define a custom colormap for errors within 10 (shades of red)
    num_colors = 10
    colors_map = [
        (255, 0, 0),      # Red        - (--, -7)
        (0, 128, 0),      # Green      - (-7, -5)
        (255, 165, 0),    # Orange     - (-5, -3)
        (128, 0, 128),    # Purple     - (-3, -1)
        (255, 192, 203),  # Pink       - (-1, -0.5)

        (255, 255, 255),  # White      - (-0.5,0.5)

        (255, 248, 220),  # Brown      - (0.5, 1)
        (128, 128, 128),  # Gray       - (1,   3)
        (64, 224, 208),   # Turquoise  - (3,   5)
        (255, 255, 0),    # Yellow     - (5,   7)
        (0, 0, 255),      # Blue       - (7,  ++)
    ]
    bound_val = np.array([(-np.inf, -7), 
                          (-7,  -5), 
                          (-5,  -3),
                          (-3,  -1),
                          (-1,  -0.3),
                          (-0.3, 0.3),
                          ( 0.3, 1),
                          ( 1,   3),
                          ( 3,   5),
                          ( 5,   7),
                          ( 7,   np.inf),])

    # Create a blank colored map with the same dimensions as the error map
    colored_map = np.zeros((improvement_map.shape[0], improvement_map.shape[1], 3), dtype=np.uint8)

    # Map error values to custom colors
    for idx in range(0, 1+num_colors):
        if idx<=4 :
            colored_map[(improvement_map>=bound_val[idx][0]) & \
                        (improvement_map<bound_val[idx][1])] = colors_map[idx]
        elif idx==5 :
            colored_map[(improvement_map>=bound_val[idx][0]) & \
                        (improvement_map<=bound_val[idx][1])] = colors_map[idx]
        else :
            colored_map[(improvement_map>bound_val[idx][0]) & \
                        (improvement_map<=bound_val[idx][1])] = colors_map[idx]

    # create corlor bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)  # Black
    if ver_hor=="hor":
        bar_size = 15
        font_scale = 0.45
        font_thickness = 1
        color_bar = np.ones((bar_size, improvement_map.shape[1], 3))*255
        step = improvement_map.shape[1]//(num_colors+1)
        for i in range(1+num_colors):
            color_bar[bar_size//3:, i*step:(i+1)*step] = colors_map[i]
        for i in range(1+num_colors):
            x = i * step + step // 8
            y = bar_size//3*2
            cv2.putText(color_bar, str(bound_val[i][0]), (x, y), font, font_scale, font_color, font_thickness)
        colored_map = np.vstack((colored_map, color_bar))

    elif ver_hor=="ver":
        bar_size = improvement_map.shape[1] // 10
        font_scale = 0.9
        font_thickness = 2
        color_bar = np.ones((improvement_map.shape[0], bar_size, 3))*255
        step = improvement_map.shape[0]//(num_colors+1)
        for i in range(1+num_colors):
            color_bar[i*step:(i+1)*step, bar_size//3:] = colors_map[i]
        for i in range(1+num_colors):
            y = i * step + step // 4
            x = bar_size//3*2
            cv2.putText(color_bar, str(bound_val[i][0]), (x, y), font, font_scale, font_color, font_thickness)
        colored_map = np.hstack((colored_map, color_bar))

    return colored_map.astype(np.uint8)