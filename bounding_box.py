class BBox:

    def __init__(self, xmin, xmax, ymin, ymax, filename, label, ID, width=-1, height=-1, is_voc=False): 
        self.xmin = xmin
        self.xmax = xmax 
        self.ymin = ymin 
        self.ymax = ymax 
        self.filename = filename 
        self.label = label  
        if width == -1:
            self.width = self.xmax - self.xmin 
        else: 
            self.width = width 
        if height == -1: 
            self.height = self.ymax - self.ymin 
        else: 
            self.height = height
        self.ID = ID
        self.is_voc = is_voc


    def get_boundary_list(self, voc=False): 
        if voc == False: 
            return [self.xmin, self.ymin, self.xmax, self.ymax] 
        else: 
            return [self.xmin, self.xmax, self.ymin, self.ymax]


    def print(self):         
        print("bounding-box ID = {0:d}:\nfilename: {1:s}\nwidth:\t{2:5d}\nheight:\t{3:5d}\nxmin:\t{4:5d}\nxmax:\t{5:5d}\nymin:\t{6:5d}\nymax:\t{7:5d}\nlabel:\t{8:s}\n".format(self.ID, self.filename, self.width, self.height, self.xmin, self.xmax, self.ymin, self.ymax, self.label))  


    def get_info(self): 
        info = "bounding-box ID = {0:d}:\nfilename: {1:s}\nwidth:\t{2:5d}\nheight:\t{3:5d}\nxmin:\t{4:5d}\nxmax:\t{5:5d}\nymin:\t{6:5d}\nymax:\t{7:5d}\nlabel:\t{8:s}\n".format(self.ID, self.filename, self.width, self.height, self.xmin, self.xmax, self.ymin, self.ymax, self.label)
        return info


    def change_size(self, x): 
        self.xmin = int(self.xmin * x) 
        self.xmax = int(self.xmax * x) 
        self.ymin = int(self.ymin * x) 
        self.ymax = int(self.ymax * x) 
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin 


    # method by which the size of bounding boxes can be altered,
    # e.g. for adding additional space. 
    def manipulate_bbox(self, width, height, adding=2):
        #width and height of the matrix, not the bbox!
        #ymin, ymax, xmin, xmax
        # 0     1     2     3
        if self.ymin - adding < 0: 
            self.ymin = 0 
        else: 
            self.ymin -= adding 

        if self.xmin - adding < 0: 
            self.xmin = 0 
        else: 
            self.xmin -= adding 

        if self.ymax + adding > width - 1: 
            self.ymax = width - 1 
        else: 
            self.ymax += adding

        if self.xmax + adding > height - 1: 
            self.xmax = height - 1 
        else: 
            self.xmax += adding
        


"""
xmin = 1
xmax = 1000
ymin = 2
ymax = 20
filename = "Filename.endung" 
label = 1
ID = 53854

bbox = BBox(xmin, xmax, ymin, ymax, filename, label, ID)


print(bbox.get_boundary_list() ) 
print() 
bbox.print()
"""