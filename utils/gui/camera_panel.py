import wx
from threading import Thread

class CameraPanel(wx.Panel):
    def __init__(self, parent, frame_gui_queue):
        wx.Panel.__init__(self, parent)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_frame, self.timer)
        self.timer.Start(100)  # Refresh every n-ms (about 1000/n FPS)
        
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.frame_gui_queue = frame_gui_queue
        self.current_frame = None
        self.width_frame, self.height_frame = 100, 100
        self.width_target, self.height_target = 100, 100

    def initialize(self):
        initialize_thread = Thread(target=self.__initialize_callback)
        initialize_thread.daemon = True
        initialize_thread.start()

    def __initialize_callback(self):
        width_target, height_target = self.GetSize()
        self.current_frame = self.frame_gui_queue.get()
        self.height_frame, self.width_frame = self.current_frame.shape[:2]
        if self.width_frame/self.height_frame > width_target/height_target:
            self.width_target = width_target
            self.height_target = int(self.height_frame*self.width_target/self.width_frame)
        else:
            self.height_target = height_target
            self.width_target = int(self.width_frame*self.height_target/self.height_frame)


    def on_paint(self, event):
        if self.current_frame is not None:
            dc = wx.PaintDC(self)
            image = wx.Image(self.width_frame, self.height_frame, self.current_frame)
            image = image.Scale(self.width_target, self.height_target)
            bitmap = wx.Bitmap(image)
            dc.DrawBitmap(bitmap, 0, 0)

    def __update_frame_callback(self):
        try:
            if not self.frame_gui_queue.empty():
                self.current_frame = self.frame_gui_queue.get()
                self.Refresh()
        except:
            pass

    def update_frame(self, event):
        self.frame_updater = Thread(target=self.__update_frame_callback)
        self.frame_updater.daemon = True
        self.frame_updater.start()

    def stop_camera(self):
        self.timer.Stop()

