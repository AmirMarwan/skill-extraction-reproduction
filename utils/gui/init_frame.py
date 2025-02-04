import wx
from utils.gui import ConvoGUI_frame_selective, ConvoGUI_frame_support, ConvoGUI_frame_autonomous


class ConvoGUI_frame_init(wx.Frame):
    def __init__(self, parent, title, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super(ConvoGUI_frame_init, self).__init__(parent, title=title, size=(520, 350))

        self.InitUI()

    def InitUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        font = wx.Font(30, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        size = (400, 70)

        self.selective_mode_btn = wx.Button(panel, label='Selective Mode', size=size)
        vbox.Add(self.selective_mode_btn, proportion=0, flag=wx.ALL|wx.CENTER, border=10)
        self.selective_mode_btn.SetFont(font)
        self.selective_mode_btn.Bind(wx.EVT_BUTTON, self.OnSelectiveMode)

        self.support_mode_btn = wx.Button(panel, label='Support Mode (dep)', size=size)
        vbox.Add(self.support_mode_btn, proportion=0, flag=wx.ALL|wx.CENTER, border=10)
        self.support_mode_btn.SetFont(font)
        self.support_mode_btn.Bind(wx.EVT_BUTTON, self.OnSupportMode)

        self.autonomous_mode_btn = wx.Button(panel, label='Autonomous Mode', size=size)
        vbox.Add(self.autonomous_mode_btn, proportion=0, flag=wx.ALL|wx.CENTER, border=10)
        self.autonomous_mode_btn.SetFont(font)
        self.autonomous_mode_btn.Bind(wx.EVT_BUTTON, self.OnAutonomousMode)

        panel.SetSizer(vbox)

    def OnSelectiveMode(self, event):
        selective_frame = ConvoGUI_frame_selective(None, title='Skill Transfer: Selective Mode', *self.args, **self.kwargs)
        selective_frame.Show()
        self.Hide()

    def OnSupportMode(self, event):
        support_frame = ConvoGUI_frame_support(None, title='Skill Transfer: Support Mode', *self.args, **self.kwargs)
        support_frame.Show()
        self.Hide()

    def OnAutonomousMode(self, event):
        autonomous_frame = ConvoGUI_frame_autonomous(None, title='Skill Transfer: Autonomous Mode', *self.args, **self.kwargs)
        autonomous_frame.Show()
        self.Hide()


class ConvoGUI_app(wx.App):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super(ConvoGUI_app, self).__init__()

    def OnInit(self):
        self.frame = ConvoGUI_frame_init(None, title='Skill Transfer', *self.args, **self.kwargs)
        self.frame.Show()
        return True
