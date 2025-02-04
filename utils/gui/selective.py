import pickle as pk
try:
    from utils import Util_lib, OperatingMode
    from utils.gui import CameraPanel
except:
    pass
from threading import Thread, Event, main_thread, current_thread
import os
import time
from time import sleep
import wx
from multiprocessing import Queue


class ConvoGUI_frame_selective(wx.Frame):
    def __init__(self, parent, title, *args, **kwargs):
        self.profile_change = kwargs.pop('profile_change', False)
        self.profile_folder = kwargs.pop('profile_folder', f'histories/')
        self.resources_folder = kwargs.pop('resources_folder', f'resources/raw/')
        self.auto_input = kwargs.pop('auto_input', False)
        self.auto_output = kwargs.pop('auto_output', False)
        self.max_turns = kwargs.pop('max_turns', 5)
        self.max_turns_reset = self.max_turns
        self.use_llamaindex = kwargs.pop('use_llamaindex', True)
        self.use_mic = kwargs.pop('use_mic', True)
        self.mic_start_queue_1 = Queue()
        self.mic_message_queue_1 = Queue()
        self.mic_start_queue_2 = Queue()
        self.mic_message_queue_2 = Queue()
        self.camera_frame_queue = Queue(maxsize=2)
        self.cropped_frame_queue = Queue(maxsize=1)
        self.cropped_frame_queue_2 = Queue(maxsize=1)
        self.frame_gui_queue = Queue(maxsize=1)
        self.image_info_queue = Queue()
        self.image_info_bool = Queue()
        self.documentation_queue = Queue()
        self.documentation_input = Queue()
        self.human_detected_queue = Queue()
        self.human_detected_bool = Queue()
        self.costumer_name_queue = Queue()
        self.costumer_name_bool = Queue()
        self.human_current_queue = Queue()
        self.costumer_name = set()
        self.VCtts = False
        self.VCrvc = False
        self.passthru = False
        self.start_event = Event()
        self.start_vc = Queue()
        self.start_passthrough = Queue()

        self.template = f'./prompts/t_select.txt'
        if self.auto_input:
            print('Auto input is on. The model will converse with itself.')

        super(ConvoGUI_frame_selective, self).__init__(parent, title = title, size = (1000,1000), *args, **kwargs)
        self.InitUI()
        self.SetDoubleBuffered(True)
        self.i = 0
        self.Maximize(True)
        self.Show(True)

    def InitUI(self):
        panel = wx.Panel(self)

        hbox_main = wx.BoxSizer(wx.HORIZONTAL)
        
        vbox_central = wx.BoxSizer(wx.VERTICAL)
        vbox_right = wx.BoxSizer(wx.VERTICAL)
        vbox_left = wx.BoxSizer(wx.VERTICAL)

        font = wx.Font(30, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        font2 = wx.Font(24, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        font3 = wx.Font(16, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        size_button_v = 65
        size_central = (400, size_button_v)
        size_central_h = 800
        size_left_h = 600
        size_right_h = 480
        border_central_v = 5

        self.initconvo_btn = wx.Button(panel, label = 'Ininitialize', size = size_central)
        self.initconvo_btn.Bind(wx.EVT_BUTTON, self.init_convo)
        self.initconvo_btn.SetFont(font)
        vbox_central.Add(self.initconvo_btn, 0, flag = wx.ALIGN_CENTER | wx.TOP, border=border_central_v)

        self.start_btn = wx.Button(panel, label = 'Start conversation', size = size_central)
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_run)
        self.start_btn.SetFont(font)
        vbox_central.Add(self.start_btn, 0, flag = wx.ALIGN_CENTER | wx.TOP, border=border_central_v)
        self.start_btn.Disable()

        vbox_central.AddStretchSpacer()
        self.label_speaker = wx.StaticText(panel, label='Speaker:')
        self.label_speaker.SetFont(font)
        vbox_central.Add(self.label_speaker, 0, flag = wx.ALIGN_LEFT | wx.TOP, border=border_central_v)

        # self.label_mic = wx.StaticText(panel, label='\n\n')
        self.label_mic = wx.TextCtrl(panel, value='',size = (size_central_h, 80), style=wx.TE_MULTILINE | wx.TE_WORDWRAP)
        self.label_mic.SetFont(font2)
        vbox_central.Add(self.label_mic, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        self.sugg_btn = []
        self.sugg_txt = []

        for sugg_indx in range(3):
            vbox_central.AddStretchSpacer()

            self.sugg_btn.append(wx.Button(panel, label = f'最高、美味しい', size = (size_central_h-100, size_button_v)))
            self.sugg_btn[sugg_indx].Bind(wx.EVT_BUTTON, lambda event, index=sugg_indx: self.on_tts(event, sugg_indx=index))
            self.sugg_btn[sugg_indx].SetFont(font)
            self.sugg_btn[sugg_indx].SetBackgroundColour(wx.Colour(255, 0, 0))
            vbox_central.Add(self.sugg_btn[sugg_indx], 0, flag = wx.ALIGN_CENTER | wx.TOP, border=border_central_v)
            self.sugg_btn[sugg_indx].Disable()

            self.sugg_txt.append(wx.TextCtrl(panel, value='ここのベーカリーは最高ですよ！美味しいパンはたくさん置いてますよ！\n',size = (size_central_h, 120), style=wx.TE_MULTILINE | wx.TE_WORDWRAP))
            self.sugg_txt[sugg_indx].SetFont(font2)
            
            vbox_central.Add(self.sugg_txt[sugg_indx], 0, flag = wx.ALIGN_LEFT | wx.TOP, border=border_central_v)


        self.chat_history_t = wx.StaticText(panel, label='Chat history\n', size = (size_left_h*2//5, size_button_v))
        self.chat_history_t.SetFont(font)

        self.chat_history_c = wx.TextCtrl(panel, size=(size_left_h*3//5, size_button_v))
        self.chat_history_c.SetFont(font)

        hbox_left = wx.BoxSizer(wx.HORIZONTAL)
        hbox_left.Add(self.chat_history_t, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        hbox_left.Add(self.chat_history_c, 1, flag=wx.EXPAND | wx.LEFT)
        vbox_left.Add(hbox_left, 0, flag=wx.ALIGN_LEFT | wx.TOP, border=10)

        self.chat_history = wx.TextCtrl(panel, value='', size = (size_left_h, 700), style=wx.TE_MULTILINE | wx.TE_WORDWRAP)
        self.chat_history.SetFont(font2)
        vbox_left.Add(self.chat_history, 0, flag = wx.ALIGN_LEFT | wx.TOP)
        
        choices = ["No Passthrough", "Passthrough", "VC through TTS", "VC through RVC"]
        self.radio_box = wx.RadioBox(panel, label="Voice Mode", choices=choices, majorDimension=1, style=wx.RA_SPECIFY_COLS, pos=(100, 50))
        self.radio_box.SetFont(font3)
        self.radio_box.Bind(wx.EVT_RADIOBOX, self.toggle_audio_mode)
        vbox_left.Add(self.radio_box, 0, flag=wx.ALIGN_LEFT | wx.TOP, border=10)
        self.radio_box.Disable()


        right_panel = wx.Panel(panel, size=(size_right_h, 300))
        vbox_right_tmp = wx.BoxSizer(wx.VERTICAL)
        self.camera_panel = CameraPanel(right_panel, self.frame_gui_queue)
        vbox_right_tmp.Add(self.camera_panel, proportion=1, flag=wx.EXPAND | wx.ALL)
        right_panel.SetSizer(vbox_right_tmp)
        vbox_right.Add(right_panel, proportion=1, flag=wx.EXPAND | wx.ALL)


        self.image_info_gui_t = wx.StaticText(panel, label='Camera info\n', size = (size_right_h, size_button_v))
        self.image_info_gui_t.SetFont(font)
        vbox_right.Add(self.image_info_gui_t, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        self.image_info_gui = wx.TextCtrl(panel, value='\n\n', size = (size_right_h, 240), style=wx.TE_MULTILINE | wx.TE_WORDWRAP)
        self.image_info_gui.SetFont(font2)
        vbox_right.Add(self.image_info_gui, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        self.time_event_t = wx.StaticText(panel, label='Time event\n', size = (size_right_h, size_button_v))
        self.time_event_t.SetFont(font)
        vbox_right.Add(self.time_event_t, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        self.time_event = wx.TextCtrl(panel, value='セールがない', size = (size_right_h, 240), style=wx.TE_MULTILINE | wx.TE_WORDWRAP)
        self.time_event.SetFont(font2)
        vbox_right.Add(self.time_event, 0, flag = wx.ALIGN_LEFT | wx.TOP)

        vbox_right.AddStretchSpacer()

        self.quit = wx.Button(panel, label = 'QUIT', size = size_central)
        self.quit.Bind(wx.EVT_BUTTON, self.OnQuit)
        self.quit.SetFont(font)
        vbox_right.Add(self.quit, 0, flag = wx.ALIGN_CENTER | wx.BOTTOM)


        hbox_main.Add(vbox_left, 1, flag=wx.EXPAND | wx.ALL, border=7)
        hbox_main.Add(vbox_central, 1, flag=wx.EXPAND | wx.ALL, border=7)
        hbox_main.Add(vbox_right, 1, flag=wx.EXPAND | wx.ALL, border=7)

        panel.SetSizer(hbox_main)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
    
    def __set_value(_, widget, value):
        if widget.GetValue() != value:
            widget.SetValue(value)
    
    def __set_label(_, widget, label):
        if widget.GetLabel() != label:
            widget.SetLabel(label)

    def on_timer(self, event):
        self.chat_history_c.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.chat_history_c.Refresh()
        self.timer.Stop()

    def on_run(self, event):
        self.on_run_thread = Thread(target=self.on_run_exec)
        self.on_run_thread.start()
        if current_thread() is main_thread():
            self.timer.Start(1500, oneShot=True)
        else:
            Thread(target=lambda: (sleep(1.5),
            self.chat_history_c.SetBackgroundColour(wx.Colour(255, 255, 255)),
            self.chat_history_c.Refresh())).start()


    def on_run_exec(self):
        self.start_btn.Disable()
        self.__set_label(self.start_btn, 'Restart conversation')
        for sugg_indx in range(3):
            self.sugg_btn[sugg_indx].Disable()

        self.chat_history_c.SetBackgroundColour(wx.Colour(0, 255, 0))
        self.chat_history_c.Refresh()

        # Stop the background thread
        if self.chat_thread != None:
            while self.chat_thread.is_alive():
                self.start_event.clear()
                print('Conversation stopping...')
                sleep(0.5)
                self.mic_start_queue_1.put(0)
                self.mic_start_queue_2.put(0)
                self.human_detected_bool.put(0)
                while not self.human_detected_queue.empty():
                    self.human_detected_queue.get()
                self.costumer_name_bool.put(0)
                while not self.costumer_name_queue.empty():
                    self.costumer_name_queue.get()
                while not self.mic_message_queue_1.empty():
                    self.mic_message_queue_1.get()
                while not self.mic_message_queue_2.empty():
                    self.mic_message_queue_2.get()
                while not self.image_info_bool.empty():
                    self.image_info_bool.get()
                while not self.image_info_queue.empty():
                    self.image_info_queue.get()
                while not self.documentation_queue.empty():
                    self.documentation_queue.get()
                while not self.documentation_input.empty():
                    self.documentation_input.get()
                print('Conversation stopped.')
                # if self.chat_thread.is_alive():
                    # print(10)
                    # print(self.start_event.is_set())
                    # self.on_run(event)
                    # return
                    # self.chat_thread.join()
                # sleep(1)
        
        if len(self.costumer_name):
            profile = list(self.costumer_name)[0]
        else:
            i = 1
            while os.path.exists(f'{self.profile_folder}costumerID_{i}_context_var'):
                i += 1
            profile = f'costumerID_{i}'
        # print(profile)

        # Start the chat thread
        self.chat_thread = Thread(target=self.start_convo, kwargs={'profile':profile})
        self.chat_thread.start()
        self.__set_value(self.chat_history_c, profile)
        
        for sugg_indx in range(3):
            self.sugg_btn[sugg_indx].Enable()
            self.__set_label(self.sugg_btn[sugg_indx], '最高、美味しい')
            self.__set_value(self.sugg_txt[sugg_indx], 'ここのベーカリーは最高ですよ！美味しいパンはたくさん置いてますよ！\n')

        self.start_btn.Enable()

    def OnQuit(self, event):
        self.Close()
    
    def init_convo(self, event):
        self.initconvo_btn.Disable()
        self.util_lib = Util_lib(operating_mode=OperatingMode.SELECTIVE, resources_folder=self.resources_folder,
                                 use_llamaindex=self.use_llamaindex, use_mic=self.use_mic,
                                 camera_frame_queue = self.camera_frame_queue, cropped_frame_queue = self.cropped_frame_queue, cropped_frame_queue_2 = self.cropped_frame_queue_2, frame_gui_queue=self.frame_gui_queue,
                                 human_detected_queue=self.human_detected_queue, human_detected_bool=self.human_detected_bool, human_current_queue=self.human_current_queue,
                                 costumer_name_queue=self.costumer_name_queue, costumer_name_bool=self.costumer_name_bool,
                                 mic_start_queue_1=self.mic_start_queue_1, mic_message_queue_1=self.mic_message_queue_1,
                                 mic_start_queue_2=self.mic_start_queue_2, mic_message_queue_2=self.mic_message_queue_2,
                                 image_info_queue=self.image_info_queue, image_info_bool=self.image_info_bool,
                                 documentation_queue=self.documentation_queue, documentation_input=self.documentation_input,
                                 start_vc=self.start_vc, start_passthrough=self.start_passthrough)
        camera_kwargs={'capture_camera_feed_fps': 15, 'cam_feed_res': (480, 270), 'human_detection_fps': 3, 'costumer_name_fps': 2, 'periodic_camera_fps': 2}
        util_lib_params = {'camera_kwargs': camera_kwargs}
        self.util_lib.initialize(**util_lib_params)
        if not os.path.exists(self.profile_folder):
            os.makedirs(self.profile_folder)
        print('Conversation class initialized.')
        self.chat_thread = None
        self.start_btn.Enable()
        self.radio_box.Enable()
        self.camera_panel.initialize()

    def __reset_max_turns(self):
        self.max_turns = self.max_turns_reset

    def get_response_llamaindex(self, search_prompt = None, **kwargs):
        if not search_prompt:
            return ''
        position = search_prompt.find('[Context]')
        messages = [
            {'role': 'system', 'content': search_prompt[:position]},
            {'role': 'user', 'content': search_prompt[position:]}
        ]
        resources_response = self.util_lib.get_response_llamaindex(messages, **kwargs)
        return resources_response
        
    def get_response(self, message, **kwargs):
        position = message.find('[Context]')
        messages = [
            {'role': 'system', 'content': message[:position]},
            {'role': 'user', 'content': message[position:]}
        ]
        return self.util_lib.get_response(messages, **kwargs)

    def __tts(self, text):
        self.util_lib.tts(text)
        # print('TTS done')

    def on_tts(self, event, sugg_indx):
        self.__colour_btn(0)
        text = self.sugg_txt[sugg_indx].GetValue()
        self.__tts(text=text)
        self.mic_message_queue_2.put((0, text))

    def __colour_btn(self, human_detected_queue):
        colour = wx.Colour(0, 255, 0) if human_detected_queue else wx.Colour(255, 0, 0)
        for sugg_indx in range(3):
            self.sugg_btn[sugg_indx].SetBackgroundColour(colour)

    def chat(self, profile):
        if not self.start_event.is_set():
            return 0

        with open(self.template, 'r') as ff:
            context_text = ''.join(ff.readlines())
        try:
            with open(self.profile_folder+profile+'_context_var', 'rb') as ff:
                context_var = pk.load(ff)
                context_text += ''.join(context_var[1:])
        except FileNotFoundError:
            print('No previous conversation found for this profile. Starting a new one...')
            context_var = [context_text[context_text.find('[Context]'):]]
        
        if not self.start_event.is_set():
            return 0
        self.__set_value(self.chat_history, context_text[context_text.find('置いてますよ！')+len('置いてますよ！'):].replace('Salesman: ','O:').replace('Costumer: ','C:').replace('\n\n','\n'))
        self.chat_history.ShowPosition(self.chat_history.GetLastPosition())

        time_event = self.time_event.GetValue().strip()
        if time_event:
            context_text = context_text.replace('<event>', time_event)
        else:
            context_text = context_text.replace('\n[Time event]\n<event>\n', '')

        if self.image_info_bool.empty():
            self.image_info_bool.put(1)
        if not self.image_info_queue.empty():
            while not self.image_info_queue.empty():
                if not self.start_event.is_set():
                    return 0
                image_info = self.image_info_queue.get()
            self.__set_value(self.image_info_gui, image_info)
            context_text = context_text.replace('<camera>', image_info)
            self.image_info_queue.put(image_info)

        if self.costumer_name_bool.empty():
            self.costumer_name_bool.put(1)
        if not self.costumer_name_queue.empty():
            while not self.costumer_name_queue.empty():
                if not self.start_event.is_set():
                    return 0
                costumer_name = self.costumer_name_queue.get()
            if self.costumer_name.isdisjoint(costumer_name):
                self.costumer_name = costumer_name
                self.on_run(0)
                return 0
            self.costumer_name = costumer_name

        if self.documentation_input.empty():
            self.documentation_input.put(context_text)
        if not self.documentation_queue.empty():
            while not self.documentation_queue.empty():
                if not self.start_event.is_set():
                    return 0
                documentation = self.documentation_queue.get()
            context_text = context_text.replace('<documentation>', documentation)
            self.documentation_queue.put(documentation)


        user_input = ''
        operator_input = ''
        # context_output = ''
        start_time = time.time()
        
        if not self.mic_message_queue_1.empty():
            self.__set_label(self.label_speaker, 'Customer: ')
            while not self.mic_message_queue_1.empty():
                if not self.start_event.is_set():
                    return 0
                mic_message_queue_1 = self.mic_message_queue_1.get()
                user_input += mic_message_queue_1[1]
                self.__set_value(self.label_mic, user_input)
            print(f'User: {user_input}')
            context_text += 'Costumer: ' + user_input + '\n'
        

        if not self.mic_message_queue_2.empty():
            while not self.mic_message_queue_2.empty():
                if not self.start_event.is_set():
                    return 0
                mic_message_queue_2 = self.mic_message_queue_2.get()
                operator_input += mic_message_queue_2[1]
            if mic_message_queue_2[0]:
                if self.passthru or self.VCtts or self.VCrvc:
                    self.__set_label(self.label_speaker, 'Operator: ')
                    self.__set_value(self.label_mic, operator_input)
                    if self.VCtts:
                        self.__tts(operator_input)
                    print(f'Operator: {operator_input}')
                    context_text += 'Salesman: ' + operator_input + '\n'
                else:
                    operator_input = ''
            else:
                context_text += 'Salesman: ' + operator_input + '\n'


        if user_input or operator_input or self.first_time:
            self.first_time = 0
            response = self.get_response_llamaindex(context_text + '\n[Expert suggestion]\n', stop=['\nCostumer', '\n Costumer:'], stream=True)
            print('\n\nGenerating response...\n')
            print('[Expert suggestion]\n')
            event_text = ''
            print(f'time: {time.time()-start_time}')
            suggestions = [['', ''], ['', ''], ['', '']]
            sugg_indx = -1
            sugg_appnd_b = 0
            sugg_appnd_t = 0
            for event in response:
                # print(self.start_event.is_set(), end='')
                if not self.start_event.is_set():
                    return 0
                print('\033[94m' + event_text + '\033[0m', end='')
                if self.use_llamaindex:
                    event_text = event
                else:
                    event_text = event.choices[0].delta.content
                if event_text != None:
                    # context_output += event_text
                    if 'keywords' in event_text:
                        sugg_indx += 1
                        sugg_appnd_b = 1
                        sugg_appnd_t = 0
                    if 'ugges' in event_text:
                        sugg_appnd_b = 0
                        sugg_appnd_t = 1
                    if '\n' in event_text:
                        sugg_appnd_b = 0
                        sugg_appnd_t = 0
                    if sugg_indx>=0 and sugg_indx<3 and sugg_appnd_b==1:
                        suggestions[sugg_indx][0] += event_text
                        self.__set_label(self.sugg_btn[sugg_indx], suggestions[sugg_indx][0][10:])
                    if sugg_indx>=0 and sugg_indx<3 and sugg_appnd_t==1:
                        suggestions[sugg_indx][1] += event_text
                        self.__set_value(self.sugg_txt[sugg_indx], suggestions[sugg_indx][1][11:])
        
            print(f'time: {time.time()-start_time}')
        

        if self.human_detected_bool.empty():
            self.human_detected_bool.put(1)
        if not self.human_detected_queue.empty():
            while not self.human_detected_queue.empty():
                if not self.start_event.is_set():
                    return 0
                human_detected = self.human_detected_queue.get()
                self.__colour_btn(human_detected)

        context_var_tmp = ''
        if user_input:
            context_var_tmp += 'Costumer: ' + user_input + '\n'
        if operator_input:
            context_var_tmp += 'Salesman: ' + operator_input + '\n'
        if context_var_tmp:
            context_var.append(context_var_tmp)
            context_text += context_var[-1]

        with open(self.profile_folder+profile+'_context_var', 'wb') as ff:
            pk.dump(context_var, ff, protocol=pk.HIGHEST_PROTOCOL)
        with open(self.profile_folder+profile+'_context_text', 'wb') as ff:
            ff.write(''.join(context_var).encode('utf-8'))
        
        return context_var_tmp if user_input or operator_input else '<no utterances>'

    def start_convo(self, profile=None):
        self.mic_start_queue_1.put(1)
        self.mic_start_queue_2.put(1)
        self.first_time = 1
        print(f'Profile: {profile}')
        self.start_event.set()
        if not profile and not self.profile_change:
            profile = input('Profile: ')
            if not profile:
                print('exit...')
                self.__reset_max_turns()
                self.start_event.clear()
        while self.start_event.is_set():
            sleep(0.2)
            if self.profile_change:
                profile = input('Profile: ')
                if not profile:
                    print('exit...')
                    self.__reset_max_turns()
                    self.start_event.clear()
            out = self.chat(profile)
            # print(out)
            if not out:
                print('Costumer changed')
                self.__reset_max_turns()
                self.start_event.clear()
                return 0
        print('Conversation stopping...')
        return 0

    def toggle_audio_mode(self, event):
        selection = self.radio_box.GetSelection()
        self.VCrvc = False
        self.VCtts = False
        self.passthru = False
        if selection == 0:
            self.start_vc.put(0)
            self.start_passthrough.put(0)
            print("No voice mode.")
        elif selection == 1:
            self.passthru = True
            self.start_vc.put(0)
            self.start_passthrough.put(1)
            print("Voice passthrough mode.")
        elif selection == 2:
            self.VCtts = True
            self.start_vc.put(0)
            self.start_passthrough.put(0)
            print("VC through TTS mode.")
        elif selection == 3:
            self.VCrvc = True
            self.start_vc.put(1)
            self.start_passthrough.put(0)
            print("VC through RVC mode.")
        print('Audio mode changed')


if __name__ == '__main__':
    import os
    if not os.getcwd().lower().endswith('res3'):
        os.chdir('Res3')
    import sys
    sys.path.append('.')
    from utils import Util_lib, OperatingMode
    from utils.gui import CameraPanel
    
    profile_change = False
    profile_folder = f'histories/'
    resources_folder = f'resources/raw/'
    auto_input = False
    auto_output = False
    use_mic = True
    max_turns = 5
    use_llamaindex = True
    app = wx.App()
    selective_frame = ConvoGUI_frame_selective(None, title='Skill Transfer: Selective Mode',
                                               profile_change=profile_change, profile_folder=profile_folder, resources_folder=resources_folder,
                                               auto_input=auto_input, auto_output=auto_output, max_turns=max_turns,
                                               use_llamaindex=use_llamaindex, use_mic=use_mic)
    selective_frame.Show()
    app.MainLoop()