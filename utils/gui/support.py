import pickle as pk
from utils.lib import Util_lib, OperatingMode
from threading import Thread
import os
import time
from time import sleep
import wx
from multiprocessing import Queue



class ConvoGUI_frame_support(wx.Frame):
    def __init__(self, parent, title, *args, **kw):
        self.profile_change = kw.pop('profile_change')
        self.profile_folder = kw.pop('profile_folder')
        self.resources_folder = kw.pop('resources_folder')
        self.auto_input = kw.pop('auto_input')
        self.auto_output = kw.pop('auto_output')
        self.max_turns = kw.pop('max_turns')
        self.max_turns_reset = self.max_turns
        self.use_llamaindex = kw.pop('use_llamaindex')
        self.use_mic = kw.pop('use_mic')
        self.mic_start_queue_1 = Queue()
        self.mic_message_queue_1 = Queue()
        self.mic_start_queue_2 = Queue()
        self.mic_message_queue_2 = Queue()

        self.template = f'./prompts/t_supp.txt'
        if self.auto_input:
            print('Auto input is on. The model will converse with itself.')

        super(ConvoGUI_frame_support, self).__init__(parent, title = title, size = (1000,500), *args, **kw)
        self.InitUI()
        self.i = 0
        self.Maximize(True)
        self.Show(True)

    def InitUI(self):
        panel = wx.Panel(self)
        vbox_central = wx.BoxSizer(wx.VERTICAL)

        font = wx.Font(30, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        size = (400, 70)
        
        self.initconvo_btn = wx.Button(panel, label = 'Ininitialize', size = size)
        self.initconvo_btn.Bind(wx.EVT_BUTTON, self.init_convo)
        self.initconvo_btn.SetFont(font)
        vbox_central.Add(self.initconvo_btn, 0, flag = wx.ALIGN_CENTER)

        self.start_btn = wx.Button(panel, label = 'Start conversation', size = size)
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_run)
        self.start_btn.SetFont(font)
        vbox_central.Add(self.start_btn, 0, flag = wx.ALIGN_CENTER)
        self.start_btn.Disable()

        # self.browser_btn = wx.Button(panel, label = 'Relaunch browsers', size = size)
        # self.browser_btn.Bind(wx.EVT_BUTTON, self.on_browser_relaunch)
        # self.browser_btn.SetFont(font)
        # vbox_central.Add(self.browser_btn, 0, flag = wx.ALIGN_CENTER)
        # self.browser_btn.Disable()

        self.label_speaker = wx.StaticText(panel, label='\n')
        self.label_speaker.SetFont(font)
        vbox_central.Add(self.label_speaker, 0, flag = wx.ALIGN_LEFT)

        self.label_mic = wx.StaticText(panel, label='\n\n')
        self.label_mic.SetFont(font)
        vbox_central.Add(self.label_mic, 0, flag = wx.ALIGN_LEFT)

        self.label_advice = wx.StaticText(panel, label='\n\n\n')
        self.label_advice.SetFont(font)
        vbox_central.Add(self.label_advice, 0, flag = wx.ALIGN_LEFT)

        self.quit = wx.Button(panel, label = 'QUIT', size = size)
        self.quit.Bind(wx.EVT_BUTTON, self.OnQuit)
        self.quit.SetFont(font)
        vbox_central.Add(self.quit, 0, flag = wx.ALIGN_CENTER)

        panel.SetSizer(vbox_central)

    # def on_browser_relaunch(self, event):
    #     self.browser_btn.Disable()
    #     self.util_lib.open_browser_instances()
    #     self.browser_btn.Enable()


    def on_run(self, event):
        self.start_btn.Disable()

        i = 1
        while os.path.exists(f'{self.profile_folder}costumerID_{i}_context_var'):
            i += 1
        profile = f'costumerID_{i}'
        print(profile)

        # Start the background thread
        self.chat_thread = Thread(target=self.start_convo, kwargs={'profile':profile})
        self.chat_thread.start()


    def OnQuit(self, event):
        self.Close()
    
    def init_convo(self, event):
        self.initconvo_btn.Disable()
        self.util_lib = Util_lib(operating_mode=OperatingMode.SUPPORT, resources_folder=self.resources_folder,
                                 use_llamaindex=self.use_llamaindex, use_mic=self.use_mic,
                                 mic_start_queue_1=self.mic_start_queue_1, mic_message_queue_1=self.mic_message_queue_1,
                                 mic_start_queue_2=self.mic_start_queue_2, mic_message_queue_2=self.mic_message_queue_2)
        if not os.path.exists(self.profile_folder):
            os.makedirs(self.profile_folder)
        print('Conversation class initialized.')
        self.start_btn.Enable()
        # self.browser_btn.Enable()

    def __reset_max_turns(self):
        self.max_turns = self.max_turns_reset

    def get_response_llamaindex(self, search_prompt = None, **kwargs):
        if not search_prompt:
            return ''
        #return ''
        position = search_prompt.find('[Context]')
        messages = [
            {'role': 'system', 'content': search_prompt[:position]},
            {'role': 'user', 'content': search_prompt[position:]}
        ]
        resources_response = self.util_lib.get_response_llamaindex(messages, **kwargs)
            # print(resources_response)
        return resources_response
        
    def get_response(self, message, **kwargs):
        position = message.find('[Context]')
        messages = [
            {'role': 'system', 'content': message[:position]},
            {'role': 'user', 'content': message[position:]}
        ]
        return self.util_lib.get_response(messages, **kwargs)



    def chat(self, profile):
        sleep(0.1)
        with open(self.template, 'r') as ff:
            context_text = ''.join(ff.readlines())
        try:
            with open(self.profile_folder+profile+'_context_var', 'rb') as ff:
                context_var = pk.load(ff)
                context_text += ''.join(context_var[1:])
        except FileNotFoundError:
            print('No previous conversation found for this profile. Starting a new one...')
            context_var = [context_text[context_text.find('[Context]'):]]


        # print(context_text[context_text.find('[Context]'):] + 'Costumer: ')
        # user_input = self.__get_input(context_text)
        # if not user_input:
        #     return 0
        user_input = ''
        # if self.mic_message_queue_1.empty():
        #     user_input = '<says nothing>'
        # else:
        while not self.mic_message_queue_1.empty():
            user_input += self.mic_message_queue_1.get()
            if not self.mic_message_queue_2.empty():
                break
        
        if user_input:
            self.label_speaker.SetLabel('Customer: ')
            self.label_mic.SetLabel(user_input)
            self.label_mic.Wrap(500)
            print(f'User: {user_input}')
            print('\n\nGenerating response...\n')
            context_input = context_text + 'Costumer: ' + user_input + '\nSalesman: '
        else:
            context_input = context_text + '\nSalesman: '


        start_time = time.time()

        operator_input = ''
        start_time = time.time()
        context_output = ''

        if user_input:
            response = self.get_response_llamaindex(context_input + '\n[Expert suggestion]\n', stop=['\nCostumer', '\n Costumer:'], stream=True)

            print('[Expert suggestion]\n')
            event_text = ''
            print(f'time: {time.time()-start_time}')
            wrap_i = 1
            for event in response:
                # if answer == '':
                #    print(f'time: {time.time()-start_time}')
                print('\033[94m' + event_text + '\033[0m', end='')

                if self.use_llamaindex:
                    event_text = event
                else:
                    event_text = event.choices[0].delta.content
                if event_text != None:
                    context_output += event_text
                # context_output = context_output.replace('keyword:','')
                # context_output = context_output.replace('「','')
                # context_output = context_output.replace('」','')
                context_output = context_output.replace('\n\n','\n')
                if len(context_output) > 20*wrap_i:
                    context_output = context_output[:20*wrap_i] + ' ' + context_output[20*wrap_i:]
                    wrap_i += 1
                self.label_advice.SetLabel(context_output)
                self.label_advice.Wrap(500)


        if self.auto_output:
            print(context_output)
        else:
            context_output = ''
            # if self.mic_message_queue_2.empty():
            #     context_output = '<says nothing>'
            # else:
            while not self.mic_message_queue_2.empty():
                context_output += self.mic_message_queue_2.get()
            print(f'Operator: {context_output}')
            self.label_speaker.SetLabel('Operator: ')
            self.label_mic.SetLabel(context_output)
            self.label_mic.Wrap(500)
        
        context_output = context_output.strip()
        context_var_tmp = ''
        if user_input:
            context_var_tmp += 'Costumer: ' + user_input + '\n'
        if context_output:
            context_var_tmp += 'Salesman: ' + context_output + '\n'
        if context_var_tmp:
            context_var.append(context_var_tmp)
            context_text += context_var[-1]
        
        # print(f'time: {time.time()-start_time}')

        with open(self.profile_folder+profile+'_context_var', 'wb') as ff:
            pk.dump(context_var, ff, protocol=pk.HIGHEST_PROTOCOL)
        with open(self.profile_folder+profile+'_context_text', 'wb') as ff:
            ff.write(''.join(context_var).encode('utf-8'))
        
        return context_output if context_output else '<no response>'


    def start_convo(self, profile=None):
        self.mic_start_queue_1.put(1)
        self.mic_start_queue_2.put(1)
        print(f'Profile: {profile}')
        self.start_bool = 1
        if not profile and not self.profile_change:
            profile = input('Profile: ')
            if not profile:
                print('exit...')
                self.__reset_max_turns()
                self.start_bool = 0
        while self.start_bool:
            sleep(0.1)
            if self.profile_change:
                profile = input('Profile: ')
                if not profile:
                    print('exit...')
                    self.__reset_max_turns()
                    self.start_bool = 0
            out = self.chat(profile)
            if not out:
                print('exit...')
                self.__reset_max_turns()
                self.start_bool = 0
        self.mic_start_queue_1.put(0)
        self.mic_start_queue_2.put(0)
        self.start_btn.Enable()
        return 0

