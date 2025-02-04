from argparse import ArgumentParser
import os

if not os.getcwd().lower().endswith('res3'):
    os.chdir('Res3')

from utils.gui import ConvoGUI_app

def main(args):

    profile_change = getattr(args, 'profile_change', False)
    profile_folder = getattr(args, 'profile_folder', f'histories/')
    resources_folder = getattr(args, 'resources_folder', f'resources/raw/')
    auto_input = getattr(args, 'auto_input', False)
    auto_output = getattr(args, 'auto_output', False)
    use_mic = getattr(args, 'use_mic', True)
    max_turns = getattr(args, 'max_turns', 5)
    use_llamaindex = getattr(args, 'use_llamaindex', True)
    
    convo_app = ConvoGUI_app(profile_change=profile_change, profile_folder=profile_folder, resources_folder=resources_folder,
                               auto_input=auto_input, auto_output=auto_output, max_turns=max_turns,
                               use_llamaindex=use_llamaindex, use_mic=use_mic)
    convo_app.MainLoop()


if __name__ == '__main__':
    
    profile_change = False
    profile_folder = f'histories/'
    resources_folder = f'resources/raw/'
    auto_input = False
    auto_output = False
    use_mic = True
    max_turns = 5
    use_llamaindex = True

    # First method
    parser = ArgumentParser()
    parser.add_argument('--profile_change',  action='store_true', default=profile_change,
                       help='Choose whether or not to change the profile after each utterance.')
    parser.add_argument('--profile_folder', action='store_true', default=profile_folder,
                       help='Choose the path to save the conversations in.')
    parser.add_argument('--resources_folder', action='store_true', default=resources_folder,
                       help='Choose the path to the resources to feed LlamaIndex.')
    parser.add_argument('--auto_input', action='store_true', default=auto_input,
                       help='Choose whether or not to have the model converse with itself automatically.')
    parser.add_argument('--auto_output', action='store_true', default=auto_output,
                       help='Choose whether or not to choose the operator\'s response automatically.')
    parser.add_argument('--use_mic', action='store_true', default=use_mic,
                       help='Choose whether or not to use the microphone for input.')
    parser.add_argument('--max_turns', action='store_true', default=max_turns,
                       help='Choose the number of turns for the auto_input option.')
    parser.add_argument('--use_llamaindex', action='store_true', default=use_llamaindex,
                       help='Choose whether or not to use llamaindex to generate responses.')
    args = parser.parse_args()
    main(args)

    # Second method
    # convo = Conversation_class(profile_change=profile_change, profile_folder=profile_folder, resources_folder=resources_folder,
    #                            auto_input=auto_input, auto_output=auto_output, max_turns=max_turns,
    #                            use_llamaindex=use_llamaindex, use_mic=use_mic)
    
