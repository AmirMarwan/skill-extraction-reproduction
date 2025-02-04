from time import sleep

def periodic_documentation(openai_key, doc_index, documentation_queue, documentation_input):
    import openai
    openai.api_key = openai_key
    while True:
        query_model = doc_index.as_query_engine()
        sleep(0.5)
        if not documentation_input.empty():
            doc_input = documentation_input.get()
            # print(doc_input)
            prompt_input = 'Extract the relevent context respose using this information. Also take into account the time delta since the last camera info and context to check if the respose should be "no utterance is needed"\n\n'
            prompt_input += f'Previous [Camera]: {doc_input[0]}\n'
            prompt_input += f'Previous [Context]: {doc_input[1]}\n'
            prompt_input += f'Time Delta: {doc_input[2]:.1f}\n'
            prompt_input += f'[Camera]: {doc_input[3]}\n'
            resources = str(query_model.query(prompt_input))
            documentation_queue.put(resources)

