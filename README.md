
# oa
Python interface to OpenAi


To install:	```pip install oa```


# Usage

## Just-do-it: A minimal-boilerplate facade to OpenAI stuff

For the typical tasks you might want to use OpenAI for.

Note there's no "enter API KEY here" code. That's because if you don't have it in the place(s) it'll look for it, it will simply ask you for it, and, with your permission, put it in a hidden file for you, so you don't have to do this every time.




```python
import oa
```


```python
print(oa.complete('chatGPT is a'))
```

     chatbot based on OpenAI's GPT-2, a natural language processing



```python
print(oa.chat('Act as a chatGPT expert. List 5 useful prompt templates'))
```

    Sure, here are 5 useful prompt templates that can be used in a chatGPT session:
    
    1. Can you provide some more details about [topic]?
    - Examples: Can you provide some more details about the symptoms you're experiencing? Or Can you provide some more details about the issue you're facing with the website?
    
    2. How long have you been experiencing [issue]?
    - Examples: How long have you been experiencing the trouble with your internet connection? Or How long have you been experiencing the pain in your back?
    
    3. Have you tried any solutions to resolve [issue]?
    - Examples: Have you tried any solutions to resolve the error message you're seeing? Or Have you tried any solutions to resolve the trouble you're having with the application?
    
    4. What is the specific error message you are receiving?
    - Examples: What is the specific error message you are receiving when you try to log in? Or What is the specific error message you are receiving when you try to submit the form?
    
    5. Is there anything else you would like to add that might be helpful for me to know?
    - Examples: Is there anything else you would like to add that might be helpful for me to know about your situation? Or Is there anything else you would like to add that might be helpful for me to know about the product you are using?



```python
url = oa.dalle('An image of Davinci, pop art style')
print(url)
```

    https://oaidalleapiprodscus.blob.core.windows.net/private/org-AY3lr3H3xB9yPQ0HGR498f9M/user-7ZNCDYLWzP0GT48V6DCiTFWt/img-pNE6fCWGN3eJGj7ycFwZREhi.png?st=2023-04-22T22%3A17%3A03Z&se=2023-04-23T00%3A17%3A03Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-22T21%3A08%3A14Z&ske=2023-04-23T21%3A08%3A14Z&sks=b&skv=2021-08-06&sig=5j6LPVO992R95dllAAjbmOXzS0MORD06Fo8unwtGNl0%3D



```python
from IPython.display import Image

Image(url=url)
```




<img src="https://oaidalleapiprodscus.blob.core.windows.net/private/org-AY3lr3H3xB9yPQ0HGR498f9M/user-7ZNCDYLWzP0GT48V6DCiTFWt/img-pNE6fCWGN3eJGj7ycFwZREhi.png?st=2023-04-22T22%3A17%3A03Z&se=2023-04-23T00%3A17%3A03Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-22T21%3A08%3A14Z&ske=2023-04-23T21%3A08%3A14Z&sks=b&skv=2021-08-06&sig=5j6LPVO992R95dllAAjbmOXzS0MORD06Fo8unwtGNl0%3D"/>



## Raw form - When you need to be closer to the metal

The `raw` object is a thin layer on top of the `openai` package, which is itself a thin layer over the web requests. 

What was unsatisfactory with the `openai` package is (1) finding the right function, (2) the signature of the function once you found it, and (3) the documentation of the function. 
What raw contains is pointers to the main functionalities (not all available -- yet), with nice signatures and documentation, extracted from the web service openAPI specs themselves. 

For example, to ask chatGPT something, the openai function is `openai.ChatCompletion.create`, or to get simple completions, the function is `openai.Completion.create` whose help is:

```
Help on method create in module openai.api_resources.completion:

create(*args, **kwargs) method of builtins.type instance
    Creates a new completion for the provided prompt and parameters.
    
    See https://platform.openai.com/docs/api-reference/completions/create for a list
    of valid parameters.
```

Not super helpful. It basically tells you to got read the docs elsewhere. 

The corresponding `raw` function is `raw.completion`, and it's help is a bit more like what you'd expect in a python function.



```python
help(oa.raw.chatcompletion)
```

    Help on Wrap in module openai.api_resources.chat_completion:
    
    chatcompletion
        Creates a new chat completion for the provided messages and parameters.
        
                See https://platform.openai.com/docs/api-reference/chat-completions/create
                for a list of valid parameters.
        
        chatcompletion(
                model: str
                messages: List[oa.openai_specs.Message]
                *
                temperature: float = 1
                top_p: float = 1
                n: int = 1
                stream: bool = False
                stop=None
                max_tokens: int = None
                presence_penalty: float = 0
                frequency_penalty: float = 0
                logit_bias: dict = None
                user: str = None
        )
        
        :param model: ID of the model to use. Currently, only `gpt-3.5-turbo` and `gpt-3.5-turbo-0301` are supported.
        
        :param messages: The messages to generate chat completions for, in the [chat format](/docs/guides/chat/introduction).
        
        :param temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.
        
        :param top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
        
        :param n: How many chat completion choices to generate for each input message.
        
        :param stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
        
        :param stop: Up to 4 sequences where the API will stop generating further tokens.
        
        :param max_tokens: The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
        
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details)
        
        :param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
        
        :param user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids).
    



```python
prompt = 'List 5 top prompt engineering tricks to write good prompts for chatGPT'

resp = oa.raw.chatcompletion(
    messages=[
        {"role": "system", "content": "You are an expert at chatGPT"},
        {"role": "user", "content": prompt},
    ],
    model='gpt-3.5-turbo-0301',
    temperature=0.5,
    max_tokens=300
)
resp
```




    <OpenAIObject chat.completion id=chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr at 0x11fd467a0> JSON: {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "message": {
            "content": "Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:\n\n1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.\n\n2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.\n\n3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.\n\n4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.\n\n5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.",
            "role": "assistant"
          }
        }
      ],
      "created": 1682207713,
      "id": "chatcmpl-78HMPgn3oy2fuvm6sLCgOsQvnTVYr",
      "model": "gpt-3.5-turbo-0301",
      "object": "chat.completion",
      "usage": {
        "completion_tokens": 214,
        "prompt_tokens": 36,
        "total_tokens": 250
      }
    }



```python
print(resp['choices'][0]['message']['content'])
```

    Sure, here are 5 top prompt engineering tricks to write good prompts for chatGPT:
    
    1. Be Specific: Ensure that your prompts are specific and clear. The more specific your prompt, the better the response from chatGPT. Avoid using vague or ambiguous language.
    
    2. Use Open-Ended Questions: Open-ended questions encourage chatGPT to provide more detailed and personalized responses. Avoid using closed-ended questions that can be answered with a simple yes or no.
    
    3. Include Context: Providing context to your prompts helps chatGPT to better understand the topic and provide more relevant responses. Include any necessary background information or details to help guide chatGPT's response.
    
    4. Use Emotion: Including emotion in your prompts can help chatGPT generate more engaging and relatable responses. Consider using prompts that evoke emotions such as happiness, sadness, or excitement.
    
    5. Test and Refine: Experiment with different prompts and evaluate the responses from chatGPT. Refine your prompts based on the quality of the responses and continue to test and improve over time.

