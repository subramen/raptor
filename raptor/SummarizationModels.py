import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import requests

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class AzureLlamaSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name='llama-2-70b-chat') -> None:
        self.endpoint = "https://Llama-2-70b-chat-suraj-demo-serverless.eastus2.inference.ai.azure.com/v1/completions"
        self.key = os.environ['KEY_70B']


    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        message = f"Write a summary of the following, including as many key details as possible: {context}:"
    
        payload = {
            "prompt": message,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stop": stop_sequence,
        }
        headers = {'Content-Type':'application/json', 'Authorization':(self.key)}
        response = requests.post(self.endpoint, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['text']
        else:
            print(f"Request failed with status code {response.status_code}")
            return ""