import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

class ChatModelLLM:
    # device is either "cuda" or "cpu"
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        if context == None or context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""Using the information contained in the context, give a detailed answer to the question.
Context: {context}.
Question: {question}"""

        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(formatted_prompt)
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt) :]  # remove input prompt from reponse
        response = response.replace("<eos>", "")  # remove eos token

        return response
    
class ChatModelGemini:
    def __init__(self, model_id: str = "gemini", device="cpu"):
        self.device = device
        self.model_id = model_id

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        if context is None or context == "":
            prompt = f"Question: {question}"
        else:
            prompt = f"Context: {context}\nQuestion: {question}"

        # Make the request to Gemini API
        response = self.query_gemini_api(prompt, max_new_tokens)

        return response

    def query_gemini_api(self, prompt, max_new_tokens):
        url = "https://api.gemini.com/v1/completions"  # Replace with the actual endpoint for Gemini's model
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,  # Model ID, if needed by Gemini
            "prompt": prompt,
            "max_tokens": max_new_tokens,
        }

        # Sending the POST request to Gemini's API
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")  # Adjust based on the response format of Gemini
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Error in generating response"