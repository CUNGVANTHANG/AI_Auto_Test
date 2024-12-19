import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
    
class ChatModelGemini:
    def __init__(self, model_id: str = "gemini-1.5-pro", device="cpu"):
        self.device = device
        self.model_id = model_id

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")

        # Define generation configuration
        self.generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Define safety settings
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

    def query_gemini_api(self, system_instruction: str, max_new_tokens: int = 8192):
        # self.generation_config["max_output_tokens"] = max_new_tokens

        model = genai.GenerativeModel(
            model_name=self.model_id,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config,
            system_instruction=system_instruction,
        )
        return model.start_chat(history=[])

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        # Prepare the prompt
        print(">>", context)
        if not context:
            prompt = f"Question: {question}"
        else:
            prompt = f"Context: {context}\nQuestion: {question}"

        system_instruction = (
            "You are an expert in software testing and test case generation. Your task is to assist users "
            "in creating, analyzing, and optimizing test cases for software systems. Provide clear and "
            "concise explanations with practical examples to ensure users understand the purpose, steps, "
            "and expected outcomes of each test case. Focus on ensuring test coverage, reliability, and "
            "adherence to software testing standards."
        )

        # Start a chat session
        chat_session = self.query_gemini_api(system_instruction=system_instruction, max_new_tokens=max_new_tokens)

        response = chat_session.send_message(prompt)

        model_response = response.text

        return model_response