# AI Auto Test

Using Python version 3.12.4

then execute the command to create a virtual environment

```
python -m venv myenv
```

```
source myenv/bin/activate
```

To use certain LLM models (such as Gemma), you need to create a .env file containing the line 

```
ACCESS_TOKEN=<your hugging face token>
```

To use gemini, you need to create an .env file containing the line 

```
GEMINI_API_KEY=<api_token>
```

Install dependencies with 

```
pip install -r requirements.txt
```

Run with `streamlit run src/app.py`
