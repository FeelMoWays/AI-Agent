import os
import sys
import argparse
import json
from dotenv import load_dotenv
from openai import OpenAI
from system_prompt import system_prompt
from call_function import available_functions,call_function



# Force load_dotenv to look in the current directory if it's struggling
load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")

# Safety check: Stop the script immediately if the key is missing
if not api_key:
    print("Error: OPENROUTER_API_KEY is missing!")
    print("Make sure your .env file is in the same folder as this script and contains your key.")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

parser = argparse.ArgumentParser(description="Chatbot")
parser.add_argument("user_prompt", type=str, help="User prompt")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

def main():
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": args.user_prompt},
    ]
    for _ in range(20):
        response = client.chat.completions.create(
            model="openrouter/free", 
            messages=messages,
            tools=available_functions
            )
        message = response.choices[0].message
        messages.append(message)
        if not message.tool_calls:
            break
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result_message = call_function(tool_call=tool_call,verbose=args.verbose)
                if not result_message['content']:
                    raise Exception("empty content")
                messages.append(result_message)
        else:
            raise RuntimeError("Agent has reached the maximum of 20 iterations without a final response")
    if response.usage == None:
        raise RuntimeError("API request failed: Usage data is missing from the response.")
    if args.verbose:
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Response tokens: {response.usage.completion_tokens}\n")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()



