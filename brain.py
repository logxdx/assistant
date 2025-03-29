import json
from typing import Dict

from openai import OpenAI
from pydantic import BaseModel, Field

import systemMsgs as sysmsg
from config import *
from tools import getCurrentDateTime, tools_dict, tools_list


# Greeting function
def greet_me(name: str = "User", assistant_name: str = ASSISTANT_NAME):
    global ASSISTANT_NAME
    ASSISTANT_NAME = assistant_name
    greetings = [{
        "role": "user",
        "content": f"You are {assistant_name}, greet {name}. {name} is your requester and will be interacting with you. Provide a very-short but a warm greeting to initiate the conversation. The current date and time is {getCurrentDateTime()}."
    }]

    client = OpenAI(base_url=GENERAL_BASE_URL, api_key=GENERAL_API_KEY)
    response = client.chat.completions.create(
        model=GENERAL_MODEL,
        messages=greetings,      # type: ignore
        stream=True
    )
    for chunk in response:
        if text := chunk.choices[0].delta.content:
            yield text

# Function to check if search is required or not
def toolRequired(conversation: list[Dict[str, str]]):
    
    client = OpenAI(base_url=DECISION_BASE_URL, api_key=DECISION_API_KEY)
    response = client.chat.completions.create(
        model=DECISION_MODEL,
        messages=conversation + [sysmsg.tool_use_check_system_prompt.copy()],
        response_format={"type": "json_object"},
    ).choices[0].message.content

    try:
        content = sysmsg.Decision.model_validate_json(response)
    except Exception as e:
        print(f"Error validating decision response: {e}")
        return False
    decision = "false"
    if content.decision:
        decision = content.decision

    if str(decision).lower() == "true":
        print("Using tools...")
        return True
    else:
        print("Not using tools...")
        return False

# Function to get tool results
def toolResults(conversation: list[Dict[str, str]]):
    global ASSISTANT_NAME
    system_prompt = sysmsg.tool_use_results_system_prompt.copy()

    client = OpenAI(base_url=TOOL_BASE_URL, api_key=TOOL_API_KEY)
    message = client.chat.completions.create(
        model=TOOL_MODEL,
        messages=conversation + [system_prompt], # type: ignore
        tools=tools_list, # type: ignore
        tool_choice="required",
        # parallel_tool_calls=False,
    ).choices[0].message

    # conversation.append(message)

    if hasattr(message, 'tool_calls') and message.tool_calls:
        # There may be multiple tool calls in the response
        for tool in message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call := tools_dict.get(tool.function.name):
                print('Function:', tool.function.name)
                arguments = json.loads(tool.function.arguments)
                print('Arguments:', arguments)

                tool_response = function_to_call(**arguments)
                print(f'Function Output: \n---\n{tool_response}\n---')

                # conversation.append({
                #     "role": "user",
                #     "content" : f"Tool: {tool.function.name}"
                #                 f"Arguments: {tool.function.arguments}"
                #                 f"content: {tool_response}"
                # })

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content" : f"{tool_response}",
                })

            else:
                print('Function', tool.function.name, 'not found')
                tool_response += f"Function: {tool.function.name} not found\n"
                conversation.append({
                    "role": "user",
                    "content": tool_response
                })
    else:
        print("No tool calls found in response")
        conversation.append({
            "role": "user",
            "content": str(message.content)
        })

# Get responses from ollama
def get_response(conversation: list[Dict[str, str]]):
    """
    Get a response from the chat model using tool calling.

    Args:
        conversation (list): The conversation history.
        tools (list): list of available tools.
        model (str): The model to use for chatting.
        system_prompt (str): System prompt to guide the assistant.

    Returns:
        str: The final response from the chat model.
    """
    global ASSISTANT_NAME
    tool_use_reqd = toolRequired(conversation)
    if tool_use_reqd:
        try:
            print("Getting tool results...")
            toolResults(conversation)
            print("Got tool results ✅")
            system_prompt = sysmsg.assistant_system_prompt.copy()
            client = OpenAI(base_url=GENERAL_BASE_URL, api_key=GENERAL_API_KEY)
            response = client.chat.completions.create(
                model=GENERAL_MODEL,
                messages=[system_prompt] + conversation,
                response_format={"type": "json_object"},
            )
            print("Response from the model received!!")
            print(response)
            response = response.choices[0].message.content

            if response is None:
                raise Exception("No response from the model!!")
            print("Validating the response...")
            try:
                response = sysmsg.AgentResponse.model_validate_json(response)
                print("Response validated successfully!!")
            except Exception as e:
                print(f"Error validating response: {e}")

            output = ""
            answer = "No descriptive answer available!"
            if response.assistant_response:
                answer = response.assistant_response
                output = "Assistant Response:\n---\n" + response.assistant_response + '\n'
            if response.points:
                output += "\nPoints:\n---\n"
                for point in response.points:
                    output += f"- {point}\n"
            if response.code:
                output += f"\nCode:\n---\n{response.code}\n"
            if response.error:
                output += f"\n{response.error}\n"
            if response.sources:
                output += "\nSources:\n"
                for source in response.sources:
                    output += f"- {source}\n"
            print(output)

            conversation.append({"role": "assistant", "content": output})
            yield answer
        except KeyboardInterrupt:
            print("Keyboard Interrupt!!")
            yield "Keyboard Interrupt!!"
        except Exception as e:
            print(f"An error occurred while using tools: {e}")
            yield f"An error occurred while using tools!!"
    else:
        try:
            system_prompt = sysmsg.assistant_system_prompt.copy()
            client = OpenAI(base_url=GENERAL_BASE_URL, api_key=GENERAL_API_KEY)
            response = client.chat.completions.create(
                model=GENERAL_MODEL,
                messages=conversation + [system_prompt],      # type: ignore
                response_format={"type": "json_object"},
            ).choices[0].message.content

            if response is None:
                print("No response from the model!!")

            print("Validating the response...")
            response = sysmsg.AgentResponse.model_validate_json(response)
            print("Response validated successfully!!")

            output = ""
            answer = "No descriptive answer available!"
            if response.assistant_response:
                answer = response.assistant_response
                output = "Assistant Response:\n---\n" + response.assistant_response + '\n'
            if response.points:
                output += "\nPoints:\n---\n"
                for point in response.points:
                    output += f"- {point}\n"
            if response.code:
                output += f"\nCode:\n---\n{response.code}\n"
            if response.error:
                output += f"\n{response.error}\n"
            if response.sources:
                output += "\nSources:\n---\n"
                for source in response.sources:
                    output += f"- {source}\n"
            print(output)

            conversation.append({"role": "assistant", "content": output})
            yield answer
        except Exception as e:
            print(f"{e}")
            yield f"An error occurred while generating the response!!"

# Main function
if __name__ == "__main__":

    ASSISTANT_NAME = "Jarvis"
    conversation = []

    print("JARVIS: ", end="", flush=True)
    for _ in greet_me():
        print(_, end="", flush=True)
    print()

    while True:
        print("\nUser: ", end="")
        user_input = input()
        if user_input:
            if user_input.lower()[0] == "/":
                if user_input.lower() in ["/exit", "/quit", "/q"]:
                    print("Session Closed")
                    break
                if user_input.lower() == "/clear":
                    conversation.clear()
                    print("Conversation History Cleared")
                    continue
            conversation.append({"role": "user", "content": user_input})
            assistant_response = ""
            response = get_response(conversation)
            for chunk in response:
                assistant_response += str(chunk)
            conversation.append({"role": "assistant", "content": assistant_response})
