import json
import os
from pathlib import Path
from textwrap import dedent
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize Rich console
console = Console()

# --------------------------------------------------------------------------------
# 1. Configure OpenAI client and load environment variables
# --------------------------------------------------------------------------------

client = OpenAI(base_url="http://localhost:11434", api_key="ollama")
# CODE_MODEL = "gpt-4o-mini"
# CODE_MODEL = "qwen2.5:3b"
# CODE_MODEL = "llama-3.3-70b"
# CODE_MODEL = "qwen-2.5-coder-32b"
# CODE_MODEL = "gemini-2.0-flash"

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------

class FileToCreate(BaseModel):
    path: str
    content: str

class FileToRead(BaseModel):
    path: str

# NEW: Diff editing structure
class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

class AssistantResponse(BaseModel):
    assistant_reply: str
    files_to_read: Optional[List[FileToRead]] = None
    files_to_create: Optional[List[FileToCreate]] = None
    files_to_edit: Optional[List[FileToEdit]] = None

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------

system_prompt = dedent("""
    You are an elite assistant called JARVIS with decades of experience across all programming domains, literature and creative knowledge.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning or creativity.

    Core capabilities:
    1. Content Analysis & Discussion
       - Analyze content with expert-level insight
       - Explain complex concepts clearly

    2. File Operations:
        a) Read existing files
          - Access user-provided file contents for context
          - Analyze multiple files to understand project structure
      
        b) Edit existing files
          - You can also write content to files using this function
          - Make precise changes using diff-based editing
          - Modify specific sections while preserving context
          - Suggest refactoring improvements
                       
        c) Create new files
            - Generate new files with code snippets or content
            - You can also use this function to clear the contents of the file by creating a new file with the same name and empty string as content

    You can read, create, and edit files as needed. You may perform simultaneous operations on same/multiple files. Example: You are asked to read a file, analyze its content, and then edit the file based on the analysis. So you add file to read, analyze the content, and then add file to edit with the changes you want to make, in a single response.

    You are to work with only the directories and files provided below:
    Code directory: ./work_dir/code/
    Scratchpad file: ./work_dir/scratchpad.md

    All code you create must be saved in the code directory. 
    Any ideation/thoughts must be written in the scratchpad file. Always append your thoughts to the scratchpad file. Do not clear the scratchpad file until asked explicitly.
    Remember that scratchpad is the common place of thought and ideas which connects you and the user. Use it to help the user understand your thought process and to help you understand the user's thought process. You can also use it to save any important information that you think might be useful in the future.
    You are not allowed to access any other files or directories outside of these paths.

    Output Format:
    You must provide responses in this JSON structure:
    {
        "assistant_reply": "Your main explanation or response, it should be concise and to the point.",
        "files_to_create": [
            {
                "path": "path/to/file",
                "content": "file content here"
            }
        ]
        "files_to_read": [
            {
                "path": "path/to/file",
            }
        ],
        "files_to_edit": [
            {
                "path": "path/to/file",
                "original_snippet": "exact content to be replaced or heading to fill with content",
                "new_snippet": "new code or content to insert or replace the original snippet"
            }
        ]
    }

    Guidelines:
    1. For normal responses, use 'assistant_reply'
    2. For creating files, use 'files_to_create' with precise file paths and content
    3. For reading files, use 'files_to_read' with precise file paths
    4. For editing files:
       - Use 'files_to_edit' for precise changes
       - Include precise changes for original_snippet to locate the change
       - Ensure new_snippet maintains proper indentation
       - Prefer targeted edits over full file replacements
    5. Always explain your changes and reasoning

    Remember: You're a helpful assistant - be thorough, precise, and thoughtful in your solutions.
""")

# --------------------------------------------------------------------------------
# 4. Helper functions 
# --------------------------------------------------------------------------------

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensures any dirs exist
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]✓[/green] Created/updated file at '[cyan]{file_path}[/cyan]'")
    
    # Record the action
    conversation_history.append({
        "role": "assistant",
        "content": f"✓ Created/updated file at '{file_path}'"
    })
    
    # NEW: Add the actual content to conversation context
    normalized_path = normalize_path(str(file_path))
    conversation_history.append({
        "role": "system",
        "content": f"Content of file '{normalized_path}':\n\n{content}"
    })

# NEW: Show the user a table of proposed edits and confirm
def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit:
        return
    
    # Enable multi-line rows by setting show_lines=True
    table = Table(title="Proposed Edits", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("File Path", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    
    console.print(table)

# NEW: Apply diff edits
def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    try:
        content = read_local_file(path)
        if original_snippet in content:
            if len(original_snippet)>0:
                updated_content = content.replace(original_snippet, new_snippet, 1)
            else:
                updated_content = content + "\n\n" + new_snippet
            create_file(path, updated_content)  # This will now also update conversation context
            console.print(f"[green]✓[/green] Applied diff edit to '[cyan]{path}[/cyan]'")
            conversation_history.append({
                "role": "assistant",
                "content": f"✓ Applied diff edit to '{path}'"
            })
        else:
            # NEW: Add debug info about the mismatch
            console.print(f"[yellow]⚠[/yellow] Original snippet not found in '[cyan]{path}[/cyan]'. No changes made.", style="yellow")
            console.print("\nExpected snippet:", style="yellow")
            console.print(Panel(original_snippet, title="Expected", border_style="yellow"))
            console.print("\nActual file content:", style="yellow")
            console.print(Panel(content, title="Actual", border_style="yellow"))
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found for diff editing: '[cyan]{path}[/cyan]'", style="red")

def try_handle_add_command(user_input: str) -> bool:
    """
    If user_input starts with '/add ', read that file and insert its content
    into conversation as a system message. Returns True if handled; else False.
    """
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        file_path = user_input[len(prefix):].strip()
        try:
            content = read_local_file(file_path)
            conversation_history.append({
                "role": "system",
                "content": f"Content of file '{file_path}':\n\n{content}"
            })
            console.print(f"[green]✓[/green] Added file '[cyan]{file_path}[/cyan]' to conversation.\n")
        except OSError as e:
            console.print(f"[red]✗[/red] Could not add file '[cyan]{file_path}[/cyan]': {e}\n", style="red")
            return False
        return True
    return False

def ensure_file_in_context(file_path: str) -> bool:
    """
    Ensures the file content is in the conversation context.
    Returns True if successful, False if file not found.
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except OSError:
        console.print(f"[red]✗[/red] Could not read file '[cyan]{file_path}[/cyan]' for editing context", style="red")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path."""
    return str(Path(path_str).resolve())

def clear_context(user_input: str) -> bool:
    """Clear the conversation context if user requests."""
    global conversation_history
    if user_input.strip().lower() == "/clear":
        os.system("cls")
        conversation_history.clear()
        console.print("[red]✓[/red] Cleared conversation context.", style="red")
        conversation_history = [
            {"role": "system", "content": system_prompt}
        ]
        return True
    return False

def info():
    console.print(Panel.fit(
        "[bold blue]Welcome to Code Engineer! 🌊",
        border_style="blue"
    ))
    console.print(
        "To include a file in the conversation, use '[bold magenta]/add path/to/file[/bold magenta]'.\n"
        "To clear conversation, use '[bold magenta]/clear[/bold magenta]'.\n"
        "Type '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end.\n"
    )

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------

conversation_history = [
    {"role": "system", "content": system_prompt}
]

# --------------------------------------------------------------------------------
# 6. OpenAI API interaction with streaming
# --------------------------------------------------------------------------------

def guess_files_in_message(user_message: str) -> List[str]:
    """
    Attempt to guess which files the user might be referencing.
    Returns normalized absolute paths.
    """
    recognized_extensions = [".css", ".html", ".js", ".py", ".json", ".md"]
    potential_paths = []
    for word in user_message.split():
        if any(ext in word for ext in recognized_extensions) or "/" in word:
            path = word.strip("',\"")
            try:
                normalized_path = normalize_path(path)
                potential_paths.append(normalized_path)
            except (OSError, ValueError):
                continue
    return potential_paths

def stream_openai_response(user_message: str):
    """
    Streams the chat completion response and handles structured output.
    Returns the final AssistantResponse.
    """
    # Attempt to guess which file(s) user references
    potential_paths = guess_files_in_message(user_message)
    
    valid_files = {}

    # Try to read all potential files before the API call
    for path in potential_paths:
        print("[blue]Reading file:[/blue]", path)
        try:
            content = read_local_file(path)
            valid_files[path] = content  # path is already normalized
            file_marker = f"Content of file '{path}'"
            # Add to conversation if we haven't already
            if not any(file_marker in msg["content"] for msg in conversation_history):
                conversation_history.append({
                    "role": "system",
                    "content": f"{file_marker}:\n\n{content}"
                })
        except OSError:
            error_msg = f"Cannot proceed: File '{path}' does not exist or is not accessible"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            continue

    # Now proceed with the API call
    conversation_history.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model=CODE_MODEL,
            messages=conversation_history,
            response_format={"type": "json_object"},
            max_completion_tokens=16000,
            # stream=True
        )

        console.print(f"\nAssistant>", style="bold blue", end="")
        full_content = stream.choices[0].message.content

        # full_content = ""
        # for chunk in stream:
        #     if chunk.choices[0].delta.content:
        #         content_chunk = chunk.choices[0].delta.content
        #         full_content += content_chunk
        #         console.print(content_chunk, end="")

        parsed_response = json.loads(full_content)
        console.print(json.dumps(parsed_response, indent=2), style="white")

        try:
            # Ensure assistant_reply is present
            if "assistant_reply" not in parsed_response:
                parsed_response["assistant_reply"] = ""

            # If assistant tries to edit files not in valid_files, remove them
            if "files_to_edit" in parsed_response and parsed_response["files_to_edit"]:
                new_files_to_edit = []
                for edit in parsed_response["files_to_edit"]:
                    try:
                        edit_abs_path = normalize_path(edit["path"])
                        # If we have the file in context or can read it now
                        if edit_abs_path in valid_files or ensure_file_in_context(edit_abs_path):
                            edit["path"] = edit_abs_path  # Use normalized path
                            new_files_to_edit.append(edit)
                    except (OSError, ValueError):
                        console.print(f"[yellow]⚠[/yellow] Skipping invalid path: '{edit['path']}'", style="yellow")
                        continue
                parsed_response["files_to_edit"] = new_files_to_edit

            response_obj = AssistantResponse(**parsed_response)

            # Save the assistant's textual reply to conversation
            conversation_history.append({
                "role": "assistant",
                "content": response_obj.assistant_reply
            })

            return response_obj

        except json.JSONDecodeError:
            error_msg = "Failed to parse JSON response from assistant"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            return AssistantResponse(
                assistant_reply=error_msg,
                files_to_create=[]
            )

    except Exception as e:
        error_msg = f"API error: {str(e)}"
        console.print(f"\n[red]✗[/red] {error_msg}", style="red")
        return AssistantResponse(
            assistant_reply=error_msg,
            files_to_create=[]
        )

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def main():

    info()

    while True:
        try:
            user_input = console.input("\n[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # If user is reading a file
        if try_handle_add_command(user_input):
            continue

        # If user is clearing the context
        if clear_context(user_input):
            info()
            continue

        # Get streaming response from OpenAI
        response_data = stream_openai_response(user_input)

        # Create any files if requested
        if response_data.files_to_create:
            for file_info in response_data.files_to_create:
                create_file(file_info.path, file_info.content)

        if response_data.files_to_read:
            for file_info in response_data.files_to_read:
                try:
                    content = read_local_file(file_info.path)
                    # add to context
                    conversation_history.append({
                        "role": "system",
                        "content": f"Content of file '{file_info.path}':\n\n{content}"
                    })
                except OSError:
                    console.print(f"[red]✗[/red] Could not read file '{file_info.path}'", style="red")

        if response_data.files_to_edit:
            show_diff_table(response_data.files_to_edit)
            confirm = console.input(
                "\nDo you want to apply these changes? ([green]y[/green]/[red]n[/red]): "
            ).strip().lower()
            if confirm == 'y':
                for edit_info in response_data.files_to_edit:
                    apply_diff_edit(edit_info.path, edit_info.original_snippet, edit_info.new_snippet)
            else:
                console.print("[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow")

    console.print("[blue]Session finished.[/blue]")

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------
# 8. Agent
# --------------------------------------------------------------------------------

def agent(user_input: str):
    """
    Main function that processes user input and returns the assistant's response.
    """
    # If user is reading a file
    if try_handle_add_command(user_input):
        return None

    # If user is clearing the context
    if clear_context(user_input):
        return None

    # Get streaming response from OpenAI (DeepSeek)
    response_data = stream_openai_response(user_input)

    # Create any files if requested
    if response_data.files_to_create:
        for file_info in response_data.files_to_create:
            create_file(file_info.path, file_info.content)

    if response_data.files_to_read:
        for file_info in response_data.files_to_read:
            try:
                content = read_local_file(file_info.path)
                # add to context
                conversation_history.append({
                    "role": "system",
                    "content": f"Content of file '{file_info.path}':\n\n{content}"
                })
            except OSError:
                console.print(f"[red]✗[/red] Could not read file '{file_info.path}'", style="red")

    # Show and confirm diff edits if requested
    if response_data.files_to_edit:
        show_diff_table(response_data.files_to_edit)
        confirm = console.input(
            "\nDo you want to apply these changes? ([green]y[/green]/[red]n[/red]): "
        ).strip().lower()
        if confirm == 'y':
            for edit_info in response_data.files_to_edit:
                apply_diff_edit(edit_info.path, edit_info.original_snippet, edit_info.new_snippet)
        else:
            console.print("[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow")

    return "Assistant response: " + response_data.assistant_reply

