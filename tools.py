import asyncio
import base64
import datetime
import json
import os
import socket
import webbrowser
from pathlib import Path
from textwrap import dedent
from urllib.parse import quote

import cv2
import pyperclip
import requests
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CacheMode,
                      CrawlerRunConfig, LLMExtractionStrategy)
from crawl4ai.async_configs import LlmConfig
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv
from openai import OpenAI
from PIL import ImageGrab

import systemMsgs as sysmsg
from config import *
from utils import *

load_dotenv()

# Constants
working_directory = Path("./work_dir").resolve()
temp_directory = Path("./work_dir/temp").resolve()
SCRATCHPAD_PATH = Path("./work_dir/scratchpad.md").resolve()

working_directory.mkdir(parents=True, exist_ok=True)
temp_directory.mkdir(parents=True, exist_ok=True)


## General Functions
# Check current time
def getCurrentDateTime():
    """
    Get the current time in a human-readable format.

    Returns:
        str: The current time as a string.
    """

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p, %d %B %Y")

# Check internet connectivity
def checkInternetConnectivity():
    """
    Check if the system has internet connectivity.

    Returns:
        bool: True if the internet is connected, False otherwise.
    """

    try:
        # Attempt to connect to a reliable public DNS server (Google's 8.8.8.8)
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return "Internet connection available"
    except OSError:
        return "Internet connection not available"

# Get current weather
def getCurrentWeather(city:str) -> str:
    """
    Get the current weather for a city
    
    Args:
        city: The city to get the weather for
    """

    city_encoded = city.replace(" ", "+")
    base_url = f"http://wttr.in/{city_encoded}?format=j1"
    response = requests.get(base_url, timeout=10)
    data = response.json()

    current_stats = {}
    current_condition = data['current_condition'][0]
    weather = data['weather']

    # current conditions
    date = current_condition['localObsDateTime'][:10]
    humidity = current_condition['humidity']
    precip = current_condition['precipMM']
    pressure = current_condition['pressure']
    temp = current_condition['temp_C']
    uvindex = current_condition['uvIndex']
    visibility = current_condition['visibility']
    weather_desc = current_condition['weatherDesc'][0]['value']
    wind_speed = current_condition['windspeedKmph']

    # weather forecast for current day
    current_stats[date] = {
        "weather": weather_desc,
        "temperature": temp,
        "visibility": visibility,
        "humidity": humidity,
        "precipitation": precip,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "uvindex": uvindex,
    }

    stats = {}
    # weather forecast for coming days
    for day in weather:
        date = day['date']
        max_temp = day['maxtempC']
        min_temp = day['mintempC']
        avg_temp = day['avgtempC']
        uv = day['uvIndex']
        stats[date] = {
            "max_temp": max_temp,
            "min_temp": min_temp,
            "avg_temp": avg_temp,
            "uvindex": uv
        }

    output = f"Current weather in {city}:\n"
    output += f"{json.dumps(current_stats, indent=2)}\n"
    output += f"Weather forecast for the coming days:\n"
    output += f"{json.dumps(stats, indent=2)}\n"

    return output


## Web Scraper Functions
# Scraper helper function
async def scraper_helper(url: str, query: str):
    instruction = dedent(f"""
                         Extract information relevant to \"{query}\".
                         Include key concepts, explanations, examples, and essential details.
                         Format the output as a clean structured markdown with proper headers. Remove any unnecessary information.""")

    browser_config = BrowserConfig(
        user_agent_mode="random",
        text_mode=True,
        light_mode=True,
        extra_args=["--disable-extensions", "--disable-infobars", "--disable-dev-tools", "--window-size=1920x1080"],
        # verbose=True,
    )

    llm_config = LlmConfig(
            provider=f"{SCRAPER_PROVIDER}/{SCRAPER_MODEL}",
            api_token=SCRAPER_API_KEY,
            base_url=SCRAPER_BASE_URL,
        )

    llm_strategy = LLMExtractionStrategy(
        llmConfig=llm_config,
        instruction=instruction,
        chunk_token_threshold=4096,
        overlap_rate=0.1,
        apply_chunking=True,
        input_format="fit_markdown",
        # verbose=True,
    )

    llm_filter = LLMContentFilter(
        llmConfig=llm_config,
        instruction=instruction,
        chunk_token_threshold=4096,
        overlap_rate=0.1,
        # verbose=True,
    )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=llm_filter,
        options={
            "body_width": 100,
            "ignore_emphasis": True,
            "ignore_links": True,
            "ignore_images": True,
            "escape_html": True,
        }
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        markdown_generator=markdown_generator,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        process_iframes=False,
        remove_overlay_elements=True,
        excluded_tags=["form", "header", "footer", "script", "style", "nav", "img", "a"],
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

        if result.success:
            print("Scraping successful.")
            try:
                extracted_content = result.extracted_content
                print("\nContent extracted successfully.")
                response = json.loads(extracted_content)

                result = ""
                for item in response:
                    error = item.get("error", None)
                    if error == "true":
                        print(f"Error in item: {item.get("index")}")
                        continue
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(content)
                    elif isinstance(content, str):
                        content = content.strip()
                    if content:
                        result += content + "\n"
                    else:
                        print(f"No content found in item: {item.get("index")}")
                        continue
                
                result = result.strip()
                print("\nContent parsed successfully.")
                print("Content length: ", len(result))

                print("---")
                llm_filter.show_usage()
                llm_strategy.show_usage()
                print("---")
                print("\n")

                return result, True
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing JSON response: {e}")
                return "Could not parse the extracted content.", False
        else:
            print(f"Error in scraping: {result.error_message}")
            return "Could not scrape the URL.", False

# Scrape URL function
def scrapeURL(url: str, query: str):
    result = asyncio.run(scraper_helper(url, query))
    return result

# Web browse function
NUMBER_OF_URLS_TO_SCRAPE = 5
def deepSearch(query: str, num_results: int = NUMBER_OF_URLS_TO_SCRAPE):
    """
    Perform a deep search, scraping mutiple urls.

    Args:
        query (str): The query to search for.
        num_results (int): The number of search results to return.

    Returns:
        str: A detailed analysis on the query.
    """

    if isinstance(num_results, str):
        try:
            num_results = int(num_results)
        except Exception as e:
            print(f"Error: {e}")
            print("The number of results should be an integer. Defaulting to 3.")
            num_results = NUMBER_OF_URLS_TO_SCRAPE
    elif isinstance(num_results, int) and num_results < 1:
        print(f"The number of results should be a positive integer. Defaulting to {NUMBER_OF_URLS_TO_SCRAPE}.")
        num_results = NUMBER_OF_URLS_TO_SCRAPE
    elif isinstance(num_results, float):
        print(f"The number of results should be an integer. Defaulting to {NUMBER_OF_URLS_TO_SCRAPE}.")
        num_results = NUMBER_OF_URLS_TO_SCRAPE

    EngineURL = f"{SEARXNG_URL}"
    params = {
        "q": query,
        "engines": ["brave", "duckduckgo", "google", "bing"],
        "format": "json",
        "language": "en",
    }
    try:
        print("Scraping the web...\n")
        response = requests.get(EngineURL, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Error in deepSearch: {e}")
        return "An error occurred while browsing the web."

    results = response.json()['results']
    web_results: list[dict] = []
    num_res = 0
    try:
        for result in results:
            if num_res >= num_results:
                break
            url = result['url']
            title = result['title']
            category = result['category']
            print()
            print(f"URL: {url}")
            print(f"Title: {title}")
            web_content, access = scrapeURL(url=url, query=query)
            if not access:
                print(f"Error in scraping URL: {url}")
                continue
            print(f"Scraped content length: {len(web_content)}")
            print(f"Content: {web_content[:200]}...")

            if len(web_content) > 50000:
                client = OpenAI(base_url=SUMMARISATION_BASE_URL, api_key=SUMMARISATION_API_KEY)
                web_content = client.chat.completions.create(
                    model=SUMMARISATION_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Extract and summarise only the relevant information from the given content which answers or completely fulfills the query: {query}. Structure the output in a clean markdown format with proper headers. Remove any unnecessary information."
                        },
                        {
                            "role": "user",
                            "content": f"CONTENT: {web_content}"
                        }
                    ],
                ).choices[0].message.content

            web_results.append(f"TITLE: {result['title']}\n--URL: {url}\n---\n{web_content}---\n\n")

            # web_results.append(
            #     {
            #         "title": title,
            #         "url": url,
            #         "content": web_content,
            #         "category": category
            #     }
            # )

            num_res += 1
    except Exception as e:
        print(f"Error in deepSearch: {e}")
        return "An error occurred while browsing the web."
    
    print(f"{num_res} Results found.")
    return web_results

# Web search function
def webSearch(query: str, engines: list[str], num_results: int = 10):
    """
    Perform a web search for the given query and get the top results.

    Args:
        query (str): The query to search for.
        engines (list): The search engines to use. Options are "brave", "duckduckgo", "google", "bing", "arxiv", "github".
        num_results (int): The number of search results to return. Default is 10.
    """

    try:
        if isinstance(num_results, str):
            try:
                num_results = int(num_results)
            except Exception as e:
                print(f"Error: {e}")
                print("The number of results should be an integer. Defaulting to 10.")
                num_results = 10

        EngineURL = f"{SEARXNG_URL}/search"
        params = {
            "q": query,
            "engines": ["brave", "duckduckgo", "google", "bing"],
            "format": "json",
            "language": "en",
        }
        try:
            print("Searching the web...\n")
            response = requests.get(EngineURL, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Error in deepSearch: {e}")
            return "Could not get web results."

        results = response.json()['results']
        web_results: list[dict] = []

        for result in results:
            if len(web_results) >= num_results:
                break
            url = result['url']
            title = result['title']
            print()
            print(f"URL: {url}")
            print(f"Title: {title}")
            web_results.append(
                {
                    "title": title,
                    "url": url,
                }
            )
            print(f"Added {title[:10]}... to results.")
        
        return web_results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while searching the web."

## Web Browser Functions
# Open links in web browser
def openBrowser(link: str):
    """
    Open a link in the web browser.

    Args:
        link (str): The link to be opened.
    """

    try:
        webbrowser.open(link)
        return f"Opening {link} in your web browser."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while opening the link."

# YouTube search function
def searchYoutube(query: str):
    """
    Open YouTube in a web browser with the search query.

    Args:
        query (str): Search query for YouTube.
    """

    try:
        base_url = "https://www.youtube.com/results?search_query="
        encoded_query = quote(query)
        search_url = f"{base_url}{encoded_query}"
        webbrowser.open(search_url)
        return f"You can see the results for {query} on your screen."
    except Exception as e:
        print(f"An error occurred while searching YouTube: {e}")
        return "An error occurred while searching YouTube."

# Spotify search function
def searchSpotify(query: str):
    """
    Open Spotify in a web browser with the search query as song or artist name.

    Args:
        query (str): Search query for Spotify.
    """

    try:
        base_url = "https://open.spotify.com/search/"
        encoded_query = quote(query)
        search_url = f"{base_url}{encoded_query}"
        webbrowser.open(search_url)
        return f"You can see the results for {query} on your screen."
    except Exception as e:
        print(f"An error occurred while searching Spotify: {e}")
        return "An error occurred while searching Spotify."


## Clipboard Functions
# Get clipboard text function
def getClipboardText():
    """
    Get the text from the clipboard.
    """

    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        return "No clipboard text to copy."


## Vision Functions
# Screenshot function
def analyseScreen(prompt: str):
    """
    Take a screenshot and analyse the contents.

    Args:
        prompt (str): The prompt to guide the vision model.
    """

    try:
        path = f"{temp_directory}/ss.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ss = ImageGrab.grab()
        ss = ss.convert('RGB')
        ss.save(path, quality=100)

        response = visionPrompt(prompt, path)
        return response
    except Exception as e:
        print(f"An error occurred while taking a screenshot: {e}")
        return "An error occurred while taking a screenshot."

# Webcam capture function
def webcamCapture(prompt: str):
    """
    Capture a frame from the webcam and analyse the contents.

    Args:
        prompt (str): The prompt to guide the vision model.
    """

    try:
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            return "Can't access webcam."
        path = f"{temp_directory}/webcam.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, frame = webcam.read()
        cv2.imwrite(path, frame)
        webcam.release()

        response = visionPrompt(prompt, path)
        return response
    except Exception as e:
        print(f"An error occurred while capturing a webcam image: {e}")
        return "An error occurred while capturing a webcam image."

# Encode image to base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Vision prompt function
def visionPrompt(prompt: str, imgpath: str):
    sys_prompt = (
        'You are a vision analysis AI that provides semantic information from images. '
        'Given the prompt as input, try to extract every bit of information from the image which '
        'is relevant to the prompt and generate as much objective but useful information as possible.'
    )

    prompt = f"{sys_prompt}\n\n{prompt}"
    response = "Error in visionPrompt."

    base64image = encode_image(imgpath)
    try:
        print(f"Using : {VISION_MODEL}")
        client = OpenAI(base_url=VISION_BASE_URL, api_key=VISION_API_KEY)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64image}"
                            },
                        },
                    ],
                },
            ],
        ).choices[0].message.content
    except Exception as e:
        print(f"An error occurred  in visionPrompt: {e}")
    
    os.remove(imgpath)
    return response


## Code Agent
def codeAgent(prompt: str):
    """
    Generate code snippets based on a prompt.
    """
    system_prompt = sysmsg.code_agent_system_prompt.copy()    
    client = OpenAI(base_url=CODE_BASE_URL, api_key=CODE_API_KEY)
    response = client.chat.completions.create(
        model=CODE_MODEL,
        messages=[
            system_prompt,
            {"role": "user", "content": f"{prompt}"},
        ],
        response_format={"type": "json_object"}
    ).choices[0].message.content

    if response is None:
        raise Exception("No response from the code agent.")

    response = sysmsg.Code.model_validate_json(response)

    output = ""
    if thought := response.thought:
        output += f"Thought Process: {thought}\n"
    if file_name := response.filename:
        output += f"Filename: {file_name}\n"
    if language := response.language:
        output += f"Language: {language}\n"
    if code := response.code:
        output += f"Code:\n{code}\n"

    create_file("code/" + file_name, code)

    return output


## File Functions
# Create file
def create_file(file_path: str, content: str) -> str:
    """Create a file
    Args:
        file_path: The path to the file, of the form "directory_name/filename.extension".
        content: The content to write to the file.
    """
    try:
        path = Path(str(working_directory.resolve()) + "/" + file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(file=path, mode="w", encoding="utf-8") as file:
            if not isinstance(content, str):
                content = str(content)
            file.write(content)
        return f"Successfully created/updated {path}."
    except Exception as e:
        print(f"Error in create_file: {e}")
        return "Error creating file."

# Read file contents
def read_file(file_path: str) -> str:
    """Read the contents of a file
    Args:
        file_path: The path to the file, of the form "directory_name/filename.extension".
    """
    try:
        path = str(working_directory.resolve()) + "/" + file_path
        with open(file=path, mode="r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error in read_file: {e}")
        return "Error reading file."

# Clear file contents
def clear_file(file_path: str) -> str:
    """Clear the contents of a file
    Args:
        file_path: The path to the file, of the form "directory_name/filename.extension".
    """
    try:
        path = str(working_directory.resolve()) + "/" + file_path
        with open(file=path, mode="w", encoding="utf-8") as file:
            file.write("")
        return f"Successfully cleared {path}."
    except Exception as e:
        print(f"Error in clear_file: {e}")
        return "Error clearing file."

# Edit file
def edit_file(file_path: str, original_content: str, new_content: str) -> str:
    """Write to a file
    Args:
        file_path: The path to the file, of the form "directory_name/filename.extension".
        original_content: The exact original content of the file
        new_content: The new content to write to the file
    """
    try:
        content = read_file(file_path)
        if original_content in content:
            if len(original_content) > 0:
                content = content.replace(original_content, new_content)
            else:
                content += "\n" + new_content
            return create_file(file_path, content)
        else:
            return "Original content not found in file."
    except Exception as e:
        print(f"Error in edit_file: {e}")
        return "Error editing file."

# Discuss File
def discuss_file(file_path: str, query: str):
    """
    Discuss the contents of a file based on a query.

    Args:
        file_path (str): The path to the file.
        query (str): The query to discuss.
    """

    try:
        content = read_file(file_path)
        client = OpenAI(base_url=SUMMARISATION_BASE_URL, api_key=SUMMARISATION_API_KEY)
        response = client.chat.completions.create(
            model=SUMMARISATION_MODEL,
            messages=[
                sysmsg.file_discussion_system_prompt,                                                      # type: ignore
                {"role": "user", "content": f"File: {file_path}\nQuery: {query}\nContent: {content}"},
            ],
        )

        return str(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred while discussing the file: {e}")
        return "An error occurred while discussing the file."

# List files in directory
def list_files(directory: str = "") -> list[str]:
    """List files in a directory
    Args:
        directory: The directory to list files in
    """
    try:
        path = str(working_directory.resolve()) + "/" + directory
        files = os.listdir(path)
        print(f"Files in {path}: {files}")
        return files
    except Exception as e:
        print(f"Error in list_files: {e}")
        return ["Error listing files."]


# Tool definitions
deep_search_tool = {
    'type': 'function',
    'function': {
        'name': 'deepSearch',
        'description': 'Perform a web search for the given query and get the top result.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The query to search for on the internet.'
                },
                'num_results': {
                    'type': 'integer',
                    'description': f'The number of search results to return. Default value is {NUMBER_OF_URLS_TO_SCRAPE}.'
                },
            },
            'required': ['query', 'num_results']
        }
    }
}

open_browser_tool = {
    'type': 'function',
    'function': {
        'name': 'openBrowser',
        'description': 'Open a link in the web browser.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'link': {
                    'type': 'string',
                    'description': 'The link to open in the web browser.'
                },
            },
            'required': ['link']
        }
    }
}

search_youtube_tool = {
    'type': 'function',
    'function': {
        'name': 'searchYoutube',
        'description': 'Search for music or videos on YouTube based on the given query.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to find videos on YouTube.'
                },
            },
            'required': ['query']
        }
    }
}

search_spotify_tool = {
    'type': 'function',
    'function': {
        'name': 'searchSpotify',
        'description': 'Search for the song name or artist on Spotify based on the given query.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to find music on Spotify.'
                },
            },
            'required': ['query']
        }
    }
}

get_clipboard_text_tool = {
    'type': 'function',
    'function': {
        'name': 'getClipboardText',
        'description': 'Get the mosty recent copied text from the clipboard.',
    },
}

check_internet_connectivity_tool = {
    'type': 'function',
    'function': {
        'name': 'checkInternetConnectivity',
        'description': 'Check if the system is connected to the internet',
    },
}

get_current_date_time_tool = {
    'type': 'function',
    'function': {
        'name': 'getCurrentDateTime',
        'description': 'Get the current date and time in a human-readable format.',
    }
}

get_current_weather_tool = {
    'type': 'function',
    'function': {
        'name': 'getCurrentWeather',
        'description': 'Get the current weather for a city.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The city to get the weather for.'
                },
            },
            'required': ['city']
        }
    }
}

check_screen_contents = {
    'type': 'function',
    'function': {
        'name': 'analyseScreen',
        'description': 'Take a screenshot and analyze the contents of the user\'s screen.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'prompt': {
                    'type': 'string',
                    'description': 'A prompt to guide the vision model in analyzing the screenshot.'
                },
            },
            'required': ['prompt']
        }
    }
}

webcam_capture_tool = {
    'type': 'function',
    'function': {
        'name': 'webcamCapture',
        'description': 'Capture an image from the webcam and analyze the contents.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'prompt': {
                    'type': 'string',
                    'description': 'A prompt to guide the vision model in analyzing the webcam image.'
                },
            },
            'required': ['prompt']
        }
    }
}

create_file_tool = {
    'type': 'function',
    'function': {
        'name': 'create_file',
        'description': 'Create a file with the given content.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The path to the file to create, of the form "directory/filename.extension".'
                },
                'content': {
                    'type': 'string',
                    'description': 'The content to write to the file.'
                },
            },
            'required': ['file_path', 'content']
        }
    }
}

read_file_tool = {
    'type': 'function',
    'function': {
        'name': 'read_file',
        'description': 'Read the contents of a file.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The path to the file to read, of the form "directory/filename.extension".'
                },
            },
            'required': ['file_path']
        }
    }
}

clear_file_tool = {
    'type': 'function',
    'function': {
        'name': 'clear_file',
        'description': 'Clear the contents of a file.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The path to the file to clear, of the form "directory/filename.extension".'
                },
            },
            'required': ['file_path']
        }
    }
}

edit_file_tool = {
    'type': 'function',
    'function': {
        'name': 'edit_file',
        'description': 'Edit a file by replacing the original content with the new content.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The path to the file to edit, of the form "directory/filename.extension".'
                },
                'original_content': {
                    'type': 'string',
                    'description': 'The exact original content to replace in the file.'
                },
                'new_content': {
                    'type': 'string',
                    'description': 'The new content to write to the file.'
                },
            },
        }
    }
}

discuss_file_tool = {
    'type': 'function',
    'function': {
        'name': 'discuss_file',
        'description': 'Discuss the contents of a file based on a query.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'file_path': {
                    'type': 'string',
                    'description': 'The path to the file to discuss.'
                },
                'query': {
                    'type': 'string',
                    'description': 'The query to discuss.'
                },
            },
            'required': ['file_path', 'query']
        }
    }
}

list_files_tool = {
    'type': 'function',
    'function': {
        'name': 'list_files',
        'description': 'List the files in a directory.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'directory': {
                    'type': 'string',
                    'description': 'The directory to list the files in, leave empty for the current directory.'
                },
            },
        }
    }
}

code_agent_tool = {
    'type': 'function',
    'function': {
        'name': 'codeAgent',
        'description': 'Generate code for a given user prompt.', 
        'parameters': {
            'type': 'object',
            'properties': {
                'prompt': {
                    'type': 'string',
                    'description': 'The prompt stating the detailed user\'s requirements of what they want the code for.'
                },
            },
            'required': ['prompt']
        }
    }
}


# list of tools
tools_list = [
    check_internet_connectivity_tool, 
    deep_search_tool, 
    search_youtube_tool, 
    search_spotify_tool, 
    check_screen_contents, 
    webcam_capture_tool, 
    code_agent_tool, 
    open_browser_tool, 
    get_current_weather_tool, 
    get_clipboard_text_tool, 
    get_current_date_time_tool, 
    create_file_tool, 
    read_file_tool, 
    clear_file_tool, 
    edit_file_tool, 
    discuss_file_tool, 
    list_files_tool, 
    ]

# List of tools by category
# fs tools
filesystem_tools = [
    create_file_tool, 
    read_file_tool, 
    clear_file_tool, 
    edit_file_tool, 
    discuss_file_tool, 
    list_files_tool,
]

# internet tools
internet_tools = [
    deep_search_tool, 
    check_internet_connectivity_tool, 
    get_current_weather_tool, 
    open_browser_tool, 
    search_youtube_tool, 
    search_spotify_tool, 
]

# vision tools
vision_tools = [
    check_screen_contents,
    webcam_capture_tool,
]

# Dictionary of tools
tools_dict = {
    'checkInternetConnectivity': checkInternetConnectivity,
    'getCurrentDateTime': getCurrentDateTime,
    'getCurrentWeather': getCurrentWeather,
    'deepSearch': deepSearch,
    'openBrowser': openBrowser,
    'searchYoutube': searchYoutube,
    'searchSpotify': searchSpotify,
    'getClipboardText': getClipboardText,
    'analyseScreen': analyseScreen,
    'webcamCapture': webcamCapture,
    'codeAgent': codeAgent,
    'create_file': create_file,
    'read_file': read_file,
    'clear_file': clear_file,
    'edit_file': edit_file,
    'discuss_file': discuss_file,
    'list_files': list_files,
}

if __name__ == "__main__":
    # Example usage
    while True:
        query = input("\nSearch query ('q' to exit): ")
        if query.lower() == 'q':
            break
        print("Searching...")
        answer = deepSearch(query=query, num_results=3)
        print("---")
        print(answer)
        print("---")
        with open("deep_search_report.md", mode="w", encoding="utf-8") as file:
            file.write(answer)
        print("Report saved to deep_search_report.")
