import asyncio
import os
import sys

from dotenv import load_dotenv
from ollama import Client, chat

from sam_config import SAM_personality
from discord_functions.discord_message_helpers import session_chat_cache
from tools.vision.gemma_vision import vision_image_cleanup
from utility_scripts.system_logging import setup_logger
from utility_scripts.utility import split_response

from pathlib import Path

# configure logging
logger = setup_logger(__name__)

load_dotenv()
os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API")

# model settings for easy swapping
sam_model_name = 'SAM'
sam_ollama_model = 'huihui_ai/deepseek-r1-abliterated'
sam_vision_model = 'gemma3'


def sam_create():
    try:
        client = Client()
        response = client.create(
            model=sam_model_name,
            from_=sam_ollama_model,
            system=SAM_personality,
            stream=False,
        )
        # print(f"# Client: {response.status}")
        logger.info(f"# Client: {response.status}")
        return

    except ConnectionError as e:
        logger.error('Ollama is not running!')
        sys.exit(1)  # Exit program with error code 1

    except Exception as e:
        # Catches any other unexpected errors
        logger.error("❌ An unexpected error occurred:", e)
        sys.exit(1)


# === Main Entry Point ===
async def sam_message(message_author_name, message_author_nickname, message_content, image_file=None, message_attachments=None):
    if image_file is None:
        # llm_response = await sam_converse(message_author_name, message_author_nickname, message_content)
        llm_response = await sam_converse(message_content)
    else:
        # llm_response = await sam_converse_image(message_author_name, message_author_nickname, message_content, image_file ,message_attachments)
        llm_response = await sam_converse(message_content)

    cleaned = llm_response.replace("'", "\\'")
    return split_response(cleaned)


async def sam_converse(user_input):
    response = await asyncio.to_thread(
        chat,
        model=sam_model_name,
        messages=(
            [
                {"role": "system", "content": SAM_personality},
                {"role": "user", "content": user_input}
            ]
        ),
        options={
            "num_ctx": 8192
        }
    )

    return response.message.content
