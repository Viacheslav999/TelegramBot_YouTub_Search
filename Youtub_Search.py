import os
import asyncio
import logging
import time
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiohttp import ClientTimeout
from aiogram.client.session.aiohttp import AiohttpSession
from subprocess import run
from aiogram.client.telegram import TelegramAPIServer
from yt_dlp import YoutubeDL
from mega import Mega
from dotenv import load_dotenv
import subprocess
from transformers import GPT2Tokenizer, GPT2LMHeadModel

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MEGA_EMAIL = os.getenv('MEGA_EMAIL')
MEGA_PASSWORD = os.getenv('MEGA_PASSWORD')
FFMPEG_PATH = os.getenv('FFMPEG_PATH')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dp = Dispatcher()

# Загружаем модель и токенизатор GPT-2 для генерации ответов
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def save_user_input_to_file(user_input, ai_response):
    data = {"user_input": user_input, "ai_response": ai_response}
    with open("user_conversations.json", "a") as file:
        json.dump(data, file)
        file.write("\n")

def generate_response(user_input: str) -> str:
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    save_user_input_to_file(user_input, response)
    return response

def get_video_resolution(file_path):
    result = subprocess.run(
        [FFMPEG_PATH, '-i', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    for line in result.stderr.splitlines():
        if "Video:" in line:
            resolution = [s for s in line.split() if 'x' in s]
            if resolution:
                return resolution[0]  # Например, 1920x1080
    return None

# Масштабирование видео с сохранением пропорций
def scale_video(input_file, output_file, target_resolution):
    width, height = map(int, target_resolution.split('x'))
    subprocess.run([
        FFMPEG_PATH, '-i', input_file,
        '-vf', f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
        output_file
    ])

# Предопределенные качества
TARGET_QUALITIES = {
    '360p': '640x360',
    '480p': '854x480',
    '720p': '1280x720',
    '1080p': '1920x1080'
}

# Логин и пароль для MEGA
mega = Mega()
m = mega.login(email=MEGA_EMAIL, password=MEGA_PASSWORD)

# Функция для создания inline клавиатуры с качествами
def get_quality_keyboard(qualities):
    if not qualities:
        logger.error("Не переданы доступные качества для видео.")
        return None

    buttons = [
        [InlineKeyboardButton(text=quality, callback_data=f"quality_{idx}")]
        for idx, quality in enumerate(qualities)
    ]
    buttons.append([InlineKeyboardButton(text="❌ Отмена", callback_data="cancel")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard

def process_video_with_aspect_ratio(filename_video, filename_audio, final_video_filename, target_resolution="1920x1080"):
    # Используем ffmpeg для масштабирования видео с сохранением пропорций
    width, height = map(int, target_resolution.split('x'))
    run([
        FFMPEG_PATH, '-i', filename_video, '-i', filename_audio,
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental',
        '-shortest', final_video_filename
    ])

# Класс состояний
class UserState(StatesGroup):
    wait_link_or_query = State()  # Ожидаем ссылку или запрос
    wait_quality_choice = State()  # Ожидаем выбора качества
    wait_ai_query = State()  # Ожидаем запроса для AI

# Команды бота
async def on_start(message: types.Message):
    keyboard = InlineKeyboardMarkup(row_width=2)
    button_1 = InlineKeyboardButton(text="Загрузка видео с YouTube", callback_data="download_video")
    button_2 = InlineKeyboardButton(text="Узнать информацию с помощью AI", callback_data="ask_ai")
    keyboard.add(button_1, button_2)
    await message.answer("👋 Привет! Я помогу тебе с двумя возможностями. Выбери действие:", reply_markup=keyboard)

async def get_video_by_link_or_query(message: types.Message, state: FSMContext):
    await message.answer("📥 Пожалуйста, отправьте ссылку на видео YouTube или запрос по названию/блогеру (например, 'cat videos' или 'tech reviews').")
    await state.set_state(UserState.wait_link_or_query)

async def get_link_or_query(message: types.Message, state: FSMContext):
    query_or_url = message.text.strip()
    await message.answer("⏳ Проверяем запрос, подождите...")

    try:
        # Если это ссылка
        if "youtube.com" in query_or_url or "youtu.be" in query_or_url:
            url = query_or_url
        else:
            # Если это запрос, ищем видео через yt-dlp
            ydl_opts_search = {
                'quiet': True,
                'format': 'best',
                'noplaylist': True,
                'extract_flat': True,  # Это не скачивает видео, а только метаданные
            }
            with YoutubeDL(ydl_opts_search) as ydl:
                info = ydl.extract_info(f"ytsearch:{query_or_url}", download=False)
                url = info['entries'][0]['url']  # Берем первое видео из поиска
                logger.info(f"Найдено видео по запросу: {url}")

        # Получаем информацию о видео
        ydl_opts = {
            'quiet': True,
            'format': 'best',
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = info.get('formats', [])

            available_resolutions = [fmt.get('height') for fmt in formats if fmt.get('height')]
            logger.info(f"Доступные разрешения: {available_resolutions}")

            quality_list = []
            filtered_formats = []

            for fmt in formats:
                height = fmt.get('height')
                if height in [int(k[:-1]) for k in TARGET_QUALITIES.keys()] and height not in [q['height'] for q in filtered_formats]:
                    resolution = f"{height}p"
                    quality_list.append(resolution)
                    filtered_formats.append({'format': fmt, 'height': height})

            if not quality_list:
                await message.answer("❌ Не удалось найти доступные качества.")
                return

            await state.update_data({'formats': filtered_formats, 'url': url})
            keyboard = get_quality_keyboard(quality_list)
            await message.answer("✅ Видео найдено! Выберите качество:", reply_markup=keyboard)
            await state.set_state(UserState.wait_quality_choice)

    except Exception as e:
        logger.error(f'Ошибка: {e}')
        await message.answer(f"❌ Ошибка при получении видео: {e}")

async def ai_query_handler(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Пожалуйста, задайте вопрос ИИ.")
    await state.set_state(UserState.wait_ai_query)

async def handle_ai_query(message: types.Message, state: FSMContext):
    user_input = message.text
    response = generate_response(user_input)  # Генерация ответа ИИ
    await message.answer(response)  # Отправка ответа пользователю
    await state.clear()  # Очищаем состояние

async def quality_choice(callback_query: types.CallbackQuery, state: FSMContext):
    if callback_query.data == "cancel":
        await callback_query.message.answer("❌ Операция отменена.")
        await state.clear()
        return

    try:
        choice = int(callback_query.data.split('_')[1])
        data = await state.get_data()
        formats = data.get('formats')
        url = data.get('url')

        if not formats or choice < 0 or choice >= len(formats):
            await callback_query.message.answer("Ошибка при выборе видео. Попробуйте заново.")
            return

        selected_format = formats[choice]['format']
        video_format_id = selected_format['format_id']
        resolution = f"{selected_format['height']}p"
        target_resolution = TARGET_QUALITIES.get(resolution, "1920x1080")
        filename_video = f"videos/{time.time()}_video.mp4"
        filename_audio = f"videos/{time.time()}_audio.mp3"
        ydl_opts_video = {
            'format': video_format_id,
            'outtmpl': filename_video
        }
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': filename_audio
        }

        await state.update_data({'ready': False})
        progress_message = await callback_query.message.answer("⏳ Видео загружается и готовится для отправки...")

        # Параллельная загрузка видео и аудио
        try:
            with YoutubeDL(ydl_opts_video) as ydl:
                ydl.download([url])

            with YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([url])

        except Exception as e:
            logger.error(f"Ошибка при загрузке видео/аудио: {e}")
            await callback_query.message.answer(f"❌ Ошибка при загрузке: {e}")
            return

        # Обрабатываем видео для телефона (например, делаем соотношение сторон 9:16)
        final_video_filename = f"videos/{time.time()}_final.mp4"

        # Преобразование видео под целевое разрешение
        process_video_with_aspect_ratio(filename_video, filename_audio, final_video_filename, target_resolution=target_resolution)

        # Загружаем в Mega
        file = m.upload(final_video_filename)
        public_url = m.get_upload_link(file)

        # Обновляем сообщение, что видео готово
        await callback_query.message.edit_text("🎬 Видео успешно загружено в хранилище! Подождите, оно скоро будет отправлено в Telegram...")

        # Отправляем видео пользователю в Telegram
        try:
            video = FSInputFile(final_video_filename)
            await callback_query.message.answer_video(video, caption="🎬 Вот ваше видео с аудио!", width=1920, height=1080)
        except Exception as e:
            await callback_query.message.answer(f"❌ Ошибка при отправке видео: {e}")
        finally:
            os.remove(final_video_filename)  # Удаление временного файла
            os.remove(filename_video)  # Удаление временного видео
            os.remove(filename_audio)  # Удаление временного аудио
            await state.clear()

    except Exception as e:
        logger.error(f"Ошибка при обработке выбора качества: {e}")
        await callback_query.message.answer("Пожалуйста, отправьте корректный номер варианта.")

# Основная асинхронная функция
async def main():
    timeout = 180000

    # Создание бота
    session = AiohttpSession(
        api=TelegramAPIServer.from_base('http://localhost:8081'),
        timeout=timeout
    )
    bot = Bot(token=BOT_TOKEN, session=session)

    # Создаем хранилище для состояний
    from aiogram.fsm.storage.memory import MemoryStorage
    storage = MemoryStorage()

    # Создание диспетчера
    dp = Dispatcher(storage=storage)
    dp['bot'] = bot

    # Регистрируем обработчики
    dp.message.register(on_start, Command("start"))
    dp.message.register(get_video_by_link_or_query, Command("link"))
    dp.message.register(get_link_or_query, UserState.wait_link_or_query)
    dp.message.register(handle_ai_query, UserState.wait_ai_query)
    dp.callback_query.register(quality_choice)
    dp.callback_query.register(ai_query_handler, lambda callback_query: callback_query.data == 'ask_ai')
    dp.callback_query.register(get_video_by_link_or_query, lambda callback_query: callback_query.data == 'download_video')

    # Запуск поллинга
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске поллинга: {e}")

# Запуск основного цикла
if __name__ == "__main__":
    asyncio.run(main())
