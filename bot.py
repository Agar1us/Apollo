import os
import json
import locale
import asyncio
import traceback
import requests
import tempfile
from dotenv import load_dotenv

# aiogram
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.context import FSMContext
from aiogram.client.telegram import TelegramAPIServer
from aiogram.client.session.aiohttp import AiohttpSession

import gigaam
import fire
import pandas as pd
from pydub import AudioSegment
from database import Database
load_dotenv()


asr_model = gigaam.load_model("v2_rnnt")

class Minerva:
    """
    Bot class Minerva.

    """
    def __init__(
        self,
        bot_token: str,
        db_path: str,
        history_max_tokens: int,
    ):
        """
        Bot initiation
        
        Args:
            bot_token (str): Bot token.
            db_path (str): Path to the database.
            history_max_tokens (int): Maximum number of tokens in history - for the future.
        """
        self.default_prompt = 'Ты бот Минерва, полное имя Богиня Минерва. \nТы отвечаешь от лица женского рода. \nТы бот. \nТы говоришь коротко и емко. \nТы была создана в компании Rutube (она же Рутьюб). \nТы работаешь на компанию Rutube (она же Рутьюб). \nТвое предназначение – отвечать на вопросы, помогать людям. \nТы эксперт в сфере сервисов Rutube.'
        self.history_max_tokens = history_max_tokens

        self.db = Database(db_path)

        self.likes_kb = InlineKeyboardBuilder()
        self.likes_kb.add(InlineKeyboardButton(
            text="👍",
            callback_data="feedback:like"
        ))
        self.likes_kb.add(InlineKeyboardButton(
            text="👎",
            callback_data="feedback:dislike"
        ))

        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()

        self.dp.message.register(self.start, Command("start"))
        self.dp.message.register(self.about, Command("about"))
        self.dp.message.register(self.team, Command("team"))
        
        self.dp.message.register(self.generate)
        
        self.dp.callback_query.register(self.save_feedback, F.data.startswith("feedback:"))


    async def start_polling(self):
        """
        Launching the bot.
        """
        await self.dp.start_polling(self.bot)

    async def start(self, message: Message):
        """
        Processing the start command.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await message.reply("Привет! Меня зовут Minerva, как тебе помочь?")
    
    async def about(self, message: Message):
        """
        The about command is a short text about the bot.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("Minerva_tg.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="MINERVA - интеллектуальный помощник оператора службы поддержки Media Wise от команды megamen!"
        )
        
    async def team(self, message: Message):
        """
        Team - a short text about the project team.

        Args:
            message (Message): User message.
        """
        chat_id = message.chat.id
        self.db.create_conv_id(chat_id)
        await self.bot.send_photo(photo=FSInputFile("megamen-team.png"), chat_id=message.chat.id)
        await self.bot.send_message(
            chat_id=message.chat.id,
            text="""Мы, команда megamen, частые участники хакатонов разного уровня. \n\nНаши проекты это: \n• отличное качество \n• высокие метрики \n• классный дизайн \n\nНадеемся, что данный бот вам будет полезен."""
        )
    
    def get_user_name(self, message: Message):
        """
        Retrieving username.

        Args:
            message (Message): User message.

        Returns:
            str: username.
        """
        return message.from_user.full_name if message.from_user.full_name else message.from_user.username

    async def generate(self, message: Message):
        """
        Generates an answer to the user's question.

        Args:
            message (Message): User message.
        """
        user_id = message.from_user.id
        user_name = self.get_user_name(message)
        chat_id = user_id
        conv_id = self.db.get_current_conv_id(chat_id)

        content = await self._build_content(message)
        if content is None:
            await message.answer("Ошибка! Такой тип сообщений пока не поддерживается!")
            return
        if isinstance(content, str):
            self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)
            placeholder = await message.answer("💬")

            try:
                answer = await self.query_api(
                    user_content=content,
                )
                markup = self.likes_kb.as_markup()
                new_message = await placeholder.edit_text(answer, parse_mode=None, reply_markup=markup)

                self.db.save_assistant_message(
                    content=answer,
                    conv_id=conv_id,
                    message_id=new_message.message_id,
                )

            except Exception:
                traceback.print_exc()
                await placeholder.edit_text("Что-то пошло не так")
        else:
            placeholder = await message.answer("💬")
            results = []
            for el in content:
                try:
                    answer = await self.query_api(
                        user_content=el,
                    )
                    results.append(answer)
                except Exception:
                    traceback.print_exc()
                    results.append('')
            df = pd.DataFrame({'Question': content, 'Answer': results})
            os.makedirs(str(message.chat.id), exist_ok=True)
            file_path = os.path.join(str(message.chat.id), 'results.csv')
            df.to_csv(file_path)
            await self.bot.delete_message(chat_id=message.chat.id, message_id=placeholder.message_id)
            await self.bot.send_document(document=FSInputFile(file_path), chat_id=message.chat.id)

    async def save_feedback(self, callback: CallbackQuery):
        """
        Processing feedback (👍 or 👎).

        Args:
            callback (CallbackQuery): feedback.
        """
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.split(":")[1]
        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )

    @staticmethod
    def _merge_messages(messages):
        """
        Message merge function.

        Args:
            messages (list): List of messages.

        Returns:
            list: Combined list of messages.
        """
        new_messages = []
        prev_role = None
        for m in messages:
            content = m["text"]
            role = m["role"]
            if content is None:
                continue
            if role == prev_role:
                is_current_str = isinstance(content, str)
                is_prev_str = isinstance(new_messages[-1]["text"], str)
                if is_current_str and is_prev_str:
                    new_messages[-1]["text"] += "\n\n" + content
                    continue
            prev_role = role
            new_messages.append(m)
        return new_messages


    async def query_api(self, user_content):
        """
        Query to the generation model.

        Args:
            user_content (str): User message content.

        Returns:
            str: Model response.
        """
        questions = {'question': user_content}
        try:
            responce = requests.post('http://127.0.0.1:9875/send/', json=questions)
        except:
            responce = ''

        if responce:
            return json.loads(responce.text)['answer']
        else:
            return 'Что-то не так, ответить не могу! \n(Напишите тех. поддержке @Agar1us)'

    async def _build_content(self, message: Message):
        """
        Content construction.

        Args:
            message (Message): User message.

        Returns:
            str: Final answer.
        """
        content_type = message.content_type
        if content_type == "text":
            return message.text
        elif content_type == "voice":
            voice = message.voice
            voice_file = await message.bot.get_file(voice.file_id)
            input_ogg_file = f"temp_voice_{voice.file_id}.ogg"
            output_wav_file = f"temp_voice_{voice.file_id}.wav"
            await message.bot.download_file(voice_file.file_path, input_ogg_file)
            try:
                self.convert_to_wav(input_ogg_file, output_wav_file)
                transcription = asr_model.transcribe(output_wav_file)
                return transcription
            finally:
                if os.path.exists(input_ogg_file):
                    os.remove(input_ogg_file)
                if os.path.exists(output_wav_file):
                    os.remove(output_wav_file)
        elif content_type == 'document':
            document = message.document
            file_name = document.file_name
            file_extension = file_name.split('.')[-1].lower()

            # Скачиваем файл
            file_path = f"temp_{document.file_id}.{file_extension}"
            file_info = await message.bot.get_file(document.file_id)
            await message.bot.download_file(file_info.file_path, file_path)
            try:
                # Обрабатываем CSV файл
                if file_extension == "csv":
                    df = pd.read_csv(file_path)
                    return df['title'].to_list()  # Преобразуем DataFrame в строку для отправки

                # Обрабатываем Excel файл
                elif file_extension in ["xls", "xlsx"]:
                    df = pd.read_excel(file_path)
                    return df.to_string()  # Преобразуем DataFrame в строку для отправки

                else:
                    return f"Unsupported file format: {file_extension}."
            
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            return None


    def convert_to_wav(self, input_file: str, output_file: str):
        """
        Converts an audio file to 16000 Hz mono WAV format.

        Args:
            input_file (str): Path to the input voice file (e.g., Ogg file).
            output_file (str): Path to save the converted WAV file.

        Returns:
            None
        """
        try:
            audio = AudioSegment.from_file(input_file)
            # Convert to mono and set frame rate to 16000 Hz
            audio = audio.set_frame_rate(16000).set_channels(1)
            # Export the audio as WAV
            audio.export(output_file, format="wav")
        except Exception as e:
            print(f"Error during conversion: {e}")
            raise


def main(
    bot_token: str,
    db_path: str,
    history_max_tokens: int = 4500,
) -> None:
    
    bot = Minerva(
        bot_token=bot_token,
        db_path=db_path,
        history_max_tokens=history_max_tokens,
    )
    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    fire.Fire(main)
    # print(torch.__version__)