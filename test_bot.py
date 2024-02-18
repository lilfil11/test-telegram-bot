import pytest
from aiogram.filters import Command
from aiogram.methods import SendMessage

from unittest.mock import MagicMock
from unittest.mock import AsyncMock

from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE

from bot import cmd_start
from bot import cmd_review
from bot import review_feedback
from bot import cmd_info
from bot import cmd_help
from bot import cmd_predict


@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(request_handler=MessageHandler(cmd_start, Command(commands=["start"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    name_mock = MagicMock(return_value='FirstName LastName')
    answer_true = (f'Привет, *{name_mock()}*\!\n\n'
                   f'Данный бот позволяет определить кожную болезнь по фотографии\. В основе работы '
                   f'бота лежит модель машинного обучения — _случайный лес_\. Она обучена на 15000 '
                   f'размеченных фотографиях с кожными болезнями 23 видов\.\n\n'
                   f'Для просмотра команд бота воспользуйтесь /help\.')
    assert answer_message == answer_true


@pytest.mark.asyncio
async def test_cmd_review():
    requester = MockedBot(request_handler=MessageHandler(cmd_review, Command(commands=["review"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/review"))
    answer_message = calls.send_message.fetchone().text
    answer_true = (f'Оставьте отзыв: ')
    assert answer_message == answer_true


@pytest.mark.asyncio
async def test_cmd_help():
    requester = MockedBot(request_handler=MessageHandler(cmd_help, Command(commands=["help"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    answer_message = calls.send_message.fetchone().text
    answer_true = (f'Список команд бота:\n'
                   f'\t\t/start - активировать бота;\n'
                   f'\t\t/help - вывести список команд бота;\n'
                   f'\t\t/info - вывести информацию о работе бота;\n'
                   f'\t\t/review - оставить отзыв сервису;\n'
                   f'\t\t/predict - получить предсказание болезни по изображению.')
    assert answer_message == answer_true


@pytest.mark.asyncio
async def test_cmd_predict():
    requester = MockedBot(request_handler=MessageHandler(cmd_predict, Command(commands=["predict"])))
    requester.add_result_for(SendMessage, ok=True)
    calls = await requester.query(MESSAGE.as_object(text="/predict"))
    answer_message = calls.send_message.fetchone().text
    answer_true = (f'Пришлите фотографию, для которой хотите получить предсказание: ')
    assert answer_message == answer_true


@pytest.mark.asyncio
async def test_cmd_info():
    msg_mock = AsyncMock()
    started_at = 'datetime'
    review_one = []
    review_two = [3, 5]
    review_three = None
    await cmd_info(msg_mock, started_at, review_one)
    await cmd_info(msg_mock, started_at, review_two)
    await cmd_info(msg_mock, started_at, review_three)
    assert True  # проверяем, что функции запускаются с различными списками review


@pytest.mark.asyncio
async def test_review_feedback():
    msg_mock = AsyncMock()
    review = []
    await review_feedback(msg_mock, review)
    assert len(review) == 1  # проверяем, что отзыв добавился в список
