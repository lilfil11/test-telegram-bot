from aiogram import Bot, Dispatcher, types
from datetime import datetime

from aiogram import F
from aiogram.filters import Command

import numpy as np
import pickle

from sklearn.decomposition import PCA
from torchvision.transforms import v2
from PIL import Image, ImageOps


dp = Dispatcher()
dp['started_at'] = datetime.now().strftime("%Y-%m-%d %H:%M")
dp['model'] = pickle.load(open('logreg_clf.pkl', 'rb'))
dp['dict_cat'] = {0: 'Acne and Rosacea', 1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 2: 'Atopic Dermatitis',
                  3: 'Bullous Disease', 4: 'Cellulitis Impetigo and other Bacterial Infections', 5: 'Eczema',
                  6: 'Exanthems and Drug Eruptions', 7: 'Hair Loss Photos Alopecia and other Hair Diseases', 8: 'Herpes HPV and other STDs',
                  9: 'Light Diseases and Disorders of Pigmentation', 10: 'Lupus and other Connective Tissue diseases', 11: 'Melanoma Skin Cancer Nevi and Moles',
                  12: 'Nail Fungus and other Nail Disease', 13: 'Poison Ivy Photos and other Contact Dermatitis', 14: 'Psoriasis pictures Lichen Planus and related diseases',
                  15: 'Scabies Lyme Disease and other Infestations and Bites', 16: 'Seborrheic Keratoses and other Benign Tumors', 17: 'Systemic Disease',
                  18: 'Tinea Ringworm Candidiasis and other Fungal Infections', 19: 'Urticaria Hives', 20: 'Vascular Tumors',
                  21: 'Vasculitis', 22: 'Warts Molluscum and other Viral Infections'}


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(f'Привет, *{message.from_user.full_name}*\!\n\n'
                         f'Данный бот позволяет определить кожную болезнь по фотографии\. В основе работы '
                         f'бота лежит модель машинного обучения — _случайный лес_\. Она обучена на 15000 '
                         f'размеченных фотографиях с кожными болезнями 23 видов\.\n\n'
                         f'Для просмотра команд бота воспользуйтесь /help\.')


@dp.message(Command("review"))
async def cmd_review(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="5"),
            types.KeyboardButton(text="4"),
            types.KeyboardButton(text="3"),
            types.KeyboardButton(text="2"),
            types.KeyboardButton(text="1")
        ]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb,
                                         resize_keyboard=True,
                                         input_field_placeholder="Выберите оценку нашему сервису")
    await message.answer(f'Оставьте отзыв: ', reply_markup=keyboard)


@dp.message(F.text.in_({'5', '4', '3', '2', '1'}))
async def review_feedback(message: types.Message, review: list[int]):
    review.append(int(message.text))
    await message.reply("Спасибо за отзыв\!", reply_markup=types.ReplyKeyboardRemove())


@dp.message(Command("info"))
async def cmd_info(message: types.Message, started_at: str, review: list[int]):
    avg_review = round(np.mean(review), 2) if review else 'нет отзывов'
    await message.answer(f'Бот запущен {started_at}\n'
                         f'Средняя оценка сервиса: {avg_review}\n', parse_mode=None)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(text=f'Список команд бота:\n'
                              f'\t\t/start - активировать бота;\n'
                              f'\t\t/help - вывести список команд бота;\n'
                              f'\t\t/info - вывести информацию о работе бота;\n'
                              f'\t\t/review - оставить отзыв сервису;\n'
                              f'\t\t/predict - получить предсказание болезни по изображению.', parse_mode=None)


@dp.message(Command("predict"))
async def cmd_predict(message: types.Message):
    await message.answer(text=f'Пришлите фотографию, для которой хотите получить предсказание: ', parse_mode=None)


@dp.message(F.photo)
async def make_predict(message: types.Message, bot: Bot, model, dict_cat):
    await bot.download(
        message.photo[-1],
        destination=f'photos/{message.photo[-1].file_id}.jpg'
    )

    X = []
    with Image.open(f"photos/{message.photo[-1].file_id}.jpg") as img:
        img = np.array(v2.Resize(size=(256, 256))(ImageOps.grayscale(img)))
        pca = PCA(75)
        img_pca = pca.fit_transform(img)
        X.append(img_pca.flatten())
        X = np.array(X)
    await message.answer(text=f'Ваша болезнь: {dict_cat[model.predict(X)[-1]]}', parse_mode=None)
