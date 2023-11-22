import discord
import os
from dotenv import load_dotenv
from typing import TypeGuard
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
import pickle

load_dotenv()

intents = discord.Intents.all()
client = discord.Client(intents=intents)

with open("tokenizer.pickle", "rb") as handle:
    loaded_tokenizer = pickle.load(handle)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2 * ((precisions * recalls) / (precisions + recalls + K.epsilon()))


keras.utils.get_custom_objects().update(
    {"recall": recall, "precision": precision, "f1": f1}
)

model = keras.models.load_model("pretrained.h5")

X_new = loaded_tokenizer.texts_to_sequences(["a"])
prediction = model.predict(X_new)

print("Test: " + str(prediction))


def is_message_toxic(message: str) -> bool:
    # TODO: Implement toxicity check
    # reads pretrained.h5 and returns a model
    print("Message: " + message)
    X_new = loaded_tokenizer.texts_to_sequences([message])
    prediction = model.predict(X_new)
    if prediction[0][0] > prediction[0][2]:
        print("Toxic")
        return True
    return False


@client.event
async def on_ready():
    print("Logged in as {0.user}".format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if is_message_toxic(message.content):
        role = discord.utils.get(message.guild.roles, name="Muted")
        if role is not None:
            await message.author.add_roles(role)
            await message.channel.send(
                f"{message.author} has been muted due to toxic behavior."
            )
        else:
            await message.channel.send('No "Muted" role found. Please create one.')


def isTokenSet(TOKEN: object) -> TypeGuard[str]:
    return TOKEN is not None


TOKEN = os.getenv("TOKEN")
if isTokenSet(TOKEN):
    client.run(TOKEN)
