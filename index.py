import nltk 
import six
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
from discord.ext import commands
from time import sleep
bot = commands.Bot(command_prefix='>')

with open("index.json", 'r') as file:
    data = json.load(file)



words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True)

print(output)

def bagofwords(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return numpy.array(bag)

@bot.event

async def on_message(message):
    print("Start talking")
    m = message.content

    s = m.split("<@599278894989836299> ")[1]
           


            # inp = input("You: ")
            # if inp.lower() == "quit":
            #     break

    res = model.predict([bagofwords(s, words)])

    rese = numpy.argmax(res)
    print(rese)
    tag = labels[rese]
    print(res)
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
        
    resp = random.choice(responses)
    print(resp)
    await message.channel.send(resp)
                # resp = resp.replace(" ", "_")
                # os.system('espeak '+str(resp))

        






async def ping(ctx):
    await ctx.send('pong') 


token = process.env.BOTTOKEN

# @bot.event
# async def on_message(message):
#     m = message.content
#     current = "@Joshua#9661"
#     print(s)
bot.run(token)
