import discord
import json
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer





class MyClient(discord.Client):

    tokenizer = AutoTokenizer.from_pretrained('./output-small')
    model = AutoModelForCausalLM.from_pretrained('./model')
    chat_history_ids = model.generate(
        tokenizer.encode("Hi" + tokenizer.eos_token, return_tensors='pt'),
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3)

    def reset(self):
        self.chat_history_ids = self.model.generate(
            self.tokenizer.encode("Hi" + self.tokenizer.eos_token, return_tensors='pt'),
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3)

    def chat(self, message):
        print(message)

        if message == 'reset':
            self.reset()



        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids],
                                      dim=-1)

        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8,)

        # pretty print last ouput tokens from bot
        return self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    async def on_ready(self):
        # print out information when the bot wakes up

        print('Logged in as')

        print(self.user.name)

        print(self.user.id)

        print('------')

    def query(self, payload):

        response = self.chat(message=payload)
        return response

    async def on_message(self, message):
        if message.author.id == self.user.id:
            return

        payload = message.content
        async with message.channel.typing():
            response = self.query(payload)
        bot_response = response
        if not bot_response:

            if 'error' in response:

                bot_response = '`Error: {}`'.format(response['error'])

            else:

                bot_response = 'Hmm... something is not right.'
            self.reset()

            # send the model's response to the Discord channel

        await message.channel.send(bot_response)


def main():

    client = MyClient()
    with open("TOKEN.txt", "r") as file:
        data = file.read()
    TOKEN = data
    client.run(data)


if __name__ == '__main__':
    main()
