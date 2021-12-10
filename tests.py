from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_size = "small"

tokenizer = AutoTokenizer.from_pretrained('./output-small')
model = AutoModelForCausalLM.from_pretrained('./model')


def chat(model, tokenizer, trained=True):
    print("type \"q\" to quit. Automatically quits after 5 messages")

    for step in range(99999):
        message = input("MESSAGE: ")
        print(message)

        if message in ["", "q"]:  # if the user doesn't wanna talk
            break

        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens,
        if (trained):
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8,
            )
        else:
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        # pretty print last ouput tokens from bot
        print("Rick: {}".format(
            tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


chat(model, tokenizer)