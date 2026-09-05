from llm_sdk import Small_LLM_Model
import numpy as np

def main():

    # model = Small_LLM_Model()
    # output = ""
    # prompt = "give the result of this operation :  199 + 5, extract only the result dont repeat the operation"
    # for i in range(20):
    #     tokens = model.encode(prompt).tolist()[0]
    #     logits = model.get_logits_from_input_ids(tokens)
    #     id = np.argmax(logits)
    #     output += model.decode(id)
    #     prompt += output

    # print(output)
    ...

a = [1]
b = [2,3]
a.append(b)
print(a)

class car:
    car_object = None
    def __new__(cls):
        if not cls.car_object:
            cls.car_object = super().__new__(cls)
            return cls.car_object
        else:
            return cls.car_object

bmw = car()
print(bmw)
merc = car()
print(merc)

