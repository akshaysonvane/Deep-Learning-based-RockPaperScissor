import tkinter as tk
import redis
from PIL import Image, ImageTk
import random

r = redis.StrictRedis(host='localhost', port=6379, db=0)

###########################
count_num = 4
data = ""
###########################


def show_frame():
    global label_winner
    label_winner.config(text="")
    label_result.config(text="")

    def count():
        global label_countdown, count_num, label_computer_img, img_rock, img_paper, img_scissor
        count_num -= 1
        label_countdown.config(text=str(count_num))
        if count_num > 0:
            label_countdown.after(1000, count)
        else:
            count_num = 4
            label_countdown.config(text="")

            gesture_list = [1, 2, 3]
            random.shuffle(gesture_list)
            random_number = gesture_list[0]
            if random_number == 1:
                label_computer_img.config(image=img_rock)
            elif random_number == 2:
                label_computer_img.config(image=img_paper)
            else:
                label_computer_img.config(image=img_scissor)

            determine_winner(random_number)

    count()


def determine_winner(computer_gesture):
    global label_winner, data
    if computer_gesture == 1 and data == "PAPER":
        label_winner.config(text="You Win!", fg="green")
        label_result.config(text="Paper beats Rock")
    elif computer_gesture == 1 and data == "SCISSOR":
        label_winner.config(text="You Lose!", fg="red")
        label_result.config(text="Rock beats Scissor")
    elif computer_gesture == 1 and data == "ROCK":
        label_winner.config(text="Tie!", fg="blue")
    elif computer_gesture == 2 and data == "PAPER":
        label_winner.config(text="Tie!", fg="blue")
    elif computer_gesture == 2 and data == "SCISSOR":
        label_winner.config(text="You Win!", fg="green")
        label_result.config(text="Scissor beats Paper")
    elif computer_gesture == 2 and data == "ROCK":
        label_winner.config(text="You Lose!", fg="red")
        label_result.config(text="Paper beats Rock")
    elif computer_gesture == 3 and data == "PAPER":
        label_winner.config(text="You Lose!", fg="red")
        label_result.config(text="Scissor beats Paper")
    elif computer_gesture == 3 and data == "SCISSOR":
        label_winner.config(text="Tie!", fg="blue")
    elif computer_gesture == 3 and data == "ROCK":
        label_winner.config(text="You Win!", fg="green")
        label_result.config(text="Rock beats Scissor")
    else:
        label_winner.config(text="Invalid", fg="black")


def read_gesture():
    global label_countdown, count_num, label_you_img, img_rock, img_paper, img_scissor, img_nothing, data
    gesture = r.get('gesture')
    data = gesture.decode("utf-8")
    if data == "ROCK":
        label_you_img.config(image=img_rock)
    elif data == "PAPER":
        label_you_img.config(image=img_paper)
    elif data == "SCISSOR":
        label_you_img.config(image=img_scissor)
    else:
        label_you_img.config(image=img_nothing)
    label_you_img.after(10, read_gesture)


root = tk.Tk()

# Images ##############################
img_nothing = ImageTk.PhotoImage(Image.open("images/nothing.jpg"))
img_rock = ImageTk.PhotoImage(Image.open("images/rock.jpg"))
img_paper = ImageTk.PhotoImage(Image.open("images/paper.jpg"))
img_scissor = ImageTk.PhotoImage(Image.open("images/scissor.jpg"))
img_rock_small = ImageTk.PhotoImage(Image.open("images/rock_small.jpg"))
img_paper_small = ImageTk.PhotoImage(Image.open("images/paper_small.jpg"))
img_scissor_small = ImageTk.PhotoImage(Image.open("images/scissor_small.jpg"))
#######################################

root.title("Rock Paper Scissor")

label_1 = tk.Label(root, text='Rock Paper Scissor', font=('comic sans', 30)).grid(row=0, column=2)

button_ok = tk.Button(root, text="Play Game", width=25, command=show_frame).grid(row=1, column=2)

label_you = tk.Label(root, text="You", font=('comic sans', 20))
label_you.grid(row=2, column=0, pady=20)

label_computer = tk.Label(root, text="Computer", font=('comic sans', 20))
label_computer.grid(row=2, column=3, pady=20)

label_you_img = tk.Label(root, image=img_paper)
label_you_img.grid(row=3, column=0, sticky='W')

label_computer_img = tk.Label(root, image=img_nothing)
label_computer_img.grid(row=3, column=3, sticky='E')

label_countdown = tk.Label(root, text="", font=('comic sans', 30))
label_countdown.grid(row=3, column=2)

label_winner = tk.Label(root, font=('comic sans', 30))
label_winner.grid(row=4, column=2)

label_result = tk.Label(root, font=('comic sans', 25))
label_result.grid(row=5, column=2)

read_gesture()

root.geometry("585x400")

root.mainloop()
