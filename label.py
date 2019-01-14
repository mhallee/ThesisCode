import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import tkinter as tk
import glob
import pandas as pd
import numpy as np

try:
    import Tkinter as Tk
except ImportError:
    import tkinter as Tk

class QAGame(Tk.Tk):
    def __init__(self, df, *args, **kwargs):
        Tk.Tk.__init__(self, *args, **kwargs)
        self.title("Image Labelling")

        self._counter = 0
        self._setup_gui()
        self._df = df
        self._show_image()

    def _setup_gui(self):
        self._instructions = "Press c for cracked, \n  n for not cracked, \n  v for not vague \n p for partial brick, and  \n x for no brick"
        self._label = Tk.Label(text=self._instructions)
        self._label.pack()

        self._photoFile = Image.open(df.loc[self._counter,'File'])
        print("Opening%s" % df.loc[self._counter,'File'])

        self._photo = ImageTk.PhotoImage(self._photoFile)
        self._photoLab = tk.Label(image=self._photo)
        self._photoLab.image = self._photo # keep a reference!
        self._photoLab.pack()

        self.bind("<Key>", self._key)

    def _key(self, event):
        kp = repr(event.char)[1]
        if (kp == 'c' or kp == 'n' or kp == 'p' or kp == 'x' or kp == 'v'):
            print(kp)
            print(self._counter)
            self._df.loc[self._counter, 'Labels'] = kp
            self._counter += 1
            self._show_image()
            df.to_csv('labels.csv')

        elif (kp == 'b'):
            self._counter -= 1
            self._show_image()
        elif (kp == 'e'):
            print(df)


    def _show_image(self):
        self._photoFile = Image.open(df.loc[self._counter,'File'])
        print("Opening%s" % df.loc[self._counter,'File'])
        self._photo = ImageTk.PhotoImage(self._photoFile)
        self._photoLab.configure(image=self._photo)
        self._photoLab.image = self._photo


    def _move_next(self):        
        self._read_answer()
        if len(self._questions) > 0:
            self._show_next_question()
            self._entry_value.set("")
        else:
            self.quit()
            self.destroy()

    def _read_answer(self):
        answer = self._entry_value.get()
        self._answers.append(answer)

    def _button_classification_callback(self, args, class_idx):
        self._classification_callback(args, self._classes[class_idx])
        self.classify_next_plot()

if __name__ == "__main__":
    #compile list of all files
    fileList = []
    for filename in glob.glob('*.jpg'):
        fileList.append(filename)

    #assemble dataframe
    d = {'File':fileList,'Labels':np.zeros(len(fileList))}
    df = pd.DataFrame(data=d)

    #open GUI
    root = QAGame(df)
    root.mainloop()
    print(df)