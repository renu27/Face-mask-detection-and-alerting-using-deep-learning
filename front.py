# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:29:14 2021

@author: T M RENUSHREE
"""
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

def webcam():
    import web
def videofile():
    import video
def imagefile():
    import image

def exitapp():
    msgbox=tk.messagebox.askquestion('QUIT','Are you sure you want to Quit',icon='warning')
    if msgbox=='yes':
        window.destroy()
   
window = tk.Tk()
cav=tk.Canvas(window,width=300,height=300)
window.title("Face Mask")
window.geometry("2000x770")
bg=ImageTk.PhotoImage(file="me.jpg",master=window)
window.grid_rowconfigure(0, weight=0)
window.grid_columnconfigure(0, weight=0)
Label1=Label(window,image=bg)
Label1.place(x=0,y=0)
message = tk.Label(window, text="Face Mask Detection And Alerting" ,bg="white"  ,fg="#234c6f"  ,width=50  ,height=3,font=('times', 30, 'bold')) 
message.place(x=200, y=110)
takeImg = tk.Button(window, text="Detect Mask in an Image" ,command=imagefile,fg="#234c6f"  ,bg="white"  ,width=20  ,height=3, activebackground = "silver" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=420)
trainImg = tk.Button(window, text="Detect Mask in an Video", command=videofile  ,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground = "silver" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=420)
trackImg = tk.Button(window, text="Detect Mask in an WebCam" ,command=webcam,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground ="silver",font=('times', 15, ' bold '))
trackImg.place(x=800, y=420)
quitWindow = tk.Button(window, text="Quit", command=exitapp ,fg="#234c6f"  ,bg="white" ,width=20  ,height=3, activebackground ="silver" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=420)
window.mainloop()