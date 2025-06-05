import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predictTumor import predictTumor

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250)) 
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        result = predictTumor(file_path)
        result_label2.config(text="" + result, fg="green" if "No Tumor" in result else "red")

root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("500x500")

result_label1 = tk.Label(root,text="Brain Tumor Detection",font=("Times New Roman",14), fg="white", background="blue",width="30")
result_label1.pack(pady=20)

panel = tk.Label(root)
panel.pack()

upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 12), bg="orange", fg="white")
upload_btn.pack(pady=20)


result_label2 = tk.Label(root, text="", font=("Arial", 14),bg="white")
result_label2.pack(pady=20)

root.mainloop()
