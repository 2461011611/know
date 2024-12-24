import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Visualization")

        # UI Elements
        self.label = Label(self.root, text="No Image Loaded")
        self.label.pack()

        self.load_button = Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.mean_button = Button(self.root, text="Mean Filter", command=self.apply_mean_filter, state='disabled')
        self.mean_button.pack()

        self.median_button = Button(self.root, text="Median Filter", command=self.apply_median_filter, state='disabled')
        self.median_button.pack()

        self.gaussian_button = Button(self.root, text="Gaussian Filter", command=self.apply_gaussian_filter, state='disabled')
        self.gaussian_button.pack()

    def load_image(self):
        # Load image file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

            # Enable filter buttons
            self.mean_button.config(state='normal')
            self.median_button.config(state='normal')
            self.gaussian_button.config(state='normal')

    def display_image(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL format
        image_pil = Image.fromarray(image_rgb)

        # Convert to ImageTk format
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update label
        self.label.config(image=image_tk)
        self.label.image = image_tk

    def apply_mean_filter(self):
        filtered_image = cv2.blur(self.image, (5, 5))
        self.display_image(filtered_image)

    def apply_median_filter(self):
        filtered_image = cv2.medianBlur(self.image, 5)
        self.display_image(filtered_image)

    def apply_gaussian_filter(self):
        filtered_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.display_image(filtered_image)

# Main program
if __name__ == "__main__":
    root = Tk()
    app = ImageFilterApp(root)
    root.mainloop()
