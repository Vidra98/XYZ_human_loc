# Use pillow to save all frames as an animation in a gif file
from PIL import Image
number_of_frame=25
images = [Image.open(f"{n}.png") for n in range(1,number_of_frame)]

images[0].save('ball.gif', save_all=True, append_images=images[1:], duration=500, loop=0)