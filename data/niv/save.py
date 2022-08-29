from PIL import Image
import numpy as np

folder = "./raw_single/"

stimuli = []
for i in range(1,28):
    im_frame = Image.open(folder + 'stimulus' + str(i) + '.png')
    im_frame = im_frame.resize((64,64))
    im_frame = im_frame.convert('RGB')
    np_frame = np.array(im_frame.getdata())
    np_frame = np.uint8(np_frame.reshape(64, 64, 3))

    im_frame.show()

    stimuli.append(np_frame)

stimuli = np.array(stimuli)
print(stimuli.shape)

np.save("stimuli.npy", stimuli)


