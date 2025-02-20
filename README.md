# ddpm_materials
Aplication of a denoising diffusion algorithm, with the goal of generating image samples of materials

# Materials
materials = {'E7': 'nematic', '8cb': 'smectic'}

# Small Dataset
We have a small dataset, containing around 600 imagesclassified by the name of the material and it's temperature. Within those images, we have little variety of pixel distributios, because an image of the material makes very small changes along the temperature increasing process, except for the limit temperature, which suddenly changes the pixel distribution of the sample significantly. Knowing that, we will test our UNet, derived from dtransposed and changed so it better suits our purposes. The nematic material has t_max=60 and the smectic has t_max=34.

# Model Architecture

![](https://i.ibb.co/X7fK3TP/unet.png)

# Results:
We trained our model providing the images labeled as their respective material (not informing the temperatures to the model, and not considering t > t_max). These are the generated images:

![](https://i.ibb.co/JdGkSRf/Figure-2.png)

# Future Works:
We think that if we use a bigger dataset, we can have bigger batch sizes which makes the training significantly better, as experienced by previous denoising works. We take that as our next goal, with the reasoning of testing the model performance further.
